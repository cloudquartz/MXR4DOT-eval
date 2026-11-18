"""
core/layers.py
==============
FakeQuant wrapper layers for PyTorch models.

FakeQuantLinear         – wraps nn.Linear
FakeQuantGPT2Conv1D     – wraps transformers Conv1D (weights stored transposed!)
FakeQuantConv           – wraps nn.Conv1d / nn.Conv2d (wav2vec2 feature extractor,
                          wav2vec2 positional conv, ResNet, ViT/DeiT patch embed)
FakeQuantConv1d         – alias of FakeQuantConv (backward compatibility)

Hardware-faithfulness rules every wrapper follows
-------------------------------------------------
1. Blocks run along the dot-product reduction axis K, and the weight row and
   the activation row of the same dot product are cut into IDENTICAL 32-element
   blocks. That is the only blocking an MX dot-product engine can compute.
2. Rows are padded with zeros to a multiple of the block size PER ROW, so a
   block never straddles two rows (two output neurons / two tokens).
   Zero padding does not change any block's amax, so it is what hardware does.
3. Convolutions use one canonical K order for both operands: taps first,
   channels fastest, i.e. NHWC im2col:  k = ((kh * KW) + kw) * Cg + c.
   * If groups == 1 and Cg % block == 0, every block is 32 channels of ONE
     input pixel, so quantising the activation along channels and calling
     F.convNd is exactly equal to the im2col formulation (fast path).
   * Otherwise (first layers with C = 1 or 3, grouped convs) the im2col rows
     are built explicitly and quantised (general path).
4. Quantised values are never rounded again: matmuls/convs run in fp32 with
   autocast disabled and TF32 off (configure_exact_numerics), and the
   quantised weight is stored as bf16 hi + bf16 lo buffers so that
   model.to(torch.bfloat16) cannot round it. hi + lo is exact for any value
   with <= 16 significant bits, which covers MXFP4, MXFP4R (scale gap <= 12),
   MXFP8, M2XFP4; NVFP4 values (fp32 global scale) are reproduced to ~2^-17
   relative error, far below their quantisation error.
"""

from __future__ import annotations

import contextlib
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.quantizer import (
    MXFP4Quantizer,
    fake_quant_mxfp4,
    fake_quant_mxfp4_residual,
    fake_quant_mxfp4_adaptive,
)


# ─────────────────────────────────────────────────────────────────────────────
# Numerics
# ─────────────────────────────────────────────────────────────────────────────

def configure_exact_numerics() -> None:
    """Disable TF32 for fp32 matmuls and convolutions (cuBLAS and cuDNN).

    TF32 keeps only 10 mantissa bits, which would re-round MXFP4R values whose
    two scales are far apart. Call once per process before evaluating.
    """
    # torch >= 2.9 has per-backend fp32_precision settings and may reject mixing
    # them with the legacy allow_tf32 flags, so use exactly one of the two APIs.
    if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _no_autocast(device: torch.device):
    if device.type in ("cuda", "cpu"):
        return torch.autocast(device_type=device.type, enabled=False)
    return contextlib.nullcontext()


def _store_quantised_weight(module: nn.Module, w_q: torch.Tensor) -> None:
    """Store an fp32 quantised weight as bf16 hi (+ bf16 lo only if needed)."""
    w_q = w_q.detach().float()
    hi = w_q.to(torch.bfloat16)
    lo = (w_q - hi.float()).to(torch.bfloat16)
    module.register_buffer("w_hi", hi, persistent=False)
    module.register_buffer("w_lo", lo if bool(torch.any(lo != 0)) else None,
                           persistent=False)


def _load_quantised_weight(module: nn.Module, device: torch.device) -> torch.Tensor:
    w = module.w_hi.to(device=device, dtype=torch.float32)
    if module.w_lo is not None:
        w = w + module.w_lo.to(device=device, dtype=torch.float32)
    return w


# ─────────────────────────────────────────────────────────────────────────────
# Row-wise quantisation helpers (blocks along the LAST dim, padded per row)
# ─────────────────────────────────────────────────────────────────────────────

_PAD_MULTIPLE = 32   # NVFP4 uses 16-element blocks, M2XFP 8-element subgroups:
                     # padding each row to a multiple of 32 serves every mode.


def _rows_padded(x: torch.Tensor, block_size: int) -> Tuple[torch.Tensor, int]:
    """Reshape to [rows, K] and zero-pad K up to a multiple of the block size."""
    k = x.shape[-1]
    mult = math.lcm(block_size, _PAD_MULTIPLE)
    x2 = x.reshape(-1, k)
    pad = (-k) % mult
    if pad:
        x2 = F.pad(x2, (0, pad))
    return x2, k


def _apply_weight_mode(w2: torch.Tensor, weight_mode: str, block_size: int,
                       tensor_amax: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, float]:
    """Quantise a padded [rows, K] tensor. All modes block along the last dim."""
    from core import quantizer as q
    if weight_mode == "mxfp4":
        return q.fake_quant_mxfp4(w2, block_size), 0.0
    if weight_mode in ("mxfp4_residual", "mxfp4_residual_full", "mxfp4_residual_full_w"):
        return q.fake_quant_mxfp4_residual(w2, block_size), 1.0
    if weight_mode.startswith("mxfp4_adaptive_"):
        thresh = float(weight_mode.split("_")[-1])
        return q.fake_quant_mxfp4_adaptive(w2, thresh, block_size)
    if weight_mode == "mxfp8_e4m3":
        return q.fake_quant_mxfp8_e4m3(w2, block_size), 0.0
    if weight_mode == "mxfp8_e5m2":
        return q.fake_quant_mxfp8_e5m2(w2, block_size), 0.0
    if weight_mode == "mxfp8_e4m3_residual":
        return q.fake_quant_mxfp8_e4m3_residual(w2, block_size), 1.0
    if weight_mode == "mxfp8_e5m2_residual":
        return q.fake_quant_mxfp8_e5m2_residual(w2, block_size), 1.0
    if weight_mode.startswith("mxfp8_e4m3_adaptive_"):
        thresh = float(weight_mode.split("_")[-1])
        return q.fake_quant_mxfp8_e4m3_adaptive(w2, thresh, block_size)
    if weight_mode.startswith("mxfp8_e5m2_adaptive_"):
        thresh = float(weight_mode.split("_")[-1])
        return q.fake_quant_mxfp8_e5m2_adaptive(w2, thresh, block_size)
    if weight_mode == "nvfp4":
        return q.fake_quant_nvfp4(w2, block_size=16, tensor_amax=tensor_amax), 0.0
    if weight_mode == "m2xfp":
        return q.fake_quant_m2xfp_weight(w2, block_size), 0.0
    raise ValueError(f"Unknown weight quant mode: {weight_mode}")


def _apply_act_mode(x2: torch.Tensor, act_mode: str, block_size: int,
                    tensor_amax: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, float]:
    from core import quantizer as q
    if act_mode == "mxfp4":
        return q.fake_quant_mxfp4(x2, block_size), 0.0
    if act_mode in ("mxfp4_residual", "mxfp4_residual_full", "mxfp4_residual_full_a"):
        return q.fake_quant_mxfp4_residual(x2, block_size), 1.0
    if act_mode.startswith("mxfp4_adaptive_"):
        thresh = float(act_mode.split("_")[-1])
        return q.fake_quant_mxfp4_adaptive(x2, sqnr_thresh_db=thresh, block_size=block_size)
    if act_mode == "mxfp8_e4m3":
        return q.fake_quant_mxfp8_e4m3(x2, block_size), 0.0
    if act_mode == "mxfp8_e5m2":
        return q.fake_quant_mxfp8_e5m2(x2, block_size), 0.0
    if act_mode == "mxfp8_e4m3_residual":
        return q.fake_quant_mxfp8_e4m3_residual(x2, block_size), 1.0
    if act_mode == "mxfp8_e5m2_residual":
        return q.fake_quant_mxfp8_e5m2_residual(x2, block_size), 1.0
    if act_mode.startswith("mxfp8_e4m3_adaptive_"):
        thresh = float(act_mode.split("_")[-1])
        return q.fake_quant_mxfp8_e4m3_adaptive(x2, sqnr_thresh_db=thresh, block_size=block_size)
    if act_mode.startswith("mxfp8_e5m2_adaptive_"):
        thresh = float(act_mode.split("_")[-1])
        return q.fake_quant_mxfp8_e5m2_adaptive(x2, sqnr_thresh_db=thresh, block_size=block_size)
    if act_mode == "nvfp4":
        return q.fake_quant_nvfp4(x2, block_size=16, tensor_amax=tensor_amax), 0.0
    if act_mode == "m2xfp":
        return q.fake_quant_m2xfp_act(x2, block_size), 0.0
    raise ValueError(f"Unknown activation quant mode: {act_mode}")


_PASSTHROUGH = ("fp32", "bf16", "none")

# Weight elements quantised per call (see _quantise_weight).
_WEIGHT_CHUNK_ELEMS = 1 << 22


def _quantise_weight(
    weight: torch.Tensor,
    weight_mode: str,
    block_size: int = 32,
) -> Tuple[torch.Tensor, float]:
    """Quantise a weight whose LAST dim is the reduction axis K (nn.Linear layout
    [out, in], or a conv weight already flattened to [out, K]).
    Returns (fp32 quantised weight with the input's shape, trigger_rate)."""
    if weight_mode in _PASSTHROUGH:
        return weight, 0.0
    w2, k = _rows_padded(weight.float(), block_size)
    rows, kp = w2.shape
    # Quantise in row chunks: blocks never span rows, so this is bit-identical
    # to one call, but bounds peak memory (M2XFP's weight search materialises
    # 4 ratios x 3 biases per element - a 14336x4096 MLP weight in one call
    # exceeds 16 GB). NVFP4's per-tensor global scale is passed explicitly.
    step = max(1, _WEIGHT_CHUNK_ELEMS // kp)
    amax = w2.abs().max() if weight_mode == "nvfp4" else None
    if rows <= step:
        w_q, rate = _apply_weight_mode(w2, weight_mode, block_size, amax)
    else:
        parts, rate_sum = [], 0.0
        for r0 in range(0, rows, step):
            part, r = _apply_weight_mode(w2[r0:r0 + step], weight_mode, block_size, amax)
            parts.append(part)
            rate_sum += r * part.shape[0]
        w_q, rate = torch.cat(parts, dim=0), rate_sum / rows
    return w_q[:, :k].reshape(weight.shape), rate


def _quantise_activation(
    x: torch.Tensor,
    act_mode: str,
    block_size: int = 32,
    tensor_amax: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """Quantise activations in blocks along the LAST dim (the reduction axis),
    padding each row separately. Returns (fp32 tensor, trigger_rate).
    tensor_amax: NVFP4 only - the whole tensor's amax when x is a slice."""
    if act_mode in _PASSTHROUGH:
        return x, 0.0
    x2, k = _rows_padded(x.float(), block_size)
    out, rate = _apply_act_mode(x2, act_mode, block_size, tensor_amax)
    return out[:, :k].reshape(x.shape), rate


# ─────────────────────────────────────────────────────────────────────────────
# FakeQuantLinear
# ─────────────────────────────────────────────────────────────────────────────

class FakeQuantLinear(nn.Module):
    """
    Drop-in fake-quant replacement for nn.Linear.
    Weight layout [out_features, in_features]; blocks along in_features.
    """

    def __init__(
        self,
        original: nn.Linear,
        weight_mode: str = "mxfp4",
        act_mode: str = "mxfp4",
        block_size: int = 32,
    ):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.weight_mode = weight_mode
        self.act_mode = act_mode
        self.block_size = block_size
        self.bias = (nn.Parameter(original.bias.data.clone())
                     if original.bias is not None else None)

        self.last_act_trigger_rate: float = 0.0
        w_q, rate = _quantise_weight(original.weight.data, weight_mode, block_size)
        self.last_weight_trigger_rate = rate
        _store_quantised_weight(self, w_q)

    @property
    def weight(self) -> torch.Tensor:
        """Quantised weight in fp32 (read-only view for diagnostics)."""
        return _load_quantised_weight(self, self.w_hi.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with _no_autocast(x.device):
            x_q, act_rate = _quantise_activation(x.float(), self.act_mode, self.block_size)
            self.last_act_trigger_rate = act_rate
            w = _load_quantised_weight(self, x.device)
            b = self.bias.float() if self.bias is not None else None
            out = F.linear(x_q.float(), w, b)
        return out.to(x.dtype)

    @classmethod
    def from_linear(cls, module, weight_mode="mxfp4", act_mode="mxfp4", block_size=32):
        return cls(module, weight_mode, act_mode, block_size)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"w={self.weight_mode}, a={self.act_mode}")


# ─────────────────────────────────────────────────────────────────────────────
# FakeQuantGPT2Conv1D
# ─────────────────────────────────────────────────────────────────────────────

class FakeQuantGPT2Conv1D(nn.Module):
    """
    Drop-in fake-quant replacement for HuggingFace GPT-2's Conv1D layer.

    HF Conv1D stores weight as [in_features, out_features] and computes
    x @ weight + bias. We quantise the transpose [out, in] (blocks along
    in_features) and keep it in that layout, so the forward is F.linear.
    """

    def __init__(
        self,
        original,  # transformers.pytorch_utils.Conv1D
        weight_mode: str = "mxfp4",
        act_mode: str = "mxfp4",
        block_size: int = 32,
    ):
        super().__init__()
        self.in_features = original.weight.shape[0]
        self.out_features = original.weight.shape[1]
        self.weight_mode = weight_mode
        self.act_mode = act_mode
        self.block_size = block_size
        self.bias = (nn.Parameter(original.bias.data.clone())
                     if original.bias is not None else None)

        self.last_act_trigger_rate: float = 0.0
        w_t = original.weight.data.t().contiguous()          # [out, in]
        w_q, rate = _quantise_weight(w_t, weight_mode, block_size)
        self.last_weight_trigger_rate = rate
        _store_quantised_weight(self, w_q)                    # stored [out, in]

    @property
    def weight(self) -> torch.Tensor:
        """Quantised weight in HF Conv1D layout [in, out], fp32."""
        return _load_quantised_weight(self, self.w_hi.device).t()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with _no_autocast(x.device):
            x_q, act_rate = _quantise_activation(x.float(), self.act_mode, self.block_size)
            self.last_act_trigger_rate = act_rate
            w = _load_quantised_weight(self, x.device)
            b = self.bias.float() if self.bias is not None else None
            out = F.linear(x_q.float(), w, b)
        return out.to(x.dtype)

    @classmethod
    def from_conv1d(cls, module, weight_mode="mxfp4", act_mode="mxfp4", block_size=32):
        return cls(module, weight_mode, act_mode, block_size)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"w={self.weight_mode}, a={self.act_mode} [Conv1D layout]")


# ─────────────────────────────────────────────────────────────────────────────
# FakeQuantConv (nn.Conv1d and nn.Conv2d)
# ─────────────────────────────────────────────────────────────────────────────

def _to_canonical_k(w: torch.Tensor) -> torch.Tensor:
    """Conv weight [O, Cg, *k] -> [O, K] with K = (taps, channels-fastest)."""
    o, cg = w.shape[0], w.shape[1]
    taps = int(math.prod(w.shape[2:]))
    return w.reshape(o, cg, taps).permute(0, 2, 1).reshape(o, taps * cg)


def _from_canonical_k(wk: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    o, cg = shape[0], shape[1]
    taps = int(math.prod(shape[2:]))
    return wk.reshape(o, taps, cg).permute(0, 2, 1).reshape(shape)


class FakeQuantConv(nn.Module):
    """
    Fake-quant replacement for nn.Conv1d / nn.Conv2d (zero padding only).

    Both operands are blocked along the canonical im2col K order (taps, then
    channels fastest). See the module docstring, rule 3.
    """

    # Upper bound on im2col elements quantised at once in the general path.
    IM2COL_CHUNK_ELEMS = 1 << 24

    def __init__(
        self,
        original: nn.Module,
        weight_mode: str = "mxfp4",
        act_mode: str = "mxfp4",
        block_size: int = 32,
        force_im2col: bool = False,
    ):
        super().__init__()
        if isinstance(original, nn.Conv1d):
            self.nd = 1
        elif isinstance(original, nn.Conv2d):
            self.nd = 2
        else:
            raise TypeError(f"FakeQuantConv supports Conv1d/Conv2d, got {type(original)}")
        if original.padding_mode != "zeros":
            raise NotImplementedError("only zero padding is supported")
        if isinstance(original.padding, str):
            raise NotImplementedError("string padding ('same'/'valid') not supported")

        w = original.weight.data                     # [O, Cg, *k] (computed if weight-normed)
        self.weight_shape = tuple(w.shape)
        self.in_channels = original.in_channels
        self.out_channels = original.out_channels
        self.kernel_size = tuple(original.kernel_size)
        self.stride = tuple(original.stride)
        self.padding = tuple(original.padding)
        self.dilation = tuple(original.dilation)
        self.groups = original.groups
        self.weight_mode = weight_mode
        self.act_mode = act_mode
        self.block_size = block_size
        cg = w.shape[1]
        self.fast = (not force_im2col) and self.groups == 1 and cg % block_size == 0
        self.bias = (nn.Parameter(original.bias.data.clone())
                     if original.bias is not None else None)

        self.last_act_trigger_rate: float = 0.0
        wk = _to_canonical_k(w.float())                       # [O, K]
        wk_q, rate = _quantise_weight(wk, weight_mode, block_size)
        self.last_weight_trigger_rate = rate
        _store_quantised_weight(self, _from_canonical_k(wk_q, w.shape))

    @property
    def weight(self) -> torch.Tensor:
        return _load_quantised_weight(self, self.w_hi.device)

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with _no_autocast(x.device):
            w = _load_quantised_weight(self, x.device)
            b = self.bias.float() if self.bias is not None else None
            if self.fast:
                out = self._forward_fast(x.float(), w, b)
            else:
                out = self._forward_im2col(x.float(), w, b)
        return out.to(x.dtype)

    def _conv(self, x, w, b):
        fn = F.conv1d if self.nd == 1 else F.conv2d
        return fn(x, w, b, self.stride, self.padding, self.dilation, self.groups)

    def _forward_fast(self, x, w, b):
        # Channels-last, quantise along channels (32 channels of one pixel per
        # block), back to channels-first. Equal to the im2col formulation.
        x_cl = x.movedim(1, -1)
        x_q, rate = _quantise_activation(x_cl, self.act_mode, self.block_size)
        self.last_act_trigger_rate = rate
        return self._conv(x_q.movedim(-1, 1), w, b)

    def _forward_im2col(self, x, w, b):
        n, c = x.shape[0], x.shape[1]
        g = self.groups
        cg = c // g
        o = self.out_channels
        og = o // g
        if self.nd == 1:
            x4 = x.unsqueeze(2)
            ks, st, pd, dl = (1, self.kernel_size[0]), (1, self.stride[0]), \
                             (0, self.padding[0]), (1, self.dilation[0])
            spatial_in = (x.shape[2],)
        else:
            x4 = x
            ks, st, pd, dl = self.kernel_size, self.stride, self.padding, self.dilation
            spatial_in = tuple(x.shape[2:])
        taps = int(math.prod(self.kernel_size))
        kdim = taps * cg

        cols = F.unfold(x4, kernel_size=ks, dilation=dl, padding=pd, stride=st)  # [N, C*T, L]
        l_out = cols.shape[-1]
        # unfold order is (c, taps); regroup to (group, L, taps, c_in_group)
        cols = cols.view(n, g, cg, taps, l_out).permute(0, 1, 4, 3, 2)          # [N,g,L,T,Cg]

        wk = _to_canonical_k(w).reshape(g, og, kdim)                            # [g, Og, K]
        out = x.new_empty((n, g, og, l_out))
        step = max(1, self.IM2COL_CHUNK_ELEMS // max(1, n * g * kdim))
        # NVFP4's global scale must come from the whole input tensor, not a chunk
        # (im2col columns are copies of x plus zero padding, so amax is equal).
        x_amax = x.abs().max() if self.act_mode == "nvfp4" else None
        rate_sum, chunks = 0.0, 0
        for l0 in range(0, l_out, step):
            l1 = min(l_out, l0 + step)
            xc = cols[:, :, l0:l1].reshape(n, g, l1 - l0, kdim)
            xq, rate = _quantise_activation(xc, self.act_mode, self.block_size, x_amax)
            rate_sum += rate
            chunks += 1
            out[..., l0:l1] = torch.einsum("nglk,gok->ngol", xq, wk)
        self.last_act_trigger_rate = rate_sum / max(1, chunks)

        out = out.reshape(n, o, l_out)
        if b is not None:
            out = out + b.view(1, o, 1)
        spatial_out = [
            (spatial_in[i] + 2 * self.padding[i] - self.dilation[i] * (self.kernel_size[i] - 1) - 1)
            // self.stride[i] + 1
            for i in range(self.nd)
        ]
        return out.reshape(n, o, *spatial_out)

    @classmethod
    def from_conv1d(cls, module, weight_mode="mxfp4", act_mode="mxfp4", block_size=32):
        return cls(module, weight_mode, act_mode, block_size)

    @classmethod
    def from_conv(cls, module, weight_mode="mxfp4", act_mode="mxfp4", block_size=32):
        return cls(module, weight_mode, act_mode, block_size)

    def extra_repr(self) -> str:
        path = "fast" if self.fast else "im2col"
        return (f"{self.nd}d w={self.weight_shape}, groups={self.groups}, "
                f"w={self.weight_mode}, a={self.act_mode}, path={path}")


FakeQuantConv1d = FakeQuantConv   # backward-compatible name
