"""
frameworks/language.py
======================
WikiText-2 perplexity harness for GPT-2 / Llama / Mistral.

Protocol (v2):
  - Dataset: Salesforce/wikitext, wikitext-2-raw-v1, test split, joined with
    "\\n\\n" and tokenised once.
  - n_chunks=None (default): ALL non-overlapping seq_len chunks, in order -
    the standard full-test-set protocol. An integer keeps the old behaviour
    (that many chunks picked by a seeded random permutation).
  - seq_len: 1024 for GPT-2 (its context limit), 2048 for the 7B/8B models.
  - Quantisation never rounds a quantised value again (see core/layers.py).
    The rest of the network runs in the model's own dtype (bf16), identical to
    the BF16 reference.

Execution:
  - exec_mode="full": whole model on the GPU (GPT-2).
  - exec_mode="sequential": layer-by-layer (core/sequential.py) for models
    that do not fit in VRAM. "auto" picks sequential above 2 B parameters.

Layer policy: every nn.Linear / HF Conv1D is quantised except lm_head.
"""
from __future__ import annotations

import os as _os
try:
    import certifi as _certifi
    _os.environ.setdefault("SSL_CERT_FILE", _certifi.where())
    _os.environ.setdefault("REQUESTS_CA_BUNDLE", _certifi.where())
except ImportError:
    pass

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

DEFAULT_SKIP_NAMES = ["lm_head"]


# ─────────────────────────────────────────────────────────────────────────────
# Mode → (weight_mode, act_mode) decomposition
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_modes(quant_mode: str) -> Tuple[str, str]:
    """
    Map a high-level quant_mode string to (weight_mode, act_mode).

      fp32 / bf16                  → unquantised
      mxfp4, nvfp4, mxfp8_*, m2xfp → same format on both operands
      mxfp4_residual(_full)        → residue on both operands
      mxfp4_residual_act_only      → w=mxfp4, a=mxfp4_residual   (MXFP4R default)
      mxfp4_residual_weight_only   → w=mxfp4_residual, a=mxfp4
      mxfp4_w_mxfp8_a              → w=mxfp4, a=mxfp8_e4m3       (W4A8, no residue)
      mxfp4_adaptive_<X>           → w=mxfp4, a=adaptive
    """
    if quant_mode in ("fp32", "bf16"):
        return quant_mode, quant_mode
    if quant_mode in ("mxfp4", "nvfp4", "mxfp8_e4m3", "mxfp8_e5m2"):
        return quant_mode, quant_mode
    if quant_mode in ("mxfp4_residual", "mxfp4_residual_full",
                      "mxfp8_e4m3_residual", "mxfp8_e5m2_residual"):
        return quant_mode, quant_mode
    if quant_mode == "mxfp4_residual_act_only":
        return "mxfp4", "mxfp4_residual"
    if quant_mode == "mxfp4_residual_weight_only":
        return "mxfp4_residual", "mxfp4"
    if quant_mode == "mxfp4_w_mxfp8_a":
        return "mxfp4", "mxfp8_e4m3"
    if quant_mode.startswith("mxfp4_adaptive"):
        return "mxfp4", quant_mode
    if quant_mode.startswith("mxfp8_e4m3_adaptive"):
        return "mxfp8_e4m3", quant_mode
    if quant_mode.startswith("mxfp8_e5m2_adaptive"):
        return "mxfp8_e5m2", quant_mode
    if quant_mode == "m2xfp":
        return "m2xfp", "m2xfp"
    raise ValueError(f"Unknown quant_mode: {quant_mode}")


# ─────────────────────────────────────────────────────────────────────────────
# Layer substitution
# ─────────────────────────────────────────────────────────────────────────────

def _replace_layers(
    model: nn.Module,
    weight_mode: str,
    act_mode: str,
    block_size: int = 32,
    skip_names: Optional[List[str]] = None,
) -> nn.Module:
    """Replace Conv1D / Linear with FakeQuant equivalents in place.
    skip_names are dotted paths relative to `model`."""
    from core.layers import FakeQuantLinear, FakeQuantGPT2Conv1D

    skip_set = set(DEFAULT_SKIP_NAMES if skip_names is None else skip_names)
    try:
        from transformers.pytorch_utils import Conv1D as HF_Conv1D
    except ImportError:
        HF_Conv1D = None

    def _recurse(module: nn.Module, prefix: str = ""):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}".lstrip(".")
            if full_name in skip_set:
                continue
            if HF_Conv1D is not None and isinstance(child, HF_Conv1D):
                setattr(module, name,
                        FakeQuantGPT2Conv1D.from_conv1d(child, weight_mode, act_mode, block_size))
            elif isinstance(child, nn.Linear):
                setattr(module, name,
                        FakeQuantLinear.from_linear(child, weight_mode, act_mode, block_size))
            else:
                _recurse(child, full_name)

    _recurse(model)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Dataset + chunking
# ─────────────────────────────────────────────────────────────────────────────

def _load_wikitext_chunks(
    tokenizer,
    seq_len: int = 1024,
    n_chunks: Optional[int] = None,
    seed: int = 42,
) -> List[torch.Tensor]:
    """WikiText-2 test split as non-overlapping seq_len token chunks.
    n_chunks=None → all chunks in order (full test set)."""
    from datasets import load_dataset

    tokenizer.model_max_length = 1_000_000_000
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    token_tensor = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    n_available = token_tensor.numel() // seq_len
    chunks = [token_tensor[i * seq_len:(i + 1) * seq_len] for i in range(n_available)]
    if n_chunks is None or n_chunks >= n_available:
        return chunks
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(chunks), generator=g)
    return [chunks[i] for i in perm[:n_chunks].tolist()]


@torch.no_grad()
def _full_chunk_nll(model: nn.Module, chunks: List[torch.Tensor], device: torch.device) -> List[float]:
    model.eval()
    out: List[float] = []
    for chunk in chunks:
        ids = chunk.unsqueeze(0).to(device)
        out.append(float(model(ids, labels=ids, use_cache=False).loss.item()))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Harness
# ─────────────────────────────────────────────────────────────────────────────

class LanguageEvalHarness:
    """
    Args:
        model_name:  HF model ID
        quant_mode:  see _resolve_modes
        seed:        only used when n_chunks is an integer (random subset)
        n_chunks:    None = full test set (default); int = random subset
        seq_len:     tokens per chunk
        block_size:  MX block size (32)
        exec_mode:   "auto" | "full" | "sequential"
        skip_names:  layers left unquantised (default ["lm_head"])
    """

    GATED_PREFIXES = ("meta-llama/", "mistralai/")

    def __init__(
        self,
        model_name: str = "gpt2",
        quant_mode: str = "fp32",
        seed: int = 42,
        n_chunks: Optional[int] = None,
        seq_len: int = 1024,
        block_size: int = 32,
        device: Optional[torch.device] = None,
        hf_token: Optional[str] = None,
        exec_mode: str = "auto",
        skip_names: Optional[List[str]] = None,
        model_dtype: Optional[str] = None,
    ):
        self.model_name = model_name
        self.quant_mode = quant_mode
        self.seed = seed
        self.n_chunks = n_chunks
        self.seq_len = seq_len
        self.block_size = block_size
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exec_mode = exec_mode
        self.skip_names = list(DEFAULT_SKIP_NAMES if skip_names is None else skip_names)
        # Reference dtype for everything that is not a quantised matmul.
        # fp32 mode -> fp32; every other mode runs in bf16 like the BF16 reference.
        if model_dtype is not None:
            self.dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[model_dtype]
        else:
            self.dtype = torch.float32 if quant_mode == "fp32" else torch.bfloat16
        # Gated models: a token from HF_TOKEN or a cached `hf auth login` both work,
        # so access errors are left to from_pretrained, which reports them clearly.

    def _load_model_and_tokenizer(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from core.compat import hf_dtype_kwargs

        tk = {"token": self.hf_token} if self.hf_token else {}
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tk)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, low_cpu_mem_usage=True, **hf_dtype_kwargs(self.dtype), **tk)
        model.eval()
        return model, tokenizer

    def _pick_exec_mode(self, model) -> str:
        from core.sequential import supports_sequential
        if self.exec_mode != "auto":
            return self.exec_mode
        n_params = sum(p.numel() for p in model.parameters())
        return "sequential" if (n_params > 2e9 and supports_sequential(model)) else "full"

    def run(self) -> Dict[str, Any]:
        from core.layers import configure_exact_numerics
        from core.quantizer import bits_per_value
        from core.sequential import sequential_chunk_nll

        configure_exact_numerics()
        weight_mode, act_mode = _resolve_modes(self.quant_mode)
        quantise = self.quant_mode not in ("fp32", "bf16")

        model, tokenizer = self._load_model_and_tokenizer()
        exec_mode = self._pick_exec_mode(model)
        chunks = _load_wikitext_chunks(tokenizer, self.seq_len, self.n_chunks, self.seed)

        print(f"\n{'=' * 64}\n  {self.model_name} | {self.quant_mode} (w={weight_mode}, a={act_mode})"
              f"\n  chunks={len(chunks)} x {self.seq_len} | exec={exec_mode} | dtype={self.dtype}"
              f" | skip={self.skip_names}\n{'=' * 64}")

        if exec_mode == "sequential":
            # Skip names are top-level paths (lm_head); decoder layers contain
            # none of them, so each layer is quantised whole.
            transform = (lambda layer: _replace_layers(
                layer, weight_mode, act_mode, self.block_size, skip_names=[])) if quantise else None
            nlls = sequential_chunk_nll(model, chunks, self.device, transform)
        else:
            model.to(self.device)
            if quantise:
                _replace_layers(model, weight_mode, act_mode, self.block_size, self.skip_names)
            nlls = _full_chunk_nll(model, chunks, self.device)

        # Equal-length chunks, so the token-weighted mean NLL is the chunk mean.
        ppl = math.exp(sum(nlls) / len(nlls))
        eff_bits = bits_per_value(self.quant_mode)
        print(f"\n  PPL = {ppl:.4f}  (eff_bits={eff_bits:.2f})\n")

        del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "ppl": ppl,
            "per_chunk_nll": nlls,
            "tokens_per_chunk": self.seq_len - 1,
            "n_chunks": len(chunks),
            "seq_len": self.seq_len,
            "exec_mode": exec_mode,
            "model_name": self.model_name,
            "quant_mode": self.quant_mode,
            "weight_mode": weight_mode,
            "act_mode": act_mode,
            "seed": self.seed,
            "skip_names": self.skip_names,
            "eff_bits": eff_bits,
        }
