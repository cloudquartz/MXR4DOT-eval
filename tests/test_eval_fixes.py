"""
tests/test_eval_fixes.py
========================
Checks for the v2 evaluation fixes. Run BEFORE any full evaluation:

    python -m pytest tests/test_eval_fixes.py -v

Everything here runs on CPU in well under a minute and downloads nothing.
(test_sequential_* builds a tiny random Llama and needs `transformers`.)

What each group proves
----------------------
1. Rounding:   the new bucketize-based RNE rounding is bit-identical to the old
               argmin implementation, and the FP6 (E2M3) rounding matches the
               M2XFP reference repo's quantize_to_grid.
2. M2XFP act:  fake_quant_m2xfp_act equals the reference get_quant_mxem
               (re-implemented below from SJTU-ReArch-Group/M2XFP_ASPLOS26,
               mxq/quantize/quant_func.py) wherever the top-1 is unique, and
               picks the lowest index on ties.
3. Blocking:   activation blocks never straddle rows; conv fast path ==
               im2col path; conv output == an independent loop-based
               implementation that cuts weight and activation of every dot
               product into the same canonical (taps, channel) blocks.
4. Exactness:  quantised weights survive model.to(bfloat16) bit-for-bit and
               autocast does not re-round quantised operands.
5. Sequential: layer-by-layer LLM evaluation gives the same NLL as a normal
               forward pass, with and without quantisation.
"""

import copy
import math
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import quantizer as q
from core.layers import (
    FakeQuantConv,
    FakeQuantLinear,
    _load_quantised_weight,
    _quantise_activation,
    _quantise_weight,
    _store_quantised_weight,
)

torch.manual_seed(0)

MODES = ["mxfp4", "mxfp4_residual", "nvfp4", "m2xfp", "mxfp8_e4m3"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Rounding
# ─────────────────────────────────────────────────────────────────────────────

def _special_points_fp4():
    lv = q._MXFP4_LEVELS
    mids = q._MXFP4_MIDPOINTS
    eps = torch.tensor([0.0, 1e-7, -1e-7, 1e-3, -1e-3])
    pts = torch.cat([lv, mids, (mids[:, None] + eps[None, :]).flatten(),
                     torch.tensor([6.5, 7.9, 100.0, 0.0])])
    return pts.clamp(min=0.0)


def test_rounding_matches_legacy():
    x = torch.cat([torch.rand(200_000) * 8.0, torch.rand(50_000) * 0.6,
                   _special_points_fp4()])
    new = q._round_to_mxfp4_levels(x)
    old = q._round_to_mxfp4_levels_legacy(x)
    assert torch.equal(new, old), f"{(new != old).sum()} mismatches"


def test_rounding_ties_are_even():
    mids = q._MXFP4_MIDPOINTS
    got = q._round_to_mxfp4_levels(mids)
    # midpoint i lies between levels i and i+1; RNE picks the even index
    expect = torch.stack([q._MXFP4_LEVELS[i if i % 2 == 0 else i + 1] for i in range(7)])
    assert torch.equal(got, expect)


def _ref_quantize_to_grid(x, levels):
    """Reference repo's quantize_to_grid (CPU copy, unchanged logic)."""
    boundaries = (levels[:-1] + levels[1:]) / 2.0
    odd_boundaries = boundaries[1::2]
    mask = torch.isin(x, odd_boundaries)
    x = x + 0.0000005 * mask
    indices = torch.bucketize(x, boundaries)
    indices.clamp_(0, len(levels) - 1)
    return levels[indices], indices


def test_fp6_rounding_matches_reference():
    lv = q._FP6_E2M3_LEVELS
    mids = q._FP6_E2M3_MIDPOINTS
    x = torch.cat([torch.rand(100_000) * 8.0, lv, mids, mids + 1e-4, mids - 1e-4])
    ours, _ = q._round_to_grid_rne(x, lv, mids)
    ref, _ = _ref_quantize_to_grid(x, lv)
    assert torch.equal(ours, ref), f"{(ours != ref).sum()} mismatches"


def test_old_searchsorted_was_a_ceiling():
    """Documents the M2XFP activation bug that was fixed: searchsorted on the
    level table returns the ceiling, not the nearest level."""
    lv = q._FP6_E2M3_LEVELS
    x = torch.tensor([1.01])                       # nearest is 1.0, ceiling is 1.125
    old = lv[torch.searchsorted(lv, x)]
    new, _ = q._round_to_grid_rne(x, lv, q._FP6_E2M3_MIDPOINTS)
    assert old.item() == 1.125 and new.item() == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. M2XFP activation path vs the reference implementation
# ─────────────────────────────────────────────────────────────────────────────

def _ref_get_quant_mxem(x, group_size=32):
    """Reference get_quant_mxem (CPU copy). Its topk tie order is unspecified,
    so callers only compare subgroups whose top-1 is unique."""
    fp4_grid = q._MXFP4_LEVELS
    fp6_grid = q._FP6_E2M3_LEVELS
    t = x.float().reshape(-1, group_size)
    max_val = t.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    exp = torch.floor(torch.log2(max_val)) - math.floor(math.log2(6.0))
    scales = torch.pow(2, exp)
    v = t / scales
    sign = torch.sign(v)
    fp4, fp4_index = _ref_quantize_to_grid(v.abs(), fp4_grid)
    _, fp6_index = _ref_quantize_to_grid(v.abs(), fp6_grid)
    fp6_index = torch.maximum(torch.minimum(fp6_index, fp4_index * 4 + 2), fp4_index * 4 - 1)
    fp6 = fp6_grid[fp6_index]
    fp4, fp6 = fp4 * sign, fp6 * sign
    tmp = fp4.reshape(-1, 8)
    _, indices = torch.topk(tmp.abs(), 1)
    mask = torch.zeros_like(tmp).scatter_(1, indices, 1).reshape(-1, group_size)
    return ((fp4 * (1 - mask) + fp6 * mask) * scales).reshape(x.shape), tmp


def test_m2xfp_act_matches_reference_where_top1_unique():
    x = torch.randn(4096, 32) * torch.rand(4096, 1) * 10
    ours = q.fake_quant_m2xfp_act(x)
    ref, fp4_sg = _ref_get_quant_mxem(x)
    a = fp4_sg.abs()
    unique = (a == a.max(dim=1, keepdim=True).values).sum(dim=1) == 1        # per subgroup
    unique_elems = unique.repeat_interleave(8).reshape(x.shape)
    assert unique.sum() > 1000, "test data has too few unique-top-1 subgroups"
    assert torch.equal(ours[unique_elems], ref[unique_elems])


def test_m2xfp_act_tie_picks_lowest_index():
    # One subgroup: elements 2 and 5 both quantise to the max FP4 magnitude.
    x = torch.zeros(32)
    x[2], x[5] = 5.6, 5.9          # both -> 6.0 in FP4 at scale 1
    x[8:] = 0.1
    out = q.fake_quant_m2xfp_act(x)
    # element 2 is promoted to FP6 (5.5), element 5 stays FP4 (6.0)
    assert out[2].item() == 5.5 and out[5].item() == 6.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Blocking
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", MODES)
def test_rows_never_straddle(mode):
    x = torch.randn(7, 40) * torch.logspace(-2, 2, 7)[:, None]
    got, _ = _quantise_activation(x, mode)
    for r in range(7):
        row = F.pad(x[r:r + 1], (0, 24))           # each row alone, padded to 64
        if mode == "nvfp4":                        # NVFP4 global scale is per TENSOR
            exp_row = q.fake_quant_nvfp4(row, tensor_amax=x.abs().max())
        else:
            exp_row, _ = _quantise_activation(row, mode)
        assert torch.equal(got[r], exp_row[0, :40]), f"row {r} differs"


@pytest.mark.parametrize("mode", MODES + ["mxfp4_adaptive_18"])
def test_weight_chunking_is_bit_identical(mode, monkeypatch):
    """Row-chunked weight quantisation (memory bound) == one-shot."""
    import core.layers as L
    w = torch.randn(37, 96) * torch.logspace(-2, 2, 37)[:, None]
    monkeypatch.setattr(L, "_WEIGHT_CHUNK_ELEMS", 1 << 30)
    one, r1 = L._quantise_weight(w, mode)
    monkeypatch.setattr(L, "_WEIGHT_CHUNK_ELEMS", 3 * 96)      # 3 rows per chunk
    chunked, r2 = L._quantise_weight(w, mode)
    assert torch.equal(one, chunked)
    assert abs(r1 - r2) < 1e-9


def test_nvfp4_tensor_amax_slice():
    x = torch.randn(64, 32)
    full = q.fake_quant_nvfp4(x)
    part = q.fake_quant_nvfp4(x[:16], tensor_amax=x.abs().max())
    assert torch.equal(full[:16], part)


def _canonical_rows(xpad_patch, cg, taps):
    """patch [Cg, *k] -> vector in (taps, channel) order."""
    return xpad_patch.reshape(cg, taps).t().reshape(-1)


def _loop_reference_conv(conv: nn.Module, x: torch.Tensor, w_mode: str, a_mode: str) -> torch.Tensor:
    """Independent, slow reference: for every output position build the
    canonical K vector of activations and of each weight row, quantise both
    with the same 32-blocks, take the dot product."""
    nd = 1 if isinstance(conv, nn.Conv1d) else 2
    w = conv.weight.data.float()
    o, cg = w.shape[:2]
    ks = w.shape[2:]
    taps = int(math.prod(ks))
    g = conv.groups
    og = o // g
    wk = w.reshape(o, cg, taps).permute(0, 2, 1).reshape(o, taps * cg)
    wq, _ = _quantise_weight(wk, w_mode)
    pad = conv.padding
    xp = F.pad(x.float(), [p for pp in reversed(pad) for p in (pp, pp)])
    if nd == 1:
        lo = (x.shape[2] + 2 * pad[0] - conv.dilation[0] * (ks[0] - 1) - 1) // conv.stride[0] + 1
        out = torch.zeros(x.shape[0], o, lo)
    else:
        ho = (x.shape[2] + 2 * pad[0] - conv.dilation[0] * (ks[0] - 1) - 1) // conv.stride[0] + 1
        wo = (x.shape[3] + 2 * pad[1] - conv.dilation[1] * (ks[1] - 1) - 1) // conv.stride[1] + 1
        out = torch.zeros(x.shape[0], o, ho, wo)
    amax = x.abs().max()
    for n in range(x.shape[0]):
        positions = [(i,) for i in range(out.shape[2])] if nd == 1 else \
                    [(i, j) for i in range(out.shape[2]) for j in range(out.shape[3])]
        for pos in positions:
            for gi in range(g):
                ch = slice(gi * cg, (gi + 1) * cg)
                if nd == 1:
                    s0 = pos[0] * conv.stride[0]
                    idx = torch.arange(ks[0]) * conv.dilation[0] + s0
                    patch = xp[n, ch][:, idx]
                else:
                    r0, c0 = pos[0] * conv.stride[0], pos[1] * conv.stride[1]
                    ri = torch.arange(ks[0]) * conv.dilation[0] + r0
                    ci = torch.arange(ks[1]) * conv.dilation[1] + c0
                    patch = xp[n, ch][:, ri][:, :, ci]
                vec = _canonical_rows(patch, cg, taps)[None, :]
                vq, _ = _quantise_activation(vec, a_mode, 32,
                                             amax if a_mode == "nvfp4" else None)
                rows = wq[gi * og:(gi + 1) * og]
                val = rows @ vq[0]
                if conv.bias is not None:
                    val = val + conv.bias.data[gi * og:(gi + 1) * og]
                out[(n, slice(gi * og, (gi + 1) * og)) + pos] = val
    return out


CONV_CASES = [
    # (name, builder, input shape)
    ("rgb_stem_2d", lambda: nn.Conv2d(3, 8, 7, stride=2, padding=3), (2, 3, 11, 11)),
    ("c64_3x3_2d", lambda: nn.Conv2d(64, 8, 3, stride=1, padding=1), (1, 64, 5, 5)),
    ("c64_1x1_s2_2d", lambda: nn.Conv2d(64, 8, 1, stride=2, padding=0, bias=False), (1, 64, 6, 6)),
    ("wav_c1_1d", lambda: nn.Conv1d(1, 8, 10, stride=5, bias=False), (2, 1, 64)),
    ("c64_k3_1d", lambda: nn.Conv1d(64, 8, 3, stride=2, bias=False), (1, 64, 20)),
    ("grouped_1d", lambda: nn.Conv1d(96, 8, 5, padding=2, groups=2), (1, 96, 12)),
    ("dilated_2d", lambda: nn.Conv2d(32, 4, 3, padding=2, dilation=2), (1, 32, 7, 7)),
]


@pytest.mark.parametrize("name,builder,shape", CONV_CASES, ids=[c[0] for c in CONV_CASES])
@pytest.mark.parametrize("mode", ["mxfp4", "mxfp4_residual", "nvfp4", "m2xfp"])
def test_conv_matches_loop_reference(name, builder, shape, mode):
    torch.manual_seed(1)
    conv = builder()
    x = torch.randn(*shape) * 3
    fq = FakeQuantConv(conv, mode, mode)
    got = fq(x)
    ref = _loop_reference_conv(conv, x, mode, mode)
    assert got.shape == ref.shape
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-5), \
        f"max abs diff {(got - ref).abs().max():.3g} (path={'fast' if fq.fast else 'im2col'})"


@pytest.mark.parametrize("mode", MODES)
def test_conv_fast_equals_im2col(mode):
    torch.manual_seed(2)
    conv = nn.Conv2d(64, 16, 3, stride=2, padding=1)
    x = torch.randn(2, 64, 9, 9)
    fast = FakeQuantConv(conv, mode, mode)
    slow = FakeQuantConv(conv, mode, mode, force_im2col=True)
    assert fast.fast and not slow.fast
    assert torch.allclose(fast(x), slow(x), rtol=1e-5, atol=1e-5)


def test_conv_unquantised_matches_torch():
    conv = nn.Conv2d(3, 8, 3, padding=1)
    x = torch.randn(1, 3, 6, 6)
    fq = FakeQuantConv(conv, "fp32", "fp32")
    # Unquantised fp32 weights are stored to ~16 significant bits (hi + lo).
    assert torch.allclose(fq(x), conv(x), rtol=1e-4, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Exactness
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", MODES)
def test_weight_storage_survives_bf16_cast(mode):
    lin = nn.Linear(96, 16)
    lin.weight.data *= torch.logspace(-3, 3, 16)[:, None]
    fq = FakeQuantLinear(lin, mode, "fp32")
    w_q, _ = _quantise_weight(lin.weight.data, mode)
    before = fq.weight.clone()
    fq.to(torch.bfloat16)
    after = fq.weight
    assert torch.equal(before, after), "bf16 cast changed the stored weight"
    if mode == "nvfp4":     # fp32 global scale: reproduced to < 2^-15 relative
        rel = ((after - w_q).abs() / w_q.abs().clamp(min=1e-30)).max()
        assert rel < 2 ** -15
    else:
        assert torch.equal(after, w_q), "stored weight is not bit-exact"


def test_linear_ignores_autocast():
    lin = nn.Linear(64, 32)
    fq = FakeQuantLinear(lin, "mxfp4_residual", "mxfp4_residual")
    x = torch.randn(4, 64) * 5
    xq, _ = _quantise_activation(x, "mxfp4_residual")
    expect = F.linear(xq, fq.weight, fq.bias.float())
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        got = fq(x)
    assert got.dtype == torch.float32
    assert torch.equal(got, expect)


def test_mxfp4r_value_needs_more_than_bf16():
    """Why exact compute matters. Block = [5.99, 0, ..., 0]:
      primary : scale 2^0, 5.99 -> 6.0
      residual: r = -0.01, scale 2^(floor(log2 0.01) - 2) = 2^-9,
                r / 2^-9 = -5.12 -> -6, so residual = -6 * 2^-9 = -3/256
      MXFP4R value = 6 - 3/256 = 1533/256  (11 significant bits)
    bf16 keeps 8 significant bits, so it rounds this to 6.0."""
    x = torch.zeros(1, 32)
    x[0, 0] = 5.99
    v = q.fake_quant_mxfp4_residual(x)[0, 0]
    assert v.item() == 1533 / 256
    assert v.to(torch.bfloat16).item() == 6.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Sequential (layer-by-layer) LLM evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _tiny_llama():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    torch.manual_seed(3)
    return transformers.LlamaForCausalLM(cfg).eval()


@pytest.mark.parametrize("mode", [None, "mxfp4_residual_act_only", "m2xfp"])
def test_sequential_matches_full(mode):
    from frameworks.language import _replace_layers, _resolve_modes, _full_chunk_nll
    from core.sequential import sequential_chunk_nll

    base = _tiny_llama()
    chunks = [torch.randint(0, 128, (32,)) for _ in range(3)]
    dev = torch.device("cpu")

    full_model = copy.deepcopy(base)
    seq_model = copy.deepcopy(base)
    transform = None
    if mode is not None:
        w, a = _resolve_modes(mode)
        _replace_layers(full_model, w, a, 32, ["lm_head"])
        transform = lambda layer: _replace_layers(layer, w, a, 32, skip_names=[])
    full = _full_chunk_nll(full_model, chunks, dev)
    seq = sequential_chunk_nll(seq_model, chunks, dev, transform, log=lambda s: None)
    assert len(full) == len(seq) == 3
    for f_, s_ in zip(full, seq):
        assert abs(f_ - s_) < 1e-5, (full, seq)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Error-analysis collector
# ─────────────────────────────────────────────────────────────────────────────

def test_error_collector_matches_manual():
    from scripts.run_error_analysis import ErrorCollector, MODES
    torch.manual_seed(5)
    model = nn.Sequential(nn.Linear(64, 32))
    # bf16-representable weights so the fp32 reference path stores them exactly
    model[0].weight.data = model[0].weight.data.bfloat16().float()
    x = torch.randn(10, 64) * 3
    col = ErrorCollector()
    col.attach(model)
    with torch.no_grad():
        model(x)
    s = col.stats["0"]
    w = model[0].weight.data
    y = x @ w.t()
    assert abs(s["den"] - float(y.pow(2).sum())) < 1e-3 * s["den"]
    wq, _ = _quantise_weight(w, "mxfp4")
    xq, _ = _quantise_activation(x, "mxfp4_residual")
    manual = float((xq @ wq.t() - y).pow(2).sum())
    assert abs(s["M:mxfp4_residual_act_only"] - manual) <= 1e-3 * max(manual, 1e-12)
    assert all(f"M:{m}" in s for m in MODES)


def test_error_collector_sequential_runs():
    from scripts.run_error_analysis import ErrorCollector, summarise
    from core.sequential import sequential_chunk_nll
    model = _tiny_llama()
    col = ErrorCollector()
    idx = [0]

    def transform(layer):
        col.drop_cache()
        col.attach(layer, prefix=f"model.layers.{idx[0]}")
        idx[0] += 1
        return layer
    chunks = [torch.randint(0, 128, (32,)) for _ in range(2)]
    sequential_chunk_nll(model, chunks, torch.device("cpu"), transform, log=lambda s: None)
    summ = summarise(col.stats)
    assert summ["n_layers"] == 3 * 7          # 3 layers x (q,k,v,o,gate,up,down)
    assert summ["M:mxfp4_residual"] < summ["M:mxfp4"]


# ─────────────────────────────────────────────────────────────────────────────
# 7. W4A8 mode: mxfp4_w_mxfp8_a = MXFP4 weights, MXFP8-E4M3 activations
# ─────────────────────────────────────────────────────────────────────────────

W4A8 = "mxfp4_w_mxfp8_a"


def test_w4a8_mode_mapping_and_bits():
    from frameworks.language import _resolve_modes
    assert _resolve_modes(W4A8) == ("mxfp4", "mxfp8_e4m3")
    assert q.bits_per_value(W4A8) == 6.25


def _w4a8_build(kind: str, layer: nn.Module, mode: str) -> nn.Module:
    """Wrap a copy of `layer` the way the harnesses do it: mode name ->
    _resolve_modes -> the framework's own layer-replacement function."""
    from frameworks.language import _replace_layers, _resolve_modes
    holder = nn.Sequential(copy.deepcopy(layer))
    w_mode, a_mode = _resolve_modes(mode)
    if kind == "language":
        _replace_layers(holder, w_mode, a_mode, 32, skip_names=[])
    elif kind == "vision":
        from frameworks.vision import replace_vision_layers
        replace_vision_layers(holder, w_mode, a_mode, 32, skip_names=[])
    else:
        from frameworks.speech import _replace_speech_layers
        _replace_speech_layers(holder, w_mode, a_mode, 32, skip_names=[])
    return holder[0]


def _w4a8_forward_recorded(fq: nn.Module, x: torch.Tensor, monkeypatch):
    """Forward `fq` and record the operands of the one matmul / conv it runs.
    Returns (output, op name, activation operand, weight operand)."""
    import core.layers as L
    calls = []

    def recorder(name, real, i_act, i_w):
        def rec(*args, **kwargs):
            calls.append((name, args[i_act].detach().clone(), args[i_w].detach().clone()))
            return real(*args, **kwargs)
        return rec

    with monkeypatch.context() as m:
        m.setattr(L.F, "linear", recorder("linear", F.linear, 0, 1))
        m.setattr(L.F, "conv1d", recorder("conv1d", F.conv1d, 0, 1))
        m.setattr(L.F, "conv2d", recorder("conv2d", F.conv2d, 0, 1))
        m.setattr(L.torch, "einsum", recorder("einsum", torch.einsum, 1, 2))
        with torch.no_grad():
            out = fq(x)
    assert len(calls) == 1, [c[0] for c in calls]
    name, act, w = calls[0]
    return out, name, act, w


def _w4a8_hf_conv1d():
    pytorch_utils = pytest.importorskip("transformers.pytorch_utils")
    c = pytorch_utils.Conv1D(24, 70)                  # nf = out, nx = in
    c.weight.data = torch.randn(70, 24) * 0.7
    c.bias.data = torch.randn(24) * 0.1
    return c


W4A8_CASES = [
    # (id, harness, layer builder, input shape, wrapper class, op that runs)
    ("linear", "language", lambda: nn.Linear(70, 24), (5, 70), "FakeQuantLinear", "linear"),
    ("gpt2_conv1d", "language", _w4a8_hf_conv1d, (2, 5, 70), "FakeQuantGPT2Conv1D", "linear"),
    ("conv2d_fast", "vision", lambda: nn.Conv2d(64, 8, 3, padding=1), (2, 64, 5, 5),
     "FakeQuantConv", "conv2d"),
    ("conv2d_im2col", "vision", lambda: nn.Conv2d(3, 8, 7, stride=2, padding=3), (2, 3, 11, 11),
     "FakeQuantConv", "einsum"),
    ("conv1d_fast", "speech", lambda: nn.Conv1d(64, 8, 3, stride=2, bias=False), (2, 64, 20),
     "FakeQuantConv", "conv1d"),
    ("conv1d_im2col", "speech", lambda: nn.Conv1d(1, 8, 10, stride=5), (2, 1, 64),
     "FakeQuantConv", "einsum"),
]


@pytest.mark.parametrize("name,kind,builder,shape,cls_name,op", W4A8_CASES,
                         ids=[c[0] for c in W4A8_CASES])
def test_w4a8_operands_and_output(name, kind, builder, shape, cls_name, op, monkeypatch):
    """(a) weight == the mxfp4 mode's, (b) activation entering the matmul / conv
    == the mxfp8_e4m3 mode's, (c) output == the op applied to (a) and (b)."""
    torch.manual_seed(11)
    layer = builder()
    x = torch.randn(*shape) * 3

    new = _w4a8_build(kind, layer, W4A8)
    w4 = _w4a8_build(kind, layer, "mxfp4")
    a8 = _w4a8_build(kind, layer, "mxfp8_e4m3")
    for fq in (new, w4, a8):
        assert type(fq).__name__ == cls_name

    out_new, op_new, xq_new, w_new = _w4a8_forward_recorded(new, x, monkeypatch)
    _, op_w4, xq_w4, w_w4 = _w4a8_forward_recorded(w4, x, monkeypatch)
    _, op_a8, xq_a8, w_a8 = _w4a8_forward_recorded(a8, x, monkeypatch)
    assert op_new == op_w4 == op_a8 == op

    # (a) quantised weight: stored buffers, loaded weight and the operand of the op
    assert new.w_hi.dtype == torch.bfloat16 and torch.equal(new.w_hi, w4.w_hi)
    assert (new.w_lo is None) == (w4.w_lo is None)
    if new.w_lo is not None:
        assert torch.equal(new.w_lo, w4.w_lo)
    assert torch.equal(new.weight, w4.weight)
    assert w_new.shape == w_w4.shape and torch.equal(w_new, w_w4)

    # (b) quantised activation that enters the matmul / conv
    assert xq_new.shape == xq_a8.shape and torch.equal(xq_new, xq_a8)

    # (a) and (b) can fail: on this data the two formats give different operands
    assert not torch.equal(w_new, w_a8)
    assert not torch.equal(xq_new, xq_w4)

    # (c) output == the op on the mxfp4 mode's weight and the mxfp8 mode's activation
    b = w4.bias.float() if w4.bias is not None else None
    if op == "linear":
        ref = F.linear(xq_a8, w_w4, b)
    elif op in ("conv1d", "conv2d"):
        ref = getattr(F, op)(xq_a8, w_w4, b, w4.stride, w4.padding, w4.dilation, w4.groups)
    else:                                             # im2col: [N,g,L,K] x [g,Og,K]
        n, o = x.shape[0], w4.out_channels
        ref = torch.einsum("nglk,gok->ngol", xq_a8, w_w4).reshape(n, o, -1)
        if b is not None:
            ref = ref + b.view(1, o, 1)
        ref = ref.reshape(out_new.shape)
    assert out_new.dtype == torch.float32 and out_new.shape == ref.shape
    assert torch.equal(out_new, ref)
