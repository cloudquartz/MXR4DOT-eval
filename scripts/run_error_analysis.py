"""
run_error_analysis.py
=====================
Per-layer quantisation-error analysis: WHERE does each format's error come
from, and how much of it does each residue placement remove?

    python run_error_analysis.py --config configs/v2/gpt2.yaml      # one model
    python run_error_analysis.py --report                          # all models -> Markdown

For every quantised layer (same layer set and blocking as run_eval.py) and a
small calibration set, it hooks the layer's real input x and computes the
layer output (bias excluded) under many (weight format, activation format)
pairs, all relative to the unquantised output y = W x:

  e_W[f]    = ||W_f x   - W x||^2 / ||W x||^2    weight quantised, activations exact
  e_A[f]    = ||W x_f   - W x||^2 / ||W x||^2    activations quantised, weights exact
  e_mode[m] = ||W_m x_m - W x||^2 / ||W x||^2    the full evaluated mode m

for f in {MXFP4, MXFP4 two-block residue, NVFP4, M2XFP4, MXFP8-E4M3}. Values
are summed over calibration inputs per layer, then averaged over layers in dB
(10 log10 NMSE; more negative = smaller error).

The report answers two questions:
  1. Which operand dominates plain MXFP4's error in each model (e_A vs e_W)?
     Prediction: residue belongs on that operand.
  2. Does that prediction match which residue placement won on accuracy
     (results_v2 JSONs from run_eval.py)?
It also gives the per-format operand-error table.

Calibration data is a small seeded subset (different from nothing - it is
drawn from the same test sets); this is a diagnostic, not an accuracy result.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import time
from pathlib import Path
from typing import Callable, Dict, List

import torch
import torch.nn as nn
import yaml

import sys
HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
OUT_DIR = HERE / "results_v2" / "error_analysis"

OPERAND_FORMATS = ["mxfp4", "mxfp4_residual", "nvfp4", "m2xfp", "mxfp8_e4m3"]
FORMAT_LABEL = {"mxfp4": "MXFP4", "mxfp4_residual": "MXFP4 2-block residue",
                "nvfp4": "NVFP4", "m2xfp": "M2XFP4", "mxfp8_e4m3": "MXFP8-E4M3"}
MODES = {   # evaluated quant_mode -> (weight format, activation format)
    "mxfp4": ("mxfp4", "mxfp4"),
    "nvfp4": ("nvfp4", "nvfp4"),
    "m2xfp": ("m2xfp", "m2xfp"),
    "mxfp4_residual_act_only": ("mxfp4", "mxfp4_residual"),
    "mxfp4_residual_weight_only": ("mxfp4_residual", "mxfp4"),
    "mxfp4_residual": ("mxfp4_residual", "mxfp4_residual"),
    "mxfp8_e4m3": ("mxfp8_e4m3", "mxfp8_e4m3"),
}
DEFAULT_N = {"language": 4, "vision": 64, "speech": 16}


# ─────────────────────────────────────────────────────────────────────────────
# Hooks
# ─────────────────────────────────────────────────────────────────────────────

def _is_target(m: nn.Module) -> bool:
    try:
        from transformers.pytorch_utils import Conv1D as HF_Conv1D
    except ImportError:
        HF_Conv1D = ()
    return isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)) or (
        HF_Conv1D != () and isinstance(m, HF_Conv1D))


def _make_fq(module: nn.Module, w_fmt: str, block_size: int) -> nn.Module:
    from core.layers import FakeQuantConv, FakeQuantGPT2Conv1D, FakeQuantLinear
    if isinstance(module, nn.Linear):
        fq = FakeQuantLinear(module, w_fmt, "fp32", block_size)
    elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
        fq = FakeQuantConv(module, w_fmt, "fp32", block_size)
    else:
        fq = FakeQuantGPT2Conv1D(module, w_fmt, "fp32", block_size)
    fq.bias = None                       # errors are about the matmul, not the bias
    return fq


class ErrorCollector:
    """Attaches forward hooks and accumulates squared errors per layer."""

    def __init__(self, block_size: int = 32):
        self.block_size = block_size
        self.stats: Dict[str, Dict[str, float]] = {}
        self._cache: Dict[str, Dict[str, nn.Module]] = {}
        self._handles = []

    def attach(self, root: nn.Module, prefix: str = "", skip: frozenset = frozenset()):
        for name, m in root.named_modules():
            full = f"{prefix}.{name}".strip(".") if name else prefix
            if not name or name in skip or full in skip or not _is_target(m):
                continue
            self._handles.append(m.register_forward_hook(self._hook(full)))

    def drop_cache(self):
        self._cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _hook(self, name: str) -> Callable:
        @torch.no_grad()
        def hook(mod, inputs, output):
            x = inputs[0].float()
            if name not in self._cache:
                self._cache[name] = {f: _make_fq(mod, f, self.block_size).to(x.device)
                                     for f in ["fp32"] + OPERAND_FORMATS}
            fqs = self._cache[name]

            def run(w, a):
                fq = fqs[w]
                fq.act_mode = a
                return fq(x).float()

            ref = run("fp32", "fp32")
            s = self.stats.setdefault(name, {"den": 0.0, "calls": 0})
            s["den"] += float(ref.pow(2).sum())
            s["calls"] += 1

            def add(key, y):
                s[key] = s.get(key, 0.0) + float((y - ref).pow(2).sum())

            for f in OPERAND_FORMATS:
                add(f"W:{f}", run(f, "fp32"))
                add(f"A:{f}", run("fp32", f))
            for mode, (w, a) in MODES.items():
                add(f"M:{mode}", run(w, a))
        return hook


# ─────────────────────────────────────────────────────────────────────────────
# Per-modality calibration passes
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _collect_language(cfg, n, dev, col: ErrorCollector):
    from frameworks.language import LanguageEvalHarness, _load_wikitext_chunks
    from core.sequential import sequential_chunk_nll, supports_sequential

    h = LanguageEvalHarness(model_name=cfg["model_name"], quant_mode="bf16",
                            seq_len=cfg.get("seq_len", 1024), device=dev)
    model, tok = h._load_model_and_tokenizer()
    chunks = _load_wikitext_chunks(tok, h.seq_len, n_chunks=n, seed=1234)
    n_params = sum(p.numel() for p in model.parameters())
    if n_params > 2e9 and supports_sequential(model):
        idx = [0]

        def transform(layer):
            col.drop_cache()                       # free the previous layer's clones
            col.attach(layer, prefix=f"model.layers.{idx[0]}")
            idx[0] += 1
            return layer
        sequential_chunk_nll(model, chunks, dev, transform, log=lambda s: print(s, flush=True))
    else:
        model.to(dev).eval()
        col.attach(model, skip=frozenset({"lm_head"}))
        for ch in chunks:
            model(ch.unsqueeze(0).to(dev), use_cache=False)


@torch.no_grad()
def _collect_vision(cfg, n, dev, col: ErrorCollector):
    from frameworks.vision import HEAD_NAMES, VisionEvalHarness, _build_model, _collate
    model, transform, _ = _build_model(cfg["model_name"])
    model.to(dev).eval()
    col.attach(model, skip=frozenset(HEAD_NAMES))
    h = VisionEvalHarness(model_name=cfg["model_name"], n_images=n, seed=1234)
    loader = torch.utils.data.DataLoader(h._dataset(transform), batch_size=32,
                                         num_workers=4, collate_fn=_collate)
    for x, _y in loader:
        model(x.to(dev))


@torch.no_grad()
def _collect_speech(cfg, n, dev, col: ErrorCollector):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    from core.compat import hf_dtype_kwargs
    from frameworks.speech import _decode_audio, _load_librispeech
    proc = Wav2Vec2Processor.from_pretrained(cfg["model_name"])
    model = Wav2Vec2ForCTC.from_pretrained(cfg["model_name"], **hf_dtype_kwargs(torch.float32))
    model.to(dev).eval()
    col.attach(model, skip=frozenset({"lm_head"}))
    for sample in _load_librispeech(n, seed=1234, config=cfg.get("librispeech_config", "clean")):
        audio, sr = _decode_audio(sample["audio"])
        inp = proc(audio, sampling_rate=sr, return_tensors="pt")
        model(inp.input_values.to(dev, dtype=torch.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Summaries
# ─────────────────────────────────────────────────────────────────────────────

def _db(x: float) -> float:
    return 10.0 * math.log10(max(x, 1e-30))


def summarise(stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Mean over layers of per-layer NMSE in dB, for every key."""
    keys = [k for k in next(iter(stats.values())) if ":" in k]
    out = {}
    for k in keys:
        vals = [_db(s[k] / s["den"]) for s in stats.values() if s["den"] > 0]
        out[k] = sum(vals) / len(vals)
    out["n_layers"] = len(stats)
    return out


def run_model(config_path: str, n: int = None):
    from core.compat import environment_info
    from core.layers import configure_exact_numerics

    configure_exact_numerics()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    modality = cfg["modality"]
    n = n or DEFAULT_N[modality]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = Path(config_path).stem
    out_path = OUT_DIR / f"{tag}.json"
    if out_path.exists():
        print(f"[error_analysis] skip {tag}: {out_path} exists")
        return
    print(f"[error_analysis] {cfg['model_name']} ({modality}), calibration n={n}", flush=True)
    t0 = time.time()
    col = ErrorCollector(cfg.get("block_size", 32))
    {"language": _collect_language, "vision": _collect_vision,
     "speech": _collect_speech}[modality](cfg, n, dev, col)
    col.drop_cache()
    summary = summarise(col.stats)
    result = {"config": cfg, "config_tag": tag, "calibration_n": n, "summary_db": summary,
              "per_layer": col.stats, "wall_seconds": round(time.time() - t0, 1),
              "env": environment_info()}
    with open(out_path, "w") as f:
        json.dump(result, f)
    print(f"[error_analysis] {tag}: {summary['n_layers']} layers, "
          f"MXFP4 e_W {summary['W:mxfp4']:.1f} dB, e_A {summary['A:mxfp4']:.1f} dB "
          f"({result['wall_seconds']} s) -> {out_path}", flush=True)


def _accuracy(cfg: Dict, mode: str):
    d = (HERE / cfg.get("results_dir", "results_v2") / cfg["modality"]
         / cfg["model_name"].replace("/", "__") / f"{mode}.json")
    if not d.exists():
        return None
    with open(d) as f:
        r = json.load(f)
    return {"language": r.get("ppl"), "speech": r.get("wer_pct"), "vision": r.get("acc1")}[cfg["modality"]]


def report() -> str:
    files = sorted(glob.glob(str(OUT_DIR / "*.json")))
    if not files:
        return "(no error-analysis results yet)"
    lines = ["## A. Which operand dominates plain-MXFP4 error, and where should residue go?", "",
             "Layer-averaged NMSE of the layer output in dB (more negative = smaller error).", "",
             "| model | e_W MXFP4 | e_A MXFP4 | dominant | full MXFP4 | residue-on-A | residue-on-W "
             "| predicted side | accuracy: better side | match |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    fmt_rows = []
    for fp in files:
        with open(fp) as f:
            r = json.load(f)
        s, cfg = r["summary_db"], r["config"]
        dom = "A" if s["A:mxfp4"] > s["W:mxfp4"] else "W"
        acc_a = _accuracy(cfg, "mxfp4_residual_act_only")
        acc_w = _accuracy(cfg, "mxfp4_residual_weight_only")
        if acc_a is None or acc_w is None:
            better = "?"
        else:
            higher_is_better = cfg["modality"] == "vision"
            a_better = acc_a > acc_w if higher_is_better else acc_a < acc_w
            better = "A" if a_better else "W"
        err_better = "A" if s["M:mxfp4_residual_act_only"] < s["M:mxfp4_residual_weight_only"] else "W"
        match = "?" if better == "?" else ("yes" if dom == better else "no")
        lines.append(
            f"| {r['config_tag']} | {s['W:mxfp4']:.1f} | {s['A:mxfp4']:.1f} | {dom} "
            f"| {s['M:mxfp4']:.1f} | {s['M:mxfp4_residual_act_only']:.1f} "
            f"| {s['M:mxfp4_residual_weight_only']:.1f} | {dom} (output error says {err_better}) "
            f"| {better} | {match} |")
        fmt_rows.append(
            f"| {r['config_tag']} | " + " | ".join(
                f"{s['W:' + f]:.1f} / {s['A:' + f]:.1f}" for f in OPERAND_FORMATS) + " |")
    lines += ["", "## B. Operand error per format", "",
              "Each cell: e_W / e_A in dB - error from quantising only the weights / only the "
              "activations with that format.", "",
              "| model | " + " | ".join(FORMAT_LABEL[f] for f in OPERAND_FORMATS) + " |",
              "|---|" + "---|" * len(OPERAND_FORMATS)] + fmt_rows
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--n", type=int, default=None, help="calibration items (default per modality)")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        print(report())
    elif args.config:
        run_model(args.config, args.n)
    else:
        ap.error("give --config or --report")


if __name__ == "__main__":
    main()
