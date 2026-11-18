"""
ci_report.py
============
Confidence intervals for the v2 results written by run_eval.py.

    python ci_report.py --results results_v2
    python ci_report.py --results results_v2 --ref mxfp4_residual_act_only --boot 10000

For every model directory it prints one Markdown table:
  - the metric on the full evaluation set,
  - a 95% percentile-bootstrap CI over evaluation items
    (language: 1 item = one chunk; speech: one utterance; vision: one image),
  - the PAIRED difference to the reference mode (same resampled items for both
    modes) with its 95% CI, and whether that CI excludes zero.

Metrics are recomputed from per-item data exactly as the harness does:
  PPL = exp(mean chunk NLL)   (equal-length chunks)
  WER = 100 * sum(edits) / sum(reference words)
  top-1 = 100 * mean(correct)
Quantisation is deterministic, so the only randomness is which test items
were sampled - that is exactly what the bootstrap estimates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from typing import Dict, List, Tuple

import numpy as np

MODE_ORDER = ["fp32", "bf16", "mxfp8_e4m3", "mxfp4_residual", "mxfp4_residual_act_only",
              "mxfp4_residual_weight_only", "m2xfp", "nvfp4", "mxfp4"]


def _items(modality: str, r: Dict) -> Tuple[np.ndarray, ...]:
    if modality == "language":
        return (np.asarray(r["per_chunk_nll"], dtype=np.float64),)
    if modality == "speech":
        return (np.asarray(r["per_item_edits"], dtype=np.float64),
                np.asarray(r["per_item_ref_words"], dtype=np.float64))
    return (np.asarray(r["per_item_top1"], dtype=np.float64),)


def _metric(modality: str, items: Tuple[np.ndarray, ...], idx: np.ndarray) -> np.ndarray:
    """Metric for each bootstrap resample. idx: [n_boot, n_items]."""
    if modality == "language":
        return np.exp(items[0][idx].mean(axis=1))
    if modality == "speech":
        return 100.0 * items[0][idx].sum(axis=1) / items[1][idx].sum(axis=1)
    return 100.0 * items[0][idx].mean(axis=1)


def _ident(modality: str, r: Dict):
    if modality == "speech":
        return tuple(r.get("item_ids", []))
    if modality == "language":
        return (r.get("n_chunks"), r.get("seq_len"))
    return (r.get("total_samples"),)


def report_model(model_dir: Path, modality: str, ref: str, n_boot: int, seed: int) -> str:
    runs: Dict[str, Dict] = {}
    for p in sorted(model_dir.glob("*.json")):
        with open(p) as f:
            runs[p.stem] = json.load(f)
    if not runs:
        return ""
    modes = [m for m in MODE_ORDER if m in runs] + sorted(m for m in runs if m not in MODE_ORDER)

    # All modes must be evaluated on the same items for paired comparisons.
    idents = {m: _ident(modality, runs[m]) for m in modes}
    first = idents[modes[0]]
    mismatched = [m for m in modes if idents[m] != first]

    items = {m: _items(modality, runs[m]) for m in modes}
    n = len(items[modes[0]][0])
    full = np.arange(n)[None, :]
    point = {m: float(_metric(modality, items[m], full)[0]) for m in modes}

    # Resample in batches (n_boot x n indices would not fit in RAM for 50k
    # images); every mode sees the SAME resampled indices -> paired bootstrap.
    rng = np.random.default_rng(seed)
    compared = [m for m in modes if m not in mismatched]
    boot = {m: np.empty(n_boot) for m in compared}
    batch = max(1, min(n_boot, 20_000_000 // max(1, n)))
    for b0 in range(0, n_boot, batch):
        b1 = min(n_boot, b0 + batch)
        idx = rng.integers(0, n, size=(b1 - b0, n))
        for m in compared:
            boot[m][b0:b1] = _metric(modality, items[m], idx)

    unit = {"language": "PPL", "speech": "WER %", "vision": "top-1 %"}[modality]
    lines = [f"### {model_dir.name}  ({unit}, n = {n} items, {n_boot} bootstrap resamples)", ""]
    have_ref = ref in boot
    hdr = f"| mode | {unit} | 95% CI |"
    sep = "|---|---|---|"
    if have_ref:
        hdr += f" diff vs {ref} | 95% CI of diff | CI excludes 0 |"
        sep += "---|---|---|"
    lines += [hdr, sep]
    for m in modes:
        if m in mismatched:
            lines.append(f"| {m} | {point[m]:.4g} | (different items - not compared) |"
                         + (" | | |" if have_ref else ""))
            continue
        lo, hi = np.percentile(boot[m], [2.5, 97.5])
        row = f"| {m} | {point[m]:.4g} | [{lo:.4g}, {hi:.4g}] |"
        if have_ref:
            if m == ref:
                row += " — | — | — |"
            else:
                d = boot[m] - boot[ref]
                dlo, dhi = np.percentile(d, [2.5, 97.5])
                excl = "yes" if (dlo > 0 or dhi < 0) else "no"
                row += f" {point[m] - point[ref]:+.4g} | [{dlo:+.4g}, {dhi:+.4g}] | {excl} |"
        lines.append(row)
    if mismatched:
        lines += ["", f"WARNING: {mismatched} were evaluated on different items than {modes[0]}."]
    smoke = [m for m in modes if runs[m].get("limit") is not None]
    if smoke:
        lines += ["", f"NOTE: {smoke} are --limit smoke runs, not full-set results."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_v2")
    ap.add_argument("--ref", default="mxfp4_residual_act_only",
                    help="mode every other mode is compared against")
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="also write the Markdown here")
    args = ap.parse_args()

    root = Path(args.results)
    parts: List[str] = []
    for modality in ("language", "vision", "speech"):
        mdir = root / modality
        if not mdir.is_dir():
            continue
        for model_dir in sorted(p for p in mdir.iterdir() if p.is_dir()):
            txt = report_model(model_dir, modality, args.ref, args.boot, args.seed)
            if txt:
                parts.append(txt)
    out = "\n".join(parts) if parts else f"(no results under {root})"
    print(out)
    if args.out:
        Path(args.out).write_text(out)


if __name__ == "__main__":
    main()
