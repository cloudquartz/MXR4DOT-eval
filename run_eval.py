"""
run_eval.py
===========
v2 evaluation driver: one config per model, one JSON result per (model, mode).

    python run_eval.py --config configs/v2/gpt2.yaml
    python run_eval.py --config configs/v2/llama2_7b.yaml --modes bf16 mxfp4
    python run_eval.py --config configs/v2/resnet50.yaml --limit 256     # quick check

Each result lands in <results_dir>/<modality>/<model>/<mode>.json with the
metric, the per-item data needed for confidence intervals (per-chunk NLL,
per-utterance edits, per-image correctness), the full config and the exact
library versions. Finished runs are skipped, so an interrupted sweep can be
restarted with the same command; pass --overwrite to redo them.
--limit runs a small subset and writes to <results_dir>/_smoke/ instead.

Then: python ci_report.py --results results_v2
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml


def _device(cfg: Dict) -> torch.device:
    d = cfg.get("device", "auto")
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def _flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _run_one(cfg: Dict, mode: str, device: torch.device, limit: Optional[int]) -> Dict[str, Any]:
    modality = cfg["modality"]
    name = cfg["model_name"]
    bs = cfg.get("block_size", 32)
    if modality == "language":
        from frameworks.language import LanguageEvalHarness
        h = LanguageEvalHarness(
            model_name=name, quant_mode=mode, seed=cfg.get("seed", 42),
            n_chunks=limit if limit is not None else cfg.get("n_chunks"),
            seq_len=cfg.get("seq_len", 1024), block_size=bs, device=device,
            exec_mode=cfg.get("exec_mode", "auto"), skip_names=cfg.get("skip_names"))
        return h.run()
    if modality == "speech":
        from frameworks.speech import SpeechEvalHarness
        h = SpeechEvalHarness(
            model_name=name, quant_mode=mode,
            n_samples=limit if limit is not None else cfg.get("n_samples"),
            seed=cfg.get("seed", 42), block_size=bs, device=device,
            skip_feature_extractor=cfg.get("skip_feature_extractor", False),
            extra_skip_names=cfg.get("extra_skip_names"),
            librispeech_config=cfg.get("librispeech_config", "clean"))
        return h.run()
    if modality == "vision":
        from frameworks.vision import VisionEvalHarness
        h = VisionEvalHarness(
            model_name=name, quant_mode=mode,
            n_images=limit if limit is not None else cfg.get("n_images"),
            batch_size=cfg.get("batch_size", 64), seed=cfg.get("seed", 42),
            block_size=bs, device=device, num_workers=cfg.get("num_workers", 8),
            skip_names=cfg.get("skip_names"))
        return h.run()
    raise ValueError(f"unknown modality {modality}")


def _headline(modality: str, r: Dict[str, Any]) -> str:
    if modality == "language":
        return f"PPL {r['ppl']:.4f}"
    if modality == "speech":
        return f"WER {r['wer_pct']:.2f}%"
    return f"top-1 {r['acc1']:.2f}%"


def main():
    ap = argparse.ArgumentParser(description="MXR4DOT v2 accuracy evaluation")
    ap.add_argument("--config", required=True)
    ap.add_argument("--modes", nargs="*", help="subset of the config's quant_modes")
    ap.add_argument("--limit", type=int, default=None,
                    help="small subset (chunks / utterances / images) for a quick check")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--results", default=None, help="results dir (default: config or results_v2)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    from core.compat import environment_info
    from core.layers import configure_exact_numerics

    configure_exact_numerics()          # before environment_info, so it records the setting
    device = _device(cfg)
    modes = args.modes or cfg["quant_modes"]
    root = Path(args.results or cfg.get("results_dir", "results_v2"))
    if args.limit is not None:
        root = root / "_smoke"
    out_dir = root / cfg["modality"] / cfg["model_name"].replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)
    env = environment_info()
    print(f"[run_eval] {cfg['model_name']} | modes={modes} | device={device} | out={out_dir}")
    print(f"[run_eval] env: {env}")

    failures = []
    for mode in modes:
        out_path = out_dir / f"{mode}.json"
        if out_path.exists() and not args.overwrite:
            print(f"[run_eval] skip {mode}: {out_path} exists (use --overwrite)")
            continue
        _flush()
        t0 = time.time()
        try:
            r = _run_one(cfg, mode, device, args.limit)
        except Exception as e:
            traceback.print_exc()
            failures.append((mode, repr(e)))
            continue
        r["wall_seconds"] = round(time.time() - t0, 1)
        if torch.cuda.is_available():
            r["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        r["config"] = cfg
        r["config_path"] = os.path.abspath(args.config)
        r["limit"] = args.limit
        r["env"] = env
        tmp = out_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(r, f)
        os.replace(tmp, out_path)
        print(f"[run_eval] {mode}: {_headline(cfg['modality'], r)} "
              f"({r['wall_seconds']} s, peak {r.get('peak_vram_gb', 0)} GB) -> {out_path}")

    if failures:
        print("\n[run_eval] FAILED modes:")
        for m, e in failures:
            print(f"  {m}: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
