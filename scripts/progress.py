"""
progress.py - one-glance status of the v2 evaluation.

    python progress.py            # status table + what is running + last log line
    watch -n 30 python progress.py    # refresh every 30 s (Ctrl-C to quit)

Reads configs/v2/*.yaml, checks which result JSONs exist, and shows the
headline number for each finished (model, format).
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
from pathlib import Path

import yaml

import sys
HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _headline(modality: str, path: Path) -> str:
    try:
        with open(path) as f:
            r = json.load(f)
        if modality == "language":
            return f"{r['ppl']:.3f}"
        if modality == "speech":
            return f"{r['wer_pct']:.2f}%"
        return f"{r['acc1']:.2f}%"
    except Exception:
        return "?"


def main():
    total = done = 0
    rows = []
    for cfg_path in sorted(glob.glob(str(HERE / "configs" / "v2" / "*.yaml"))):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        if "modality" not in cfg:          # zero-shot configs are reported separately below
            continue
        out_dir = (HERE / cfg.get("results_dir", "results_v2") / cfg["modality"]
                   / cfg["model_name"].replace("/", "__"))
        finished, missing = [], []
        for mode in cfg["quant_modes"]:
            p = out_dir / f"{mode}.json"
            total += 1
            if p.exists():
                done += 1
                finished.append(f"{mode}={_headline(cfg['modality'], p)}")
            else:
                missing.append(mode)
        name = Path(cfg_path).stem
        n = len(cfg["quant_modes"])
        status = "DONE" if not missing else f"missing: {', '.join(missing)}"
        rows.append(f"  {name:<15} {n - len(missing)}/{n}  {status}")

    print("  ACCURACY RUNS (run_eval.py)")
    print("\n".join(rows))
    print(f"  total {done}/{total} finished")

    ea_dir = HERE / "results_v2" / "error_analysis"
    ea_cfgs = ["gpt2", "vit_b16", "deit_s16", "resnet50", "wav2vec2",
               "llama2_7b", "mistral_7b", "llama3_8b"]
    ea_done = [c for c in ea_cfgs if (ea_dir / f"{c}.json").exists()]
    print(f"\n  ERROR ANALYSIS  {len(ea_done)}/{len(ea_cfgs)} models"
          + (f"   missing: {', '.join(c for c in ea_cfgs if c not in ea_done)}"
             if len(ea_done) < len(ea_cfgs) else "   DONE"))

    zs_lines = []
    for zc in sorted(glob.glob(str(HERE / "configs" / "v2" / "zeroshot_*.yaml"))):
        with open(zc) as f:
            z = yaml.safe_load(f)
        zdir = HERE / "results_v2" / "zeroshot" / z["model_name"].replace("/", "__")
        need = [(m, t) for m in z["quant_modes"] for t in z["tasks"]]
        got = sum((zdir / f"{m}__{t}.json").exists() for m, t in need)
        zs_lines.append(f"  {Path(zc).stem:<20} {got}/{len(need)} (mode, task) pairs")
    if zs_lines:
        print("\n  ZERO-SHOT")
        print("\n".join(zs_lines))

    try:
        ps = subprocess.run(["pgrep", "-af", "run_eval.py|run_error_analysis.py|run_zeroshot_v2.py"],
                            capture_output=True, text=True).stdout
        running = [l.split(None, 1)[1] for l in ps.splitlines() if "pgrep" not in l]
        running = [r[r.find("run_"):] for r in running if "run_" in r]
    except Exception:
        running = []
    print(f"\n  RUNNING: {running[0] if running else 'nothing'}")

    logs = sorted(glob.glob(str(HERE / "*.log")), key=os.path.getmtime)
    if logs:
        with open(logs[-1], errors="replace") as f:
            lines = [l.rstrip() for l in f.read().replace("\r", "\n").splitlines() if l.strip()]
        if lines:
            print(f"  LAST LINE of {Path(logs[-1]).name}: {lines[-1][:150]}")


if __name__ == "__main__":
    main()
