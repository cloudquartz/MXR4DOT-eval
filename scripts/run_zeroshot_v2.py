"""
run_zeroshot_v2.py
==================
Zero-shot downstream accuracy (lm-eval) with the v2 fake-quant layers.
Replaces run_zeroshot.py for paper numbers (that script used the v1 layers).

    python run_zeroshot_v2.py --config configs/v2/zeroshot_gpt2.yaml
    python run_zeroshot_v2.py --report

One JSON per (model, mode, task) in results_v2/zeroshot/<model>/, holding the
accuracy, lm-eval's stderr and the per-question correctness (so paired
bootstrap CIs can be computed later). Finished (mode, task) pairs are skipped.
Each task runs on its own, so a dataset problem in one task does not stop the
others.

Memory: models above 2 B parameters are placed on the GPU layer by layer while
quantising, so peak memory = the quantised model. Formats whose quantised
weights are exactly representable in bf16 (MXFP4, MXFP4R-on-activations,
M2XFP4, MXFP8) keep the bf16 model size (13.5 GB for Llama-2-7B, fits 16 GB);
NVFP4 and residue-on-weights need a second bf16 copy of each weight and do not
fit a 16 GB card for 7B models, so the 7B config leaves them out.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import time
import traceback
from pathlib import Path

import torch
import yaml

import sys
HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
OUT_ROOT = HERE / "results_v2" / "zeroshot"
METRIC_PREF = ("acc_norm,none", "acc,none")


def _load(model_name: str, mode: str, device: torch.device, block_size: int):
    from frameworks.language import LanguageEvalHarness, _replace_layers, _resolve_modes
    from core.sequential import supports_sequential

    h = LanguageEvalHarness(model_name=model_name, quant_mode="bf16", device=device)
    model, tok = h._load_model_and_tokenizer()
    quant = mode not in ("bf16", "fp32")
    w, a = _resolve_modes(mode) if quant else (None, None)
    big = sum(p.numel() for p in model.parameters()) > 2e9
    if big and supports_sequential(model):
        layers = model.model.layers
        for i in range(len(layers)):
            layer = layers[i].to(device)
            if quant:
                _replace_layers(layer, w, a, block_size, skip_names=[])
            layers[i] = layer
            if device.type == "cuda":
                torch.cuda.empty_cache()
        model.to(device)
    else:
        model.to(device)
        if quant:
            _replace_layers(model, w, a, block_size, ["lm_head"])
    model.eval()
    return model, tok


def _per_doc(samples, metric_key):
    name = metric_key.split(",")[0]
    vals = []
    for s in samples or []:
        v = s.get(name, s.get(metric_key))
        if isinstance(v, (list, tuple)):
            v = v[0]
        if v is not None:
            vals.append(float(v))
    return vals


def run(config_path: str):
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from core.compat import environment_info
    from core.layers import configure_exact_numerics

    configure_exact_numerics()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = OUT_ROOT / cfg["model_name"].replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)
    env = environment_info()
    try:
        import lm_eval
        env["lm_eval"] = getattr(lm_eval, "__version__", "?")
    except Exception:
        pass

    for mode in cfg["quant_modes"]:
        todo = [t for t in cfg["tasks"] if not (out_dir / f"{mode}__{t}.json").exists()]
        if not todo:
            print(f"[zeroshot] skip {mode}: all tasks done", flush=True)
            continue
        print(f"[zeroshot] {cfg['model_name']} | {mode} | tasks {todo}", flush=True)
        try:
            model, tok = _load(cfg["model_name"], mode, dev, cfg.get("block_size", 32))
            lm = HFLM(pretrained=model, tokenizer=tok, batch_size=cfg.get("batch_size", "auto"),
                      max_batch_size=cfg.get("max_batch_size", 16))
        except Exception:
            traceback.print_exc()
            print(f"[zeroshot] FAILED to load {mode}", flush=True)
            continue
        for task in todo:
            t0 = time.time()
            try:
                res = simple_evaluate(model=lm, tasks=[task], num_fewshot=0, log_samples=True,
                                      random_seed=0, numpy_random_seed=0, torch_random_seed=0,
                                      fewshot_random_seed=0)
                tr = res["results"][task]
                key = next(k for k in METRIC_PREF if k in tr)
                out = {
                    "model_name": cfg["model_name"], "mode": mode, "task": task,
                    "metric": key.split(",")[0], "value": float(tr[key]),
                    "stderr": float(tr.get(key.replace(",", "_stderr,"), float("nan"))),
                    "per_doc": _per_doc(res.get("samples", {}).get(task), key),
                    "wall_seconds": round(time.time() - t0, 1), "config": cfg, "env": env,
                }
                with open(out_dir / f"{mode}__{task}.json", "w") as f:
                    json.dump(out, f)
                print(f"[zeroshot] {mode} {task}: {out['metric']} = {100 * out['value']:.2f} "
                      f"({out['wall_seconds']} s)", flush=True)
            except Exception:
                traceback.print_exc()
                print(f"[zeroshot] FAILED {mode} {task}", flush=True)
        del lm, model
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()


def report() -> str:
    import numpy as np
    lines = []
    for model_dir in sorted(p for p in OUT_ROOT.glob("*") if p.is_dir()):
        rows = {}
        for fp in glob.glob(str(model_dir / "*.json")):
            with open(fp) as f:
                r = json.load(f)
            rows.setdefault(r["mode"], {})[r["task"]] = r
        if not rows:
            continue
        tasks = sorted({t for m in rows.values() for t in m})
        ref = rows.get("mxfp4_residual_act_only", {})
        lines += [f"### {model_dir.name} (zero-shot accuracy %, ± lm-eval stderr; "
                  f"[paired 95% CI of diff vs MXFP4R-A])", "",
                  "| mode | " + " | ".join(tasks) + " |", "|---|" + "---|" * len(tasks)]
        rng = np.random.default_rng(0)
        for mode in sorted(rows):
            cells = []
            for t in tasks:
                r = rows[mode].get(t)
                if r is None:
                    cells.append("—")
                    continue
                cell = f"{100 * r['value']:.1f} ± {100 * r['stderr']:.1f}"
                rr = ref.get(t)
                if rr and mode != "mxfp4_residual_act_only" and \
                        len(rr["per_doc"]) == len(r["per_doc"]) and r["per_doc"]:
                    a, b = np.asarray(r["per_doc"]), np.asarray(rr["per_doc"])
                    idx = rng.integers(0, len(a), size=(5000, len(a)))
                    d = 100 * (a[idx].mean(1) - b[idx].mean(1))
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    cell += f" [{lo:+.1f}, {hi:+.1f}]"
                cells.append(cell)
            lines.append(f"| {mode} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines) if lines else "(no zero-shot results yet)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        print(report())
    elif args.config:
        run(args.config)
    else:
        ap.error("give --config or --report")


if __name__ == "__main__":
    main()
