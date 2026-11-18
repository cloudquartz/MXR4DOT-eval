#!/usr/bin/env bash
# Main v2 evaluation pipeline runner.
# Usage:
#   bash run_all.sh smoke      # CPU unit tests + quick subset smoke runs
#   bash run_all.sh eval       # Full v2 accuracy runs + bootstrap CI reports
#   bash run_all.sh remaining  # Per-layer error analysis + zero-shot evaluations
#   bash run_all.sh all        # Full accuracy runs, CI reports, error analysis, zero-shot
set -euo pipefail

cd "$(dirname "$0")"
PY="${PY:-python}"
MODE="${1:-all}"

case "$MODE" in
  smoke)
    "$PY" tests/test_quantizer.py
    "$PY" tests/test_mxfp8.py
    "$PY" -u run_eval.py --config configs/v2/gpt2.yaml --limit 4
    "$PY" -u run_eval.py --config configs/v2/wav2vec2.yaml --limit 20
    "$PY" -u run_eval.py --config configs/v2/resnet50.yaml --limit 256
    "$PY" -u run_eval.py --config configs/v2/llama2_7b.yaml --limit 2 --modes bf16 mxfp4_residual_act_only
    ;;
  eval)
    for c in gpt2 wav2vec2 wav2vec2_other resnet50 vit_b16 deit_s16 llama2_7b mistral_7b llama3_8b; do
      "$PY" -u run_eval.py --config "configs/v2/${c}.yaml" 2>&1 | tee -a run_v2.log
    done
    "$PY" scripts/ci_report.py --results results_v2 --ref mxfp4_residual_act_only --out results_v2/ci_report.md
    "$PY" scripts/ci_report.py --results results_v2_other --out results_v2_other/ci_report.md
    ;;
  remaining)
    mkdir -p results_v2/error_analysis
    for c in gpt2 vit_b16 deit_s16 resnet50 wav2vec2 llama2_7b mistral_7b llama3_8b; do
      "$PY" -u scripts/run_error_analysis.py --config "configs/v2/${c}.yaml"
    done
    "$PY" -u scripts/run_error_analysis.py --report > results_v2/error_analysis/report.md

    "$PY" -u scripts/run_zeroshot_v2.py --config configs/v2/zeroshot_gpt2.yaml
    "$PY" -u scripts/run_zeroshot_v2.py --report > results_v2/zeroshot_report.md
    "$PY" -u scripts/run_zeroshot_v2.py --config configs/v2/zeroshot_llama2_7b.yaml
    "$PY" -u scripts/run_zeroshot_v2.py --report > results_v2/zeroshot_report.md
    ;;
  all)
    "$0" eval
    "$0" remaining
    ;;
  *)
    echo "Usage: $0 [smoke|eval|remaining|all]" >&2
    exit 1
    ;;
esac
