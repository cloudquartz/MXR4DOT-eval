# Accuracy evaluation of microscaling number formats

This repository contains a PyTorch software accuracy evaluation of low-bit number formats via fake quantisation: MXFP4, MXFP8 (E4M3), M2XFP4, MXFP4 with a residual pass ("MXFP4 residue"), NVFP4, and W4A8 (MXFP4 weights with MXFP8 E4M3 activations). Weights and activations of Linear and Conv layers (all except the output head) are quantised and dequantised in PyTorch before scoring each model on its task. Benchmarks cover language (perplexity and zero-shot tasks), vision (ImageNet top-1), and speech (word error rate). No custom hardware is needed; a GPU is used for the larger models and a CPU suffices for small runs and unit tests.

| Path | Contents |
|---|---|
| `run_eval.py` | Evaluation driver: one config per model, one JSON result per `(model, mode)` |
| `run_all.sh` | Shell runner for smoke checks, full sweeps, error analysis, and zero-shot runs |
| `core/` | Quantisers (`quantizer.py`), quantised layers, metrics, sequential layer-by-layer execution |
| `frameworks/` | Model and dataset loading per modality (`language`, `vision`, `speech`, `recsys`) |
| `configs/v2/` | Configs used by `run_eval.py` (one per model, plus zero-shot configs) |
| `configs/` | Older v1 configs, used by `legacy/run_sweep.py` and related scripts |
| `scripts/` | Confidence-interval reports (`ci_report.py`), status (`progress.py`), `run_zeroshot_v2.py`, `run_error_analysis.py` |
| `tests/` | Unit tests, hand-trace scripts, and probes |
| `legacy/` | Earlier v1 sweep runners (`run_sweep.py`, `run_m2xfp.py`), diagnostics, and SQLite ledger scripts |
| `analysis/eval_v2_results/` | Stored results and reports (see `RESULTS.md` inside that folder) |
| `environment.yml`, `requirements_v2.txt` | Environment definitions |

## Setup and usage

Create a Conda environment from `environment.yml`, or install dependencies into a virtualenv on top of an existing CUDA PyTorch environment without overwriting `torch`:

```bash
# Option A: Conda
conda env create -f environment.yml
conda activate mxfp

# Option B: Virtualenv reusing an existing CUDA PyTorch install
TORCH_SITE=$(python -c "import torch, os; print(os.path.dirname(os.path.dirname(torch.__file__)))")
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
python -m venv ~/venvs/mxfp
deactivate 2>/dev/null; conda deactivate 2>/dev/null
source ~/venvs/mxfp/bin/activate
python -c "import site, sys; open(site.getsitepackages()[0] + '/_shared_torch.pth', 'w').write(sys.argv[1] + '\n')" "$TORCH_SITE"
echo "torch==$TORCH_VER" > constraints_local.txt
pip install torchvision -c constraints_local.txt --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements_v2.txt -c constraints_local.txt
```

Gated checkpoints and datasets (`meta-llama/Llama-2-7b-hf`, `meta-llama/Meta-Llama-3-8B`, `mistralai/Mistral-7B-v0.1`, `ILSVRC/imagenet-1k`) require accepting terms on Hugging Face and logging in (`hf auth login` or `huggingface-cli login`).

```bash
# Unit tests (CPU, no downloads required)
python tests/test_quantizer.py
python tests/test_mxfp8.py
python -m pytest tests/test_eval_fixes.py tests/test_quantizer.py -v

# Main evaluation driver (or run `bash run_all.sh [smoke|eval|remaining|all]`)
python run_eval.py --config configs/v2/<name>.yaml --modes <mode> [<mode> ...] --results <dir>

# Reports, status, error analysis, and zero-shot evaluation
python scripts/ci_report.py --results <dir>
python scripts/progress.py
python scripts/run_error_analysis.py --config configs/v2/<name>.yaml
python scripts/run_zeroshot_v2.py --config configs/v2/zeroshot_gpt2.yaml
```

Here `<name>` is a config in `configs/v2/` (such as `gpt2`, `resnet50`, or `wav2vec2`). Pass `--modes` to select a subset of the config's `quant_modes`, or omit it to run all of them. Use `--results` to change the output directory and `--overwrite` to re-run finished entries. Pass `--limit N` to run a quick subset into `<dir>/_smoke/`. Outputs are written to `<dir>/<modality>/<model>/<mode>.json`.

## Results and companion repository

`analysis/eval_v2_results/` stores the saved outputs: per-format JSONs under `language/`, `speech/`, `vision/`, and `other/`, zero-shot JSONs under `zeroshot/`, and generated Markdown reports (`ci/`, `ci_report.md`, `zeroshot_report.md`, `error_analysis/`). Details for each file appear in `analysis/eval_v2_results/RESULTS.md`, where `<EVAL_ROOT>` stands for the harness root path on the evaluation host (redacted in this copy).

The RTL of the MX dot-product coprocessor is in a companion repository:
https://github.com/cloudquartz/MXR4DOT-rtl.
