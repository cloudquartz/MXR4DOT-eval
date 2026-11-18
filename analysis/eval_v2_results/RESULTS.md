# Eval-v2 results (copies)

Byte-identical copies of the eval-v2 outputs. Outputs produced on an RTX 4060 Ti (16 GB) using the v2 evaluation harness (`<EVAL_ROOT>`), organized as `<modality>/<model>/<mode>.json`.

| Folder / file here | Source / run setting | Contents |
|---|---|---|
| `language/` | `results_v2/language/` | WikiText-2 perplexity: GPT-2, Llama-2-7B, Mistral-7B, Llama-3-8B × 8 formats (key `ppl`, `per_chunk_nll`) |
| `speech/` | `results_v2/speech/` | wav2vec2-base-960h, LibriSpeech test-clean, WER × 8 formats |
| `vision/` | `results_v2/vision/` | ResNet-50, ViT-B/16, DeiT-S/16, ImageNet validation (50,000 images) top-1 × 8 formats |
| `error_analysis/` | `results_v2/error_analysis/` | per-layer operand error (NMSE, dB) per model; `report.md` |
| `zeroshot/` | `results_v2/zeroshot/` | lm-eval 0-shot: GPT-2 8 formats × 5 tasks; Llama-2-7B 5 formats × 4 tasks (its boolq failed there: CUDA out of memory; re-run on 24 Oct 2026, next row) |
| `zeroshot/meta-llama__Llama-2-7b-hf/*__boolq.json` | batch size 1 re-run (24 Oct 2026) | Llama-2-7B boolq × 5 formats, re-run at **batch size 1** (the other 4 Llama-2 tasks ran at batch size 4) `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| `zeroshot_report.md` | `scripts/run_zeroshot_v2.py --report` (5,000 resamples) | zero-shot table with paired CIs; Llama-2 CIs after the bf16 / arc_easy cell differ from the old report by bootstrap noise (one RNG per model, boolq drawn second) |
| `ci_report.md` | `results_v2/ci_report.md` | paired bootstrap CIs vs MXFP4R (residue on activations), all modalities |
| `other/speech/` | `results_v2_other/speech/` | wav2vec2-base-960h, LibriSpeech test-other, WER × 8 formats |
| `other/ci_report.md` | `results_v2_other/ci_report.md` | CIs for test-other |
| `<modality>/<model>/mxfp4_w_mxfp8_a.json` (9 files: `language/` × 4 models, `vision/` × 3, `speech/`, `other/speech/`) | W4A8 runs (15–16 Nov 2026) | **W4A8**: MXFP4 weights, MXFP8-E4M3 activations, no residue (`weight_mode` `mxfp4`, `act_mode` `mxfp8_e4m3`, `eff_bits` 6.25), on the same full test sets and items as the other formats. Run with `run_eval.py --config configs/v2/<model>.yaml --modes mxfp4_w_mxfp8_a --results <folder>`. `env.lm_eval` is `0.4.13` in these files and `None` in the `1925f9e-dirty` results (see the provenance note) |
| `ci/ci_ref_*.md` | post-processed | `scripts/ci_report.py` on `language/` (default ref and `--ref mxfp8_e4m3`, 10,000 resamples, seed 0). Regenerated 16 Nov 2026 with the W4A8 files: every earlier line is unchanged, one `mxfp4_w_mxfp8_a` row added per model |
| `ci/all_ref_*.md`, `ci/other_ref_*.md` | post-processed | the same script and settings (16 Nov 2026) on `language/` + `vision/` + `speech/` (`all`) and on `other/speech/` (`other`), each with the default ref and `--ref mxfp8_e4m3`, W4A8 included. Without the W4A8 files the default-ref runs reproduce `ci_report.md` and `other/ci_report.md` byte for byte; those two files are kept as generated during the sweep and have no W4A8 row |
| `index/` | model shard indices | `model.safetensors.index.json` of the Llama-2 / Mistral / Llama-3 snapshots eval v2 used (Hugging Face model shard indices) |

The 8 formats: `bf16`/`fp32`, `mxfp4`, `nvfp4`, `m2xfp`, `mxfp4_residual_act_only`, `mxfp4_residual_weight_only`,
`mxfp4_residual`, `mxfp8_e4m3`. A ninth, `mxfp4_w_mxfp8_a` (W4A8), was added on 16 Nov 2026; it is not in
`error_analysis/` or `zeroshot/`. Every quantised run leaves the output head unquantised: `lm_head` for
language and speech (`skip_names` in the JSONs), the classifier head for vision (stated in the vision configs).

Not copied: `results_v2/_smoke/` (smoke subsets) and the run logs (`run_v2.log`, `run_remaining.log`).

**Provenance note:** 70 of the 72 accuracy results (all except Mistral-7B and Llama-3-8B `m2xfp`) were written
on 19 Oct 2026, before the last edit of `core/layers.py` (20 Oct, row-chunked weight quantisation, claimed
bit-identical). Re-run checks at the committed code, each bit-identical to the stored result in every
per-item record: Llama-2-7B `bf16`, `mxfp4`, `nvfp4`, `m2xfp`, `mxfp4_residual_act_only` at `ab59463` (23–24 Oct 2026,
166 chunks each); GPT-2 `bf16`, `mxfp4`, `mxfp4_residual_act_only` at `9021868` (15 Nov 2026, 280 chunks each);
DeiT-S/16 and wav2vec2 test-clean `mxfp4`, `mxfp8_e4m3` at `9021868` (16 Nov 2026; 50,000 images, top-1 and top-5
hit per image; 2,620 utterances, hypothesis and edit count per utterance). For the other stored results the
equivalence is an inference, not a measurement. These re-runs record `env.lm_eval` `0.4.13` where the stored
results record `None` (lm-eval was not importable in the environment on 19 Oct); every other recorded
environment field except the commit is equal.
