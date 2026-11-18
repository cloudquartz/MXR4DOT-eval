"""
frameworks/speech.py
====================
LibriSpeech WER harness for wav2vec2 (facebook/wav2vec2-base-960h).

Protocol (v2):
  - Dataset: openslr/librispeech_asr (parquet), config "clean" (or "other"),
    split "test". n_samples=None (default) evaluates ALL 2,620 test-clean
    utterances in order. An integer evaluates a seeded random subset.
  - Audio is decoded with soundfile (Audio(decode=False)), which avoids the
    torchcodec/FFmpeg dependency of datasets >= 4.
  - Metric: corpus WER (sum of word edits / sum of reference words), greedy
    CTC decoding, no LM. Per-utterance edits and word counts are returned so
    confidence intervals can be bootstrapped over utterances.
  - Model runs in fp32; quantised matmuls/convs are exact (core/layers.py).

Layer policy: all nn.Linear and nn.Conv1d are quantised (feature extractor,
positional conv, transformer), except the CTC head `lm_head`.
"""

from __future__ import annotations

import os as _os
try:
    import certifi as _certifi
    _os.environ.setdefault("SSL_CERT_FILE", _certifi.where())
    _os.environ.setdefault("REQUESTS_CA_BUNDLE", _certifi.where())
except ImportError:
    pass

import io
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

DEFAULT_SKIP_NAMES = ["lm_head"]
DOCUMENTED_REFERENCE_WER = 3.4   # %, HF model card, greedy, test-clean


# ─────────────────────────────────────────────────────────────────────────────
# WER helpers
# ─────────────────────────────────────────────────────────────────────────────

def _levenshtein(s1: list, s2: list) -> int:
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev if s1[i - 1] == s2[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[n]


def wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    if not ref_words:
        return 0.0 if not hypothesis.split() else 1.0
    return _levenshtein(ref_words, hypothesis.split()) / len(ref_words)


def corpus_wer(references: List[str], hypotheses: List[str]) -> float:
    edits = sum(_levenshtein(r.split(), h.split()) for r, h in zip(references, hypotheses))
    words = sum(len(r.split()) for r in references)
    return 100.0 * edits / words if words else 0.0


def _ctc_greedy_decode(logits: torch.Tensor, processor) -> str:
    pred_ids = logits.argmax(dim=-1)
    return processor.batch_decode(pred_ids)[0].strip().upper()


def _resolve_modes(quant_mode: str) -> Tuple[str, str]:
    from frameworks.language import _resolve_modes as _lang_resolve
    return _lang_resolve(quant_mode)


# ─────────────────────────────────────────────────────────────────────────────
# Layer substitution
# ─────────────────────────────────────────────────────────────────────────────

def _replace_speech_layers(
    model: nn.Module,
    weight_mode: str,
    act_mode: str,
    block_size: int = 32,
    skip_names: Optional[List[str]] = None,
    skip_feature_extractor: bool = False,
) -> nn.Module:
    """Replace nn.Conv1d / nn.Linear with FakeQuant equivalents in place."""
    from core.layers import FakeQuantLinear, FakeQuantConv

    skip_set = set(DEFAULT_SKIP_NAMES if skip_names is None else skip_names)

    def _recurse(module: nn.Module, prefix: str = "") -> None:
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}".lstrip(".")
            if skip_feature_extractor and full_name.startswith("wav2vec2.feature_extractor"):
                continue
            if full_name in skip_set:
                continue
            if isinstance(child, nn.Conv1d):        # includes the weight-normed pos_conv
                setattr(module, name, FakeQuantConv(child, weight_mode, act_mode, block_size))
            elif isinstance(child, nn.Linear):
                setattr(module, name, FakeQuantLinear(child, weight_mode, act_mode, block_size))
            else:
                _recurse(child, full_name)

    _recurse(model)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def _decode_audio(audio: Dict[str, Any]) -> Tuple[Any, int]:
    import soundfile as sf
    if audio.get("bytes"):
        data, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
    else:
        data, sr = sf.read(audio["path"], dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def _load_librispeech(
    n_samples: Optional[int] = None,
    seed: int = 42,
    config: str = "clean",
    split: str = "test",
    dataset_name: str = "openslr/librispeech_asr",
):
    """Returns a datasets.Dataset with undecoded audio. n_samples=None → all."""
    from datasets import Audio, load_dataset
    # Only fetch this split's parquet shards. Passing (config, split) makes
    # recent `datasets` download every split of the config (the training
    # audio is tens of GB), even though only `split` is returned.
    ds = load_dataset(dataset_name, data_files={split: f"{config}/{split}/*.parquet"},
                      split=split)
    ds = ds.cast_column("audio", Audio(decode=False))
    if n_samples is not None and n_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n_samples))
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _run_inference(model, processor, ds, device) -> Tuple[List[str], List[str], List[str]]:
    model.eval()
    refs, hyps, ids = [], [], []
    for i, sample in enumerate(ds):
        audio, sr = _decode_audio(sample["audio"])
        if sr != 16_000:
            raise ValueError(f"utterance {sample.get('id')} has sr={sr}, expected 16000")
        inputs = processor(audio, sampling_rate=16_000, return_tensors="pt", padding=False)
        logits = model(inputs.input_values.to(device, dtype=torch.float32)).logits.float().cpu()
        refs.append(sample["text"].strip().upper())
        hyps.append(_ctc_greedy_decode(logits, processor))
        ids.append(str(sample.get("id", i)))
        if (i + 1) % 250 == 0:
            print(f"    [{i + 1}/{len(ds)}] running WER = {corpus_wer(refs, hyps):.2f}%")
    return refs, hyps, ids


# ─────────────────────────────────────────────────────────────────────────────
# Harness
# ─────────────────────────────────────────────────────────────────────────────

class SpeechEvalHarness:
    """
    Args:
        model_name:              HF model ID
        quant_mode:              see frameworks.language._resolve_modes
        n_samples:               None = whole split (default); int = seeded subset
        seed:                    subset seed (ignored for the whole split)
        librispeech_config:      "clean" or "other"
        skip_feature_extractor:  leave the conv feature extractor unquantised
        extra_skip_names:        added to the default skip list (lm_head)
    """

    DOCUMENTED_REFERENCE_WER = DOCUMENTED_REFERENCE_WER

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        quant_mode: str = "fp32",
        n_samples: Optional[int] = None,
        seed: int = 42,
        block_size: int = 32,
        device: Optional[torch.device] = None,
        skip_feature_extractor: bool = False,
        extra_skip_names: Optional[List[str]] = None,
        librispeech_config: str = "clean",
    ):
        self.model_name = model_name
        self.quant_mode = quant_mode
        self.n_samples = n_samples
        self.seed = seed
        self.block_size = block_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.skip_feature_extractor = skip_feature_extractor
        self.extra_skip_names = list(extra_skip_names or [])
        self.librispeech_config = librispeech_config

    def run(self) -> Dict[str, Any]:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        from core.compat import hf_dtype_kwargs
        from core.layers import configure_exact_numerics
        from core.quantizer import bits_per_value

        configure_exact_numerics()
        weight_mode, act_mode = _resolve_modes(self.quant_mode)
        skip_names = sorted(set(DEFAULT_SKIP_NAMES) | set(self.extra_skip_names))

        processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        model = Wav2Vec2ForCTC.from_pretrained(self.model_name, **hf_dtype_kwargs(torch.float32))
        model.to(self.device).eval()
        if self.quant_mode not in ("fp32", "bf16"):
            _replace_speech_layers(model, weight_mode, act_mode, self.block_size,
                                   skip_names, self.skip_feature_extractor)

        ds = _load_librispeech(self.n_samples, self.seed, self.librispeech_config)
        print(f"\n{'=' * 64}\n  {self.model_name} | {self.quant_mode} (w={weight_mode}, a={act_mode})"
              f"\n  LibriSpeech test-{self.librispeech_config}: {len(ds)} utterances"
              f" | skip={skip_names} | skip_FE={self.skip_feature_extractor}\n{'=' * 64}")

        refs, hyps, ids = _run_inference(model, processor, ds, self.device)
        edits = [_levenshtein(r.split(), h.split()) for r, h in zip(refs, hyps)]
        words = [len(r.split()) for r in refs]
        wer_pct = 100.0 * sum(edits) / sum(words)
        print(f"\n  WER = {wer_pct:.2f}%  (reference {DOCUMENTED_REFERENCE_WER:.1f}%)\n")

        del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "wer_pct": wer_pct,
            "per_item_edits": edits,
            "per_item_ref_words": words,
            "item_ids": ids,
            "references": refs,
            "hypotheses": hyps,
            "model_name": self.model_name,
            "quant_mode": self.quant_mode,
            "weight_mode": weight_mode,
            "act_mode": act_mode,
            "n_samples": len(refs),
            "librispeech_config": self.librispeech_config,
            "seed": self.seed,
            "skip_names": skip_names,
            "skip_feature_extractor": self.skip_feature_extractor,
            "eff_bits": bits_per_value(self.quant_mode),
        }
