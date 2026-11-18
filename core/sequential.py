"""
core/sequential.py
==================
Layer-by-layer ("sequential") perplexity evaluation for decoder-only LLMs
whose full weights do not fit on the GPU (Llama-2-7B, Llama-3-8B, Mistral-7B
on a 16 GB card).

The model stays in CPU RAM. The embeddings run once for every chunk, then each
decoder layer is moved to the GPU, optionally quantised (layer_transform),
applied to the hidden states of ALL chunks, and discarded. Finally the norm
and lm_head produce the per-chunk NLL. The arithmetic is the same as a normal
forward pass; only the order of the loops changes (chunks inner, layers
outer). tests/test_eval_fixes.py::test_sequential_matches_full checks this
on a tiny random Llama.

Supported: models exposing model.model.{embed_tokens, layers, norm} and
model.lm_head (Llama, Mistral, Qwen2 and other Llama-style families).
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _StopForward(Exception):
    pass


class _Catcher(nn.Module):
    """Stands in for decoder layer 0, records its inputs, aborts the forward."""

    def __init__(self, inner: nn.Module, store: dict):
        super().__init__()
        self.inner = inner
        self.store = store
        # Some model classes read per-layer attributes while building masks.
        for attr in ("attention_type", "layer_idx", "is_sliding"):
            if hasattr(inner, attr):
                setattr(self, attr, getattr(inner, attr))

    def forward(self, hidden_states, *args, **kwargs):
        self.store["hidden"].append(hidden_states.detach())
        if self.store["args"] is None:
            self.store["args"] = args
            self.store["kwargs"] = kwargs
        raise _StopForward


def supports_sequential(model: nn.Module) -> bool:
    base = getattr(model, "model", None)
    return (base is not None and hasattr(base, "layers") and hasattr(base, "embed_tokens")
            and hasattr(base, "norm") and hasattr(model, "lm_head"))


@torch.no_grad()
def sequential_chunk_nll(
    model: nn.Module,
    chunks: List[torch.Tensor],
    device: torch.device,
    layer_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    log: Callable[[str], None] = print,
) -> List[float]:
    """Return the mean next-token NLL of every chunk (same as HF outputs.loss).

    All chunks must have the same length (the attention mask / rotary inputs
    captured from the first chunk are reused for the others).
    The model is consumed: its decoder layers are replaced by nn.Identity.
    """
    if not supports_sequential(model):
        raise ValueError("model does not expose model.model.layers / embed_tokens / norm")
    lengths = {int(c.numel()) for c in chunks}
    if len(lengths) != 1:
        raise ValueError(f"sequential evaluation needs equal-length chunks, got {sorted(lengths)}")

    model.eval()
    base = model.model
    layers = base.layers

    base.embed_tokens.to(device)
    if hasattr(base, "rotary_emb"):
        base.rotary_emb.to(device)
    base.norm.to(device)
    model.lm_head.to(device)

    # ── 1. embeddings + layer-0 inputs for every chunk ──────────────────────
    store = {"hidden": [], "args": None, "kwargs": None}
    layers[0] = _Catcher(layers[0], store)
    try:
        for ch in chunks:
            try:
                model(ch.unsqueeze(0).to(device), use_cache=False)
            except _StopForward:
                pass
    finally:
        layers[0] = layers[0].inner
    hidden = store["hidden"]
    args, kwargs = store["args"], store["kwargs"]
    log(f"    sequential: captured {len(hidden)} chunks, running {len(layers)} layers")

    # ── 2. layer by layer ───────────────────────────────────────────────────
    for i in range(len(layers)):
        layer = layers[i].to(device)
        if layer_transform is not None:
            layer = layer_transform(layer)
        for j in range(len(hidden)):
            out = layer(hidden[j], *args, **kwargs)
            hidden[j] = out[0] if isinstance(out, (tuple, list)) else out
        layers[i] = nn.Identity()
        del layer
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if (i + 1) % 8 == 0 or i + 1 == len(layers):
            log(f"    sequential: layer {i + 1}/{len(layers)} done")

    # ── 3. norm + lm_head + NLL ─────────────────────────────────────────────
    nlls: List[float] = []
    for j, ch in enumerate(chunks):
        logits = model.lm_head(base.norm(hidden[j])).float()       # [1, L, V]
        labels = ch.to(device)
        loss = F.cross_entropy(logits[0, :-1, :], labels[1:], reduction="mean")
        nlls.append(float(loss.item()))
        hidden[j] = None
    return nlls
