"""
frameworks/vision.py
====================
ImageNet-1k validation harness (top-1 / top-5).

Models: torchvision resnet18 / resnet50 (DEFAULT weights and their own
transforms), timm deit_* / vit_* (pretrained, preprocessing taken from the
model's own pretrained config via timm.data.resolve_model_data_config).

Protocol (v2):
  - Dataset: ILSVRC/imagenet-1k (parquet, gated - accept the terms on the Hub
    and log in), split "validation". n_images=None (default) = all 50,000.
    An integer evaluates a seeded random subset.
  - Model runs in fp32; quantised matmuls/convs are exact (core/layers.py).
  - Per-image top-1 / top-5 correctness is returned for bootstrap CIs.

Layer policy: every nn.Conv2d and nn.Linear is quantised (including the
first conv / patch embedding) except the classifier head (`fc` for ResNet,
`head` for DeiT/ViT), matching lm_head being skipped for language and speech.

Note on the previous version: timm ViT/DeiT models were fed ImageNet mean/std
regardless of their pretrained config. The default timm vit_base_patch16_224
weights expect mean = std = 0.5, so the old ViT-B baseline was degraded.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

HEAD_NAMES = {"fc", "head", "head_dist", "classifier"}


def replace_vision_layers(
    model: nn.Module,
    weight_mode: str,
    act_mode: str,
    block_size: int = 32,
    skip_names: Optional[List[str]] = None,
    _prefix: str = "",
) -> nn.Module:
    """Replace Conv2d / Linear with FakeQuant equivalents in place.
    skip_names are dotted paths from the model root (default: classifier heads)."""
    from core.layers import FakeQuantConv, FakeQuantLinear

    skip_set = set(HEAD_NAMES if skip_names is None else skip_names)
    for name, child in list(model.named_children()):
        full_name = f"{_prefix}.{name}".lstrip(".")
        if full_name in skip_set:
            continue
        if isinstance(child, nn.Conv2d):
            setattr(model, name, FakeQuantConv(child, weight_mode, act_mode, block_size))
        elif isinstance(child, nn.Linear):
            setattr(model, name, FakeQuantLinear(child, weight_mode, act_mode, block_size))
        else:
            replace_vision_layers(child, weight_mode, act_mode, block_size,
                                  list(skip_set), full_name)
    return model


class _ImageTransform:
    """Picklable batch transform for datasets.with_transform (DataLoader
    workers may be started with 'spawn'/'forkserver', which cannot pickle
    lambdas or closures)."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, batch):
        batch["pixel_values"] = [self.transform(img.convert("RGB")) for img in batch["image"]]
        del batch["image"]
        return batch


def _collate(batch):
    return (torch.stack([x["pixel_values"] for x in batch]),
            torch.tensor([x["label"] for x in batch]))


def _build_model(model_name: str):
    """Return (model, preprocessing transform)."""
    if model_name in ("resnet18", "resnet50"):
        import torchvision.models as tvm
        weights = {"resnet18": tvm.ResNet18_Weights.DEFAULT,
                   "resnet50": tvm.ResNet50_Weights.DEFAULT}[model_name]
        model = getattr(tvm, model_name)(weights=weights)
        return model, weights.transforms(), str(weights)
    import timm
    from timm.data import create_transform, resolve_model_data_config
    model = timm.create_model(model_name, pretrained=True)
    cfg = resolve_model_data_config(model)
    transform = create_transform(**cfg, is_training=False)
    tag = (getattr(model, "pretrained_cfg", None) or {}).get("tag", "")
    return model, transform, f"timm:{model_name}.{tag} mean={cfg.get('mean')} std={cfg.get('std')}"


class VisionEvalHarness:
    """
    Args:
        model_name:  resnet18 | resnet50 | deit_* | vit_* (timm)
        quant_mode:  see frameworks.language._resolve_modes
        n_images:    None = all 50,000 validation images (default)
        n_batches:   legacy alternative to n_images (n_images = n_batches * batch_size)
        batch_size:  images per forward pass
        num_workers: DataLoader workers for JPEG decode + resize
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        quant_mode: str = "fp32",
        n_batches: Optional[int] = None,
        batch_size: int = 64,
        seed: int = 42,
        block_size: int = 32,
        device: Optional[torch.device] = None,
        n_images: Optional[int] = None,
        num_workers: int = 8,
        skip_names: Optional[List[str]] = None,
    ):
        self.model_name = model_name
        self.quant_mode = quant_mode
        self.batch_size = batch_size
        self.seed = seed
        self.block_size = block_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_images = n_images if n_images is not None else (
            n_batches * batch_size if n_batches is not None else None)
        self.num_workers = num_workers
        self.skip_names = skip_names

    def _dataset(self, transform):
        from datasets import load_dataset
        # Only the validation shards (~6.4 GB). split="validation" alone makes
        # recent `datasets` download the 294 training shards (~140 GB) too.
        ds = load_dataset("ILSVRC/imagenet-1k",
                          data_files={"validation": "data/validation-*.parquet"},
                          split="validation")
        if self.n_images is not None and self.n_images < len(ds):
            ds = ds.shuffle(seed=self.seed).select(range(self.n_images))
        return ds.with_transform(_ImageTransform(transform))

    @torch.no_grad()
    def run(self) -> Dict[str, Any]:
        from frameworks.language import _resolve_modes
        from core.layers import configure_exact_numerics
        from core.quantizer import bits_per_value

        configure_exact_numerics()
        weight_mode, act_mode = _resolve_modes(self.quant_mode)

        model, transform, weights_desc = _build_model(self.model_name)
        model.to(self.device).eval()
        if self.quant_mode not in ("fp32", "bf16"):
            replace_vision_layers(model, weight_mode, act_mode, self.block_size, self.skip_names)

        ds = self._dataset(transform)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
            collate_fn=_collate)

        print(f"\n{'=' * 64}\n  {self.model_name} | {self.quant_mode} (w={weight_mode}, a={act_mode})"
              f"\n  {weights_desc}\n  ImageNet val: {len(ds)} images"
              f" | skip={sorted(self.skip_names or HEAD_NAMES)}\n{'=' * 64}")

        top1: List[int] = []
        top5: List[int] = []
        n_batches = len(loader)
        for b, (x, y) in enumerate(loader):
            logits = model(x.to(self.device))
            pred = logits.topk(5, dim=1).indices.cpu()
            hit = pred.eq(y.view(-1, 1))
            top1.extend(hit[:, 0].int().tolist())
            top5.extend(hit.any(dim=1).int().tolist())
            if (b + 1) % max(1, n_batches // 10) == 0:
                print(f"    batch {b + 1}/{n_batches} | top1 {100 * sum(top1) / len(top1):.2f}%")

        acc1 = 100.0 * sum(top1) / len(top1)
        acc5 = 100.0 * sum(top5) / len(top5)
        print(f"\n  Acc@1 = {acc1:.2f}% | Acc@5 = {acc5:.2f}%\n")

        del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "acc1": acc1,
            "acc5": acc5,
            "per_item_top1": top1,
            "per_item_top5": top5,
            "total_samples": len(top1),
            "weights": weights_desc,
            "model_name": self.model_name,
            "quant_mode": self.quant_mode,
            "weight_mode": weight_mode,
            "act_mode": act_mode,
            "eff_bits": bits_per_value(self.quant_mode),
        }
