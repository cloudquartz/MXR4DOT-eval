"""
core/compat.py
==============
Small helpers that keep the harnesses working across library versions, and
record the exact environment next to every result.
"""

from __future__ import annotations

import platform
import subprocess
from typing import Any, Dict


def hf_dtype_kwargs(dtype) -> Dict[str, Any]:
    """transformers >= 4.56 renamed from_pretrained(torch_dtype=) to dtype=.
    Passing the wrong one is silently ignored by old versions (model loads in
    fp32), so pick by version instead of trying both."""
    import transformers
    from packaging.version import Version
    if Version(transformers.__version__) >= Version("4.56.0"):
        return {"dtype": dtype}
    return {"torch_dtype": dtype}


def _git_hash() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=3)
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, timeout=3)
        h = r.stdout.strip() if r.returncode == 0 else "unknown"
        return h + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


def environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"python": platform.python_version(), "git": _git_hash()}
    for mod in ("torch", "torchvision", "transformers", "datasets", "timm",
                "numpy", "accelerate", "lm_eval", "soundfile"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            info[mod] = None
    try:
        import torch
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
        try:
            info["matmul_fp32_precision"] = torch.backends.cuda.matmul.fp32_precision
            info["conv_fp32_precision"] = torch.backends.cudnn.conv.fp32_precision
        except AttributeError:
            info["matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
            info["cudnn_allow_tf32"] = torch.backends.cudnn.allow_tf32
    except Exception:
        pass
    return info
