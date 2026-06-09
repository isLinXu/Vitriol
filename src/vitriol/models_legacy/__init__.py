# ⚠️ DEPRECATED: This module is superseded by vitriol.adapters.
# Kept for backward compatibility only. Use vitriol.adapters instead.
from . import deepseek as deepseek
from . import llama as llama
from . import qwen as qwen
from .registry import ModelAdapter as ModelAdapter
from .registry import ModelRegistry as ModelRegistry

__all__ = [
    "deepseek",
    "llama",
    "qwen",
    "ModelAdapter",
    "ModelRegistry",
]
