"""
Patches module for Vitriol.

This module contains all necessary patches for compatibility with various
transformers versions and model architectures.
"""

from .cache_hooks import CacheHookConfig, CacheHookPatcher, UniversalAttentionPatcher
from .detectron2_mock import mock_detectron2
from .dynamic_model_patches import patch_remote_module
from .kv_runtime_patches import KVRuntimePatchConfig, KVRuntimePatcher, patch_kv_runtime
from .model_family_patches import PatchRegistry
from .transformers_patches import (
    patch_fx_available,
    patch_pretrained_init,
    patch_transformers_generic,
)

__all__ = [
    "apply_all_patches",
    "mock_detectron2",
    "patch_remote_module",
    "patch_transformers_generic",
    "patch_fx_available",
    "patch_pretrained_init",
    "PatchRegistry",
    "KVRuntimePatchConfig",
    "KVRuntimePatcher",
    "patch_kv_runtime",
    "CacheHookConfig",
    "CacheHookPatcher",
    "UniversalAttentionPatcher",
]

_PATCHES_APPLIED = False


def apply_all_patches():
    """Apply all necessary patches for compatibility."""
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return

    mock_detectron2()
    patch_transformers_generic()
    patch_fx_available()
    patch_pretrained_init()
    _PATCHES_APPLIED = True
