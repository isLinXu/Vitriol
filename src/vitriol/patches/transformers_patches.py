"""
Patches for transformers library compatibility.

This module provides patches to ensure compatibility with various
transformers versions and handle missing components.
"""

import importlib.util
import logging
import sys
from typing import Any

import transformers.modeling_utils
import transformers.utils.import_utils

logger = logging.getLogger(__name__)


def patch_transformers_generic() -> None:
    """
    Inject OutputRecorder and check_model_inputs if absent in some transformers builds.

    Some older or custom builds of transformers may not have these components,
    which can cause import errors. This function provides no-op implementations.
    """
    try:
        from transformers.utils import generic

        if not hasattr(generic, "OutputRecorder"):
            class OutputRecorder:
                def __init__(self, *args: Any, **kwargs: Any):
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, *args: Any) -> None:
                    pass

                def record(self, *args: Any, **kwargs: Any) -> None:
                    pass

            generic.OutputRecorder = OutputRecorder  # type: ignore[attr-defined]
            mod = sys.modules.get("transformers.utils.generic")
            if mod:
                mod.OutputRecorder = OutputRecorder

        if not hasattr(generic, "check_model_inputs"):
            def _check(*args: Any, **kwargs: Any) -> None:
                pass

            generic.check_model_inputs = _check  # type: ignore[attr-defined]
            mod = sys.modules.get("transformers.utils.generic")
            if mod:
                mod.check_model_inputs = _check

    except ImportError:
        pass


def patch_fx_available() -> None:
    """Ensure is_torch_fx_available is present in transformers."""
    if not hasattr(transformers.utils.import_utils, "is_torch_fx_available"):
        transformers.utils.import_utils.is_torch_fx_available = lambda: True


def patch_pretrained_init() -> None:
    """
    Force eager attention when flash_attn is absent.

    Some models default to FlashAttention 2, which requires the flash_attn library.
    This patch automatically detects if flash_attn is available and falls back
    to eager attention if not.
    """
    if getattr(transformers.modeling_utils.PreTrainedModel.__init__, "_vitriol_patched", False):
        return

    _fa2_available: bool = False
    _orig = transformers.modeling_utils.PreTrainedModel.__init__

    def _patched(self, config, *args: Any, **kwargs: Any) -> Any:
        nonlocal _fa2_available

        # Check flash_attn availability (cached)
        if not hasattr(_patched, '_checked'):
            _fa2_available = importlib.util.find_spec("flash_attn") is not None
            _patched._checked = True  # type: ignore[attr-defined]

        # Force eager attention if flash_attn not available
        if not _fa2_available:
            for attr in ("_attn_implementation", "attn_implementation"):
                if hasattr(config, attr):
                    try:
                        object.__setattr__(config, attr, "eager")
                    except Exception as e:
                        logger.debug("Could not set %s=eager on config: %s", attr, e)

            for attr in ("use_flash_attention_2", "use_flash_attn"):
                if hasattr(config, attr):
                    try:
                        object.__setattr__(config, attr, False)
                    except Exception as e:
                        logger.debug("Could not set %s=False on config: %s", attr, e)

        return _orig(self, config, *args, **kwargs)

    _patched._vitriol_patched = True  # type: ignore[attr-defined]
    transformers.modeling_utils.PreTrainedModel.__init__ = _patched
