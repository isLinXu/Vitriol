"""
Mock detectron2 to prevent segfaults from multimodal models.

Some multimodal models (like ERNIE-VL) import detectron2 for vision processing,
which can cause segfaults in certain environments. This module provides a
no-op mock implementation.
"""

import logging
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Any, Optional

logger = logging.getLogger(__name__)


def mock_detectron2() -> Optional[Any]:
    """
    Prevent segfaults from multimodal models that import detectron2.

    This function creates a minimal mock of detectron2 module with no-op
    implementations, allowing the code to run without the actual detectron2
    library.

    Example:
        >>> from vitriol.patches import apply_all_patches
        >>> apply_all_patches()  # Includes detectron2 mock
    """
    if "detectron2" in sys.modules:
        return

    try:
        def _noop(*args: Any, **kwargs: Any) -> Optional[Any]:
            return None

        class _Stub:
            """Stub object that raises ImportError on attribute access."""
            def __init__(self, *args: Any, **kwargs: Any):
                pass

            def __call__(self, *args: Any, **kwargs: Any) -> Optional[Any]:
                return None

        # Create root module
        root = ModuleType("detectron2")
        root.__spec__ = ModuleSpec("detectron2", loader=None)

        # Create submodules
        for sub in ("layers", "config", "structures", "modeling", "data", "engine"):
            m = ModuleType(f"detectron2.{sub}")
            m.__spec__ = ModuleSpec(f"detectron2.{sub}", loader=None)
            sys.modules[f"detectron2.{sub}"] = m
            setattr(root, sub, m)

        # Add common functions/classes
        root.layers.nms_rotated = _noop  # type: ignore[attr-defined]
        root.layers.batched_nms = _noop  # type: ignore[attr-defined]
        root.layers.ROIAlign = _Stub  # type: ignore[attr-defined]
        root.config.get_cfg = _noop  # type: ignore[attr-defined]

        sys.modules["detectron2"] = root
        logger.info("detectron2 mocked successfully")

    except Exception as exc:
        logger.warning("Failed to mock detectron2: %s", exc)
