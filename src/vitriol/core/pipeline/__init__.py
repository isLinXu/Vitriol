"""Generator pipeline internals (experimental).

This package provides an internal pipeline abstraction used by the weight generator.
Public API remains `vitriol.core.generator.MinimalWeightGenerator`.

.. warning::
   This package is experimental and may change or be removed in future versions.
"""

from .steps.bootstrap import BootstrapStep
from .steps.legacy_generate import LegacyGenerateStep

# ResolveShardMapStep is intentionally not exported (internal use only).

__all__ = ["BootstrapStep", "LegacyGenerateStep"]

