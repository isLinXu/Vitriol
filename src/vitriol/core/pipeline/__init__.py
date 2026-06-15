"""Generator pipeline internals.

This package provides the internal pipeline abstraction used by
:class:`~vitriol.core.generator.MinimalWeightGenerator`.

The pipeline decouples generation into discrete, testable steps:

1. :class:`BootstrapStep` — initialise context, load config, resolve adapter
2. :class:`LegacyGenerateStep` — run the legacy shard-generation loop

Public API remains ``vitriol.core.generator.MinimalWeightGenerator``;
classes exported here are intended for advanced users who wish to
compose or extend the generation pipeline.
"""

from .steps.bootstrap import BootstrapStep
from .steps.legacy_generate import LegacyGenerateStep

# ResolveShardMapStep is intentionally not exported (internal use only).

__all__ = ["BootstrapStep", "LegacyGenerateStep"]

