from __future__ import annotations

from ....strategies import get_strategy
from ....utils.size import parse_size_to_bytes as _parse_size
from ...incremental import IncrementalGenerator
from ..context import GenerationContext


class BootstrapStep:
    """Populate common runtime fields in context.

    This step is intentionally lightweight and offline-friendly.
    """

    name = "bootstrap"

    def run(self, ctx: GenerationContext) -> None:
        ctx.incremental = IncrementalGenerator(ctx.output_dir)
        ctx.strategy = get_strategy(
            ctx.config.strategy,
            n_bits=ctx.config.n_bits,
            rank=ctx.config.rank,
            sparsity=ctx.config.sparsity,
        )
        ctx.max_shard_size = _parse_size(ctx.config.max_shard_size)

        if ctx.shrink_config is None:
            ctx.shrink_config = ctx.config.strategy in ("ultra", "hybrid_ultra")
