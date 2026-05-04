from __future__ import annotations


from ...incremental import IncrementalGenerator
from ....strategies import get_strategy
from ..context import GenerationContext


def _parse_size(size_str: str) -> int:
    upper = size_str.upper()
    for unit, mult in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if upper.endswith(unit):
            return int(float(size_str[:-2]) * mult)
    return int(size_str)


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

