from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from ..context import GenerationContext


@dataclass
class ResolveShardMapStep:
    """Resolve and normalize shard map.

    Offline-friendly mode is supported for unit tests via `_use_ctx_original_map_only`.
    In normal runs, shard map resolution remains inside legacy implementation
    until we fully migrate generator internals into pipeline steps.
    """

    _use_ctx_original_map_only: bool = False

    name: str = "resolve_shard_map"

    def run(self, ctx: GenerationContext) -> None:
        if self._use_ctx_original_map_only:
            original_map = dict(ctx.original_shard_map or {})
        else:
            # Not yet migrated: keep behavior unchanged by not re-fetching here.
            original_map = dict(ctx.original_shard_map or {})

        if not original_map:
            ctx.original_shard_map = {}
            ctx.expected_shards = []
            return

        unique_shards = sorted(set(original_map.values()))
        total_shards = len(unique_shards)

        # Normalize filenames to -NNNNN-of-MMMMM format following existing generator behavior:
        # enumerate unique shards in sorted order and rewrite indices to 1..total_shards.
        norm_table: Dict[str, str] = {}
        for seq_idx, filename in enumerate(unique_shards, start=1):
            ext = "bin" if filename.endswith(".bin") else "safetensors"
            prefix = "pytorch_model" if "pytorch_model" in filename else "model"
            norm_table[filename] = f"{prefix}-{seq_idx:05d}-of-{total_shards:05d}.{ext}"

        ctx.original_shard_map = {p: norm_table.get(f, f) for p, f in original_map.items()}
        ctx.expected_shards = list(norm_table.values())
