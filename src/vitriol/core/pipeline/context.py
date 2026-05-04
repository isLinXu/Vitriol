from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ...config.manager import GenerationConfig


@dataclass
class GenerationContext:
    model_id: str
    output_dir: str
    config: GenerationConfig

    # Optional back-reference to the generator instance (legacy wrapper stage)
    generator: Any = None

    # Derived / runtime fields (filled by steps)
    shrink_config: Optional[bool] = None
    strategy: Any = None
    max_shard_size: int = 0

    hf_config: Any = None
    adapter: Any = None
    model_empty: Any = None

    original_shard_map: Dict[str, str] = field(default_factory=dict)
    expected_shards: List[str] = field(default_factory=list)

    incremental: Any = None
    checkpoint: Dict[str, Any] = field(default_factory=dict)
    generated_param_names: Set[str] = field(default_factory=set)
    total_size: int = 0
    shard_map: Dict[str, str] = field(default_factory=dict)

    shard_buffers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    buf_bytes: Dict[str, int] = field(default_factory=dict)
    flushed_shards: Set[str] = field(default_factory=set)
    shard_count: int = 0

