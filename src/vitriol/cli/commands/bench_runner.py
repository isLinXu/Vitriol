"""Bench runner and result collector."""

import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any


def run_bench(
    model_id: str,
    preset: str,
    mode: str,
    output_dir: str,
    shard_range: Optional[str] = None,
) -> Dict[str, Any]:
    """Run benchmark with given model and strategy."""
    result = {
        "model_id": model_id,
        "preset": preset,
        "mode": mode,
        "output_dir": output_dir,
        "shard_range": shard_range,
    }
    return result
