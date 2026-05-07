from __future__ import annotations

from typing import Any, Dict


def merge_kv_stats(*parts: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Merge multiple KV stats dicts into a single dict.

    Phase2 minimal behavior:
    - Shallow merge
    - Later parts override earlier ones on key conflict
    - None parts are ignored
    """
    out: Dict[str, Any] = {}
    for p in parts:
        for k, v in (p or {}).items():
            out[k] = v
    return out

