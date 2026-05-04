from __future__ import annotations

from typing import Any, Dict, Optional


def build_manifest(
    *,
    schema_version: int,
    generated_at: str,
    source: Dict[str, Any],
    environment: Dict[str, Any],
    security: Dict[str, Any],
    security_context: Optional[Dict[str, Any]],
    generation: Dict[str, Any],
    artifacts: Dict[str, Any],
    loadability: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the contents of vitriol-manifest.json.

    This module intentionally keeps dependencies light so it can be tested and reused
    even in environments without torch/transformers.
    """
    manifest: Dict[str, Any] = {
        "schema_version": int(schema_version),
        "generated_at": str(generated_at),
        "source": dict(source or {}),
        "environment": dict(environment or {}),
        "security": dict(security or {}),
        "generation": dict(generation or {}),
        "artifacts": dict(artifacts or {}),
        "loadability": dict(loadability or {}),
    }

    if security_context is not None:
        manifest["security_context"] = dict(security_context)

    return manifest
