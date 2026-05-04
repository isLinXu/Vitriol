"""
CLI command modules.

This package intentionally avoids importing command implementations at module
import time so the top-level CLI can stay lightweight and lazy-load heavy
dependencies only when needed.
"""

__all__ = [
    "infer",
    "generate",
    "trace",
    "validate",
    "analyze",
    "batch",
    "bench_group",
    "export",
    "visualize",
    "arch_viz",
    "nas",
    "vocab_viz",
    "weight_viz",
    "evolve_group",
    "launch_webui",
]
