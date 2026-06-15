"""
Vitriol Resilience & Fault Tolerance Module
============================================

Provides checkpoint management, recovery mechanisms, and graceful
degradation for long-running generation and benchmarking tasks.

Features:
    - Incremental checkpointing with resume support
    - Automatic retry with exponential backoff
    - Disk-offload fallback for low-memory scenarios

Example::

    from vitriol.resilience import CheckpointManager
"""

from .checkpoint import CheckpointManager

__all__ = [
    "CheckpointManager",
]
