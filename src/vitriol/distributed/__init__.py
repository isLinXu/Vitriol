"""
Vitriol Distributed Generation Module
======================================

Provides Master-Worker distributed coordination for parallel weight
generation across multiple machines.

Features:
    - Async task dispatch via ``asyncio.Queue``
    - Heartbeat monitoring (30s interval)
    - Timeout retry (300s, max 3 attempts)
    - Worker capability reporting (GPU, memory)

Example::

    from vitriol.distributed import DistributedCoordinator, WorkerInfo, GenerationTask
"""

from .coordinator import DistributedCoordinator, GenerationTask, WorkerInfo

__all__ = [
    "DistributedCoordinator",
    "GenerationTask",
    "WorkerInfo",
]
