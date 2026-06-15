"""
Vitriol Telemetry Module
========================

Provides runtime telemetry and observability for Vitriol operations.

Tracks execution context, performance metrics, and operational metadata
across generation, benchmarking, and inference tasks.

Example::

    from vitriol.telemetry import RunContext, new_run_id

    run_id = new_run_id()
    ctx = RunContext(run_id=run_id, task="generation")
"""

from __future__ import annotations

from .run_context import RunContext, new_run_id

__all__ = [
    "RunContext",
    "new_run_id",
]
