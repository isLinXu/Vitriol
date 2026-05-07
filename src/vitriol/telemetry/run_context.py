from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


def new_run_id() -> str:
    """
    Generate a globally-unique run id.

    The id is intended to correlate outputs produced by:
    - vitriol.bench runner results (run_smoke / run_generate_preset)
    - vitriol.cli trace exports (trace.v1)
    """

    return str(uuid.uuid4())


@dataclass(frozen=True, slots=True)
class RunContext:
    """
    Minimal context shared across a single "run" (bench/trace/etc).

    Keep this small and serializable: we primarily use it to propagate run_id.
    """

    run_id: str = field(default_factory=new_run_id)
    created_at_s: float = field(default_factory=time.time)

