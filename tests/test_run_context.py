from __future__ import annotations

import uuid


def test_new_run_id_is_uuid4() -> None:
    from vitriol.telemetry.run_context import new_run_id

    run_id = new_run_id()
    parsed = uuid.UUID(str(run_id))
    assert parsed.version == 4


def test_run_context_default_fields() -> None:
    from vitriol.telemetry.run_context import RunContext

    ctx = RunContext()
    assert isinstance(ctx.run_id, str) and ctx.run_id
    uuid.UUID(ctx.run_id)  # should parse
    assert isinstance(ctx.created_at_s, float)
    assert ctx.created_at_s > 0.0

