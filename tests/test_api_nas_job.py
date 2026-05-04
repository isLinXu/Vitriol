from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path


def _install_torch_stub(monkeypatch) -> None:
    import sys
    import types

    if "torch" in sys.modules:
        return

    torch_stub = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    torch_stub.cuda = _Cuda()

    def _device(_type: str):
        class _Dev:
            def __init__(self, t: str):
                self.type = t

        return _Dev(_type)

    torch_stub.device = _device
    torch_stub.float32 = object()
    torch_stub.float16 = object()
    torch_stub.bfloat16 = object()

    monkeypatch.setitem(sys.modules, "torch", torch_stub)


def test_process_nas_job_produces_non_placeholder_result(monkeypatch, tmp_path: Path) -> None:
    """
    The API NAS job should not be "sleep + fake result"; it should run an actual search and produce
    best_gene/best_metrics.
    """
    _install_torch_stub(monkeypatch)

    # Avoid real sleep
    async def _noop_sleep(_seconds: float):
        return None

    from vitriol.api import server

    monkeypatch.setattr(server.asyncio, "sleep", _noop_sleep)

    job_id = "job-test-nas-1"
    server.active_jobs[job_id] = {
        "id": job_id,
        "type": "nas",
        "status": "queued",
        "request": {"algorithm": "evolutionary", "n_iterations": 10, "population_size": 5, "dataset": None},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "progress": 0,
        "artifacts_dir": str(tmp_path),
    }

    asyncio.run(server.process_nas_job(job_id))

    job = server.active_jobs[job_id]
    assert job["status"] == "completed"
    assert job["progress"] == 100

    result = job.get("result") or {}
    assert "best_gene" in result
    assert "best_metrics" in result
    assert result["best_gene"] != {"layers": 24, "hidden_size": 1024}
