from __future__ import annotations

import asyncio
import sys
import types
from datetime import datetime, timezone


def test_api_generation_job_includes_security_context(monkeypatch, tmp_path) -> None:
    """
    P4: auditability — API generation job results must include security_context (with provenance).
    """
    from vitriol.api import server

    # Stub generator to avoid importing heavy deps
    gen_mod = types.ModuleType("vitriol.core.generator")

    class DummyGen:
        def __init__(self, model_id, output_dir, config, **kwargs):
            self.output_dir = output_dir
            self.config = config

        def generate(self):
            return {"output_dir": self.output_dir}

    gen_mod.MinimalWeightGenerator = DummyGen
    monkeypatch.setitem(sys.modules, "vitriol.core.generator", gen_mod)

    job_id = "job-test-gen-secctx-1"
    server.active_jobs[job_id] = {
        "id": job_id,
        "type": "generation",
        "status": "queued",
        "request": {
            "model_id": "demo/model",
            "strategy": "compact",
            "dtype": "bfloat16",
            "output_dir": str(tmp_path),
            "trust_remote_code": False,
            "allow_network": False,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "progress": 0,
    }

    asyncio.run(server.process_generation_job(job_id))
    result = server.active_jobs[job_id]["result"]
    sec = result.get("security_context") or {}
    assert sec.get("allow_network") is False
    assert sec.get("local_files_only") is True
    assert isinstance(sec.get("provenance"), dict)
