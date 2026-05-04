from __future__ import annotations

import asyncio
import sys
import types
from datetime import datetime, timezone


def test_api_generation_request_propagates_offline_flags(monkeypatch, tmp_path) -> None:
    """
    P2 end-to-end consistency: allow_network/local_files_only in the API request must be propagated
    into build_generation_config overrides.
    """
    from vitriol.api import server

    captured = {"overrides": None}

    def fake_build_generation_config(*, overrides=None, **kwargs):
        captured["overrides"] = dict(overrides or {})
        return {"_dummy_cfg": True, **captured["overrides"]}

    monkeypatch.setattr(server, "build_generation_config", fake_build_generation_config)

    # Stub vitriol.core.generator to avoid importing heavy deps
    gen_mod = types.ModuleType("vitriol.core.generator")

    class DummyGen:
        def __init__(self, model_id, output_dir, config, **kwargs):
            self.model_id = model_id
            self.output_dir = output_dir
            self.config = config

        def generate(self):
            return {"output_dir": self.output_dir}

    gen_mod.MinimalWeightGenerator = DummyGen
    monkeypatch.setitem(sys.modules, "vitriol.core.generator", gen_mod)

    job_id = "job-test-gen-1"
    server.active_jobs[job_id] = {
        "id": job_id,
        "type": "generation",
        "status": "queued",
        "request": {
            "model_id": "demo/model",
            "strategy": "compact",
            "dtype": "bfloat16",
            "max_shard_size": "5GB",
            "output_dir": str(tmp_path),
            "trust_remote_code": False,
            "allow_network": False,
            "local_files_only": True,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "progress": 0,
    }

    asyncio.run(server.process_generation_job(job_id))

    assert captured["overrides"]["trust_remote_code"] is False
    assert captured["overrides"]["allow_network"] is False
    assert captured["overrides"]["local_files_only"] is True
