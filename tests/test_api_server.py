import asyncio
import logging
import sys

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("pydantic")
pytest.importorskip("uvicorn")
TestClient = pytest.importorskip("fastapi.testclient").TestClient

from vitriol.api import server  # noqa: E402
from vitriol.config.settings import init_config  # noqa: E402


@pytest.fixture(autouse=True)
def reset_api_state():
    server.active_jobs.clear()
    while not server.job_queue.empty():
        try:
            server.job_queue.get_nowait()
        except Exception:
            break
    yield
    server.active_jobs.clear()
    while not server.job_queue.empty():
        try:
            server.job_queue.get_nowait()
        except Exception:
            break


async def _noop_background_job(*_args, **_kwargs):
    return None


class StubGenerationResult:
    def __init__(self, output_dir="/tmp/out", total_size=4096):
        self.output_dir = output_dir
        self.manifest_path = f"{output_dir}/vitriol-manifest.json"
        self.index_path = f"{output_dir}/model.safetensors.index.json"
        self.total_size = total_size
        self.generated_at = "2026-04-04T00:00:00Z"

    def to_dict(self):
        return {
            "output_dir": self.output_dir,
            "manifest_path": self.manifest_path,
            "index_path": self.index_path,
            "total_size": self.total_size,
            "generated_at": self.generated_at,
        }


def test_generate_allows_missing_api_key_when_auth_disabled(monkeypatch):
    cfg = init_config()
    cfg.set("security.api_key_required", False)
    monkeypatch.setattr(server, "process_generation_job", _noop_background_job)
    client = TestClient(server.app)

    response = client.post(
        "/generate",
        json={"model_id": "demo/model", "strategy": "compact", "dtype": "bfloat16"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "queued"
    assert "job_id" in body


def test_generate_rejects_missing_api_key_when_auth_enabled():
    cfg = init_config()
    cfg.set("security.api_key_required", True)
    cfg.set("security.api_keys", ["secret-key"])
    client = TestClient(server.app)

    response = client.post(
        "/generate",
        json={"model_id": "demo/model", "strategy": "compact", "dtype": "bfloat16"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key"


def test_public_root_and_health_stay_open_when_auth_enabled():
    cfg = init_config()
    cfg.set("security.api_key_required", True)
    cfg.set("security.api_keys", ["secret-key"])
    client = TestClient(server.app)

    assert client.get("/").status_code == 200
    assert client.get("/health").status_code == 200


@pytest.mark.parametrize(
    "path",
    [
        "/status",
        "/jobs",
        "/jobs/missing",
        "/batch/missing",
        "/models",
        "/models/families",
        "/models/adapters",
        "/strategies",
        "/stream/logs",
    ],
)
def test_sensitive_get_endpoints_require_api_key_when_auth_enabled(path):
    cfg = init_config()
    cfg.set("security.api_key_required", True)
    cfg.set("security.api_keys", ["secret-key"])
    client = TestClient(server.app)

    response = client.get(path)

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key"


def test_generate_accepts_x_api_key_header_when_auth_enabled(monkeypatch):
    cfg = init_config()
    cfg.set("security.api_key_required", True)
    cfg.set("security.api_keys", ["secret-key"])
    monkeypatch.setattr(server, "process_generation_job", _noop_background_job)
    client = TestClient(server.app)

    response = client.post(
        "/generate",
        headers={"X-API-Key": "secret-key"},
        json={"model_id": "demo/model", "strategy": "compact", "dtype": "bfloat16"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "queued"


def test_status_accepts_x_api_key_header_when_auth_enabled():
    cfg = init_config()
    cfg.set("security.api_key_required", True)
    cfg.set("security.api_keys", ["secret-key"])
    client = TestClient(server.app)

    response = client.get("/status", headers={"X-API-Key": "secret-key"})

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_generate_accepts_bearer_token_when_auth_enabled(monkeypatch):
    cfg = init_config()
    cfg.set("security.api_key_required", True)
    cfg.set("security.api_keys", ["secret-key"])
    monkeypatch.setattr(server, "process_generation_job", _noop_background_job)
    client = TestClient(server.app)

    response = client.post(
        "/generate",
        headers={"Authorization": "Bearer secret-key"},
        json={"model_id": "demo/model", "strategy": "compact", "dtype": "bfloat16"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "queued"


def test_process_generation_job_persists_generation_result(monkeypatch):
    server.active_jobs["job-1"] = {
        "id": "job-1",
        "type": "generation",
        "status": "queued",
        "request": {"model_id": "demo/model", "output_dir": "/tmp/out"},
    }

    class StubGenerator:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self):
            return StubGenerationResult()

    monkeypatch.setattr("vitriol.core.generator.MinimalWeightGenerator", StubGenerator)

    asyncio.run(server.process_generation_job("job-1"))

    assert server.active_jobs["job-1"]["status"] == "completed"
    assert server.active_jobs["job-1"]["output_dir"] == "/tmp/out"
    assert server.active_jobs["job-1"]["result"]["total_size"] == 4096


def test_process_batch_job_uses_generation_result_without_dict_get(monkeypatch):
    request = server.BatchGenerateRequest(
        models=[{"model_id": "demo/a", "output_dir": "/tmp/a"}],
        default_strategy="compact",
        default_dtype="bfloat16",
        parallel=False,
    )
    server.active_jobs["batch-1"] = {
        "id": "batch-1",
        "type": "batch",
        "status": "queued",
        "total_models": 1,
        "completed": 0,
        "failed": 0,
        "results": [],
        "request": request.model_dump() if hasattr(request, "model_dump") else request.dict(),
    }

    class StubGenerator:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self):
            return StubGenerationResult(output_dir="/tmp/a", total_size=8192)

    monkeypatch.setattr("vitriol.core.generator.MinimalWeightGenerator", StubGenerator)

    asyncio.run(server.process_batch_job("batch-1", request))

    result = server.active_jobs["batch-1"]["results"][0]
    assert server.active_jobs["batch-1"]["completed"] == 1
    assert result["status"] == "success"
    assert result["output_dir"] == "/tmp/a"
    assert result["total_size"] == 8192
    assert result["size_mb"] == pytest.approx(8192 / (1024 * 1024))


def test_log_stream_queue_keeps_exception_records():
    handler = server.LogStreamQueue()

    try:
        raise RuntimeError("boom")
    except RuntimeError:
        record = logging.LogRecord(
            "vitriol.test",
            logging.ERROR,
            __file__,
            1,
            "failed",
            (),
            sys.exc_info(),
        )

    handler.emit(record)

    logs = handler.get_logs()
    assert len(logs) == 1
    assert logs[0]["message"] == "failed"
    assert "RuntimeError: boom" in logs[0]["exception"]


def test_rate_limiter_ignores_proxy_headers_by_default(monkeypatch):
    monkeypatch.delenv("VITRIOL_API_TRUST_PROXY_HEADERS", raising=False)
    limiter = server.RateLimiter()

    request = fastapi.Request(
        {
            "type": "http",
            "headers": [(b"x-forwarded-for", b"203.0.113.9")],
            "client": ("127.0.0.1", 12345),
        }
    )

    assert limiter._get_client_ip(request) == "127.0.0.1"


def test_rate_limiter_can_trust_proxy_headers_when_enabled(monkeypatch):
    monkeypatch.setenv("VITRIOL_API_TRUST_PROXY_HEADERS", "1")
    limiter = server.RateLimiter()

    request = fastapi.Request(
        {
            "type": "http",
            "headers": [(b"x-forwarded-for", b"203.0.113.9, 10.0.0.2")],
            "client": ("127.0.0.1", 12345),
        }
    )

    assert limiter._get_client_ip(request) == "203.0.113.9"


def test_api_main_binds_localhost_by_default(monkeypatch):
    captured = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.delenv("VITRIOL_API_HOST", raising=False)
    monkeypatch.delenv("VITRIOL_API_PORT", raising=False)
    monkeypatch.setattr(server.uvicorn, "run", fake_run)

    server.main()

    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8000


def test_status_reports_process_uptime_not_boot_timestamp(monkeypatch):
    cfg = init_config()
    cfg.set("security.api_key_required", False)
    monkeypatch.setattr(server, "_APP_STARTED_AT", 100.0)
    monkeypatch.setattr(server.time, "monotonic", lambda: 142.5)
    client = TestClient(server.app)

    response = client.get("/status")

    assert response.status_code == 200
    body = response.json()
    assert body["uptime"] == pytest.approx(42.5)
    assert body["system_info"]["system_boot_time"] > body["uptime"]


def test_process_batch_job_records_missing_model_id_as_item_failure():
    request = server.BatchGenerateRequest(
        models=[{}],
        default_strategy="compact",
        default_dtype="bfloat16",
        parallel=False,
    )
    server.active_jobs["batch-missing-model"] = {
        "id": "batch-missing-model",
        "type": "batch",
        "status": "queued",
        "total_models": 1,
        "completed": 0,
        "failed": 0,
        "results": [],
        "request": request.model_dump() if hasattr(request, "model_dump") else request.dict(),
    }

    asyncio.run(server.process_batch_job("batch-missing-model", request))

    job = server.active_jobs["batch-missing-model"]
    assert job["status"] == "completed_with_errors"
    assert job["completed"] == 0
    assert job["failed"] == 1
    assert job["progress"] == 100
    assert job["results"][0]["status"] == "failed"
    assert "model_id is required" in job["results"][0]["error"]


def test_batch_generate_rejects_parallel_requests():
    cfg = init_config()
    cfg.set("security.api_key_required", False)
    client = TestClient(server.app)

    response = client.post(
        "/batch/generate",
        json={
            "models": [{"model_id": "demo/model"}],
            "default_strategy": "compact",
            "default_dtype": "bfloat16",
            "parallel": True,
        },
    )

    assert response.status_code == 400
    assert "parallel batch generation is not supported yet" in response.json()["detail"]


def test_process_generation_job_respects_request_trust_remote_code(monkeypatch):
    captured = {}
    server.active_jobs["job-trc"] = {
        "id": "job-trc",
        "type": "generation",
        "status": "queued",
        "request": {
            "model_id": "demo/model",
            "output_dir": "/tmp/out",
            "trust_remote_code": False,
        },
    }

    def fake_build_generation_config(*, overrides=None, config_path=None):
        captured.update(overrides or {})
        assert config_path is None
        return object()

    class StubGenerator:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self):
            return StubGenerationResult()

    monkeypatch.setattr(server, "build_generation_config", fake_build_generation_config)
    monkeypatch.setattr("vitriol.core.generator.MinimalWeightGenerator", StubGenerator)

    asyncio.run(server.process_generation_job("job-trc"))

    assert captured["trust_remote_code"] is False


def test_process_batch_job_respects_request_trust_remote_code(monkeypatch):
    captured = {}
    request = server.BatchGenerateRequest(
        models=[{"model_id": "demo/a", "output_dir": "/tmp/a"}],
        default_strategy="compact",
        default_dtype="bfloat16",
        parallel=False,
        trust_remote_code=False,
    )
    server.active_jobs["batch-trc"] = {
        "id": "batch-trc",
        "type": "batch",
        "status": "queued",
        "total_models": 1,
        "completed": 0,
        "failed": 0,
        "results": [],
        "request": request.model_dump() if hasattr(request, "model_dump") else request.dict(),
    }

    def fake_build_generation_config(*, overrides=None, config_path=None):
        captured.update(overrides or {})
        assert config_path is None
        return object()

    class StubGenerator:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self):
            return StubGenerationResult(output_dir="/tmp/a", total_size=8192)

    monkeypatch.setattr(server, "build_generation_config", fake_build_generation_config)
    monkeypatch.setattr("vitriol.core.generator.MinimalWeightGenerator", StubGenerator)

    asyncio.run(server.process_batch_job("batch-trc", request))

    assert captured["trust_remote_code"] is False
