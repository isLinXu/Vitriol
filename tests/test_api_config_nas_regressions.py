"""Regression coverage for API defaults, configuration precedence and NAS scoring."""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vitriol.config.manager import build_generation_config


def test_yaml_overrides_environment_and_explicit_overrides_yaml(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("default:\n  dtype: float16\n  max_shard_size: 1GB\n")
    monkeypatch.setenv("VITRIOL_DTYPE", "float32")
    monkeypatch.setenv("VITRIOL_MAX_SHARD_SIZE", "2GB")
    config = build_generation_config(config_path=path)
    assert (config.dtype, config.max_shard_size) == ("float16", "1GB")
    assert build_generation_config(config_path=path, overrides={"dtype": "bfloat16"}).dtype == "bfloat16"


def test_nested_security_override_is_applied_and_flat_override_wins(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    config = build_generation_config(overrides={"security": {"trust_remote_code": True}})
    assert config.security.trust_remote_code is True
    assert config.security_context["provenance"]["trust_remote_code"] == "explicit"
    config = build_generation_config(overrides={
        "security": {"trust_remote_code": True}, "trust_remote_code": False,
    })
    assert config.security.trust_remote_code is False


@pytest.fixture
def api(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    from vitriol.api import server
    monkeypatch.setattr(server, "active_jobs", {})
    monkeypatch.setattr(server, "_rate_limiter", server.RateLimiter())
    return server


def test_rate_limit_returns_http_429(api):
    from fastapi.testclient import TestClient
    api._rate_limiter._max_requests = 1
    client = TestClient(api.app, raise_server_exceptions=False)
    assert client.get("/health").status_code == 200
    response = client.get("/health")
    assert response.status_code == 429
    assert int(response.headers["retry-after"]) >= 1


def test_generate_request_defaults_preserve_network_and_output(api, monkeypatch):
    captured = {}
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    class Generator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def generate(self):
            return {"output_dir": captured["output_dir"]}

    monkeypatch.setattr("vitriol.core.generator.MinimalWeightGenerator", Generator)
    api.active_jobs["defaults"] = {"request": api.GenerateRequest(model_id="demo/model").model_dump()}
    asyncio.run(api.process_generation_job("defaults"))
    assert api.active_jobs["defaults"]["status"] == "completed"
    assert captured["output_dir"] == "./output/model"
    assert captured["config"].security.allow_network is True
    assert captured["config"].security.trust_remote_code is False


@pytest.mark.parametrize("batch", [False, True])
def test_generation_does_not_block_event_loop(api, monkeypatch, batch):
    started = threading.Event()
    release = threading.Event()
    worker_thread = []
    main_thread = threading.get_ident()

    class Generator:
        def __init__(self, **kwargs):
            worker_thread.append(threading.get_ident())
            started.set()
            release.wait(timeout=2)

        def generate(self):
            return {"output_dir": "/tmp/fake", "total_size": 0}

    monkeypatch.setattr("vitriol.core.generator.MinimalWeightGenerator", Generator)
    monkeypatch.setattr(api, "build_generation_config", lambda **kwargs: SimpleNamespace(security_context={}))
    request = api.BatchGenerateRequest(models=[{"model_id": "demo/model"}])
    api.active_jobs["responsive"] = {
        "request": {"model_id": "demo/model"}, "results": [], "completed": 0, "failed": 0,
    }

    async def run():
        coro = api.process_batch_job("responsive", request) if batch else api.process_generation_job("responsive")
        task = asyncio.create_task(coro)
        try:
            for _ in range(100):
                if started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert started.is_set()
            assert not task.done(), "generation blocked the event loop until completion"
        finally:
            release.set()
            await task

    asyncio.run(run())
    assert len(worker_thread) == 1
    assert worker_thread[0] != main_thread


def test_constrained_search_uses_evaluator_scores():
    from vitriol.nas.search_space import LLMSearchSpace
    from vitriol.nas.targeted_nas import ConstraintOptimizer, ObjectiveType, OptimizationTarget
    first, second = LLMSearchSpace().sample(), LLMSearchSpace().sample()
    space = SimpleNamespace(sample=Mock(side_effect=[first, second]))
    evaluator = SimpleNamespace(evaluate=Mock(side_effect=[{"score": 0.1}, {"score": 0.9}]))
    opt = ConstraintOptimizer(objectives=[OptimizationTarget(ObjectiveType.MAXIMIZE_SCORE)])
    gene, score, metrics = opt.optimize(space, evaluator, n_iterations=2, verbose=False)
    assert evaluator.evaluate.call_count == 2
    assert gene is second
    assert score == 0.9
    assert metrics["nas_score"] == 0.9


def test_pareto_search_uses_evaluator_scores():
    from vitriol.nas.search_space import LLMSearchSpace
    from vitriol.nas.targeted_nas import MultiObjectiveOptimizer, ObjectiveType
    first, second = LLMSearchSpace().sample(), LLMSearchSpace().sample()
    space = SimpleNamespace(sample=Mock(side_effect=[first, second]))
    evaluator = SimpleNamespace(evaluate=Mock(side_effect=[{"score": 0.1}, {"score": 0.9}]))
    front = MultiObjectiveOptimizer([ObjectiveType.MAXIMIZE_SCORE]).optimize(
        space, evaluator, n_iterations=2, verbose=False,
    )
    assert evaluator.evaluate.call_count == 2
    assert len(front) == 1
    assert front[0].gene is second
    assert front[0].metrics["nas_score"] == 0.9


def test_metric_only_search_has_no_fabricated_quality_score():
    from vitriol.nas.search_space import LLMSearchSpace
    from vitriol.nas.targeted_nas import ConstraintOptimizer, ObjectiveType, OptimizationTarget
    opt = ConstraintOptimizer(objectives=[OptimizationTarget(ObjectiveType.MAXIMIZE_SCORE)])
    _, score, metrics = opt.optimize(LLMSearchSpace(), None, n_iterations=1, verbose=False)
    assert score == 0.0
    assert metrics["nas_score"] == 0.0


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_search_rejects_nonfinite_quality_scores(score):
    from vitriol.nas.search_space import LLMSearchSpace
    from vitriol.nas.targeted_nas import ConstraintOptimizer
    evaluator = SimpleNamespace(evaluate=lambda gene: {"score": score})
    with pytest.raises(ValueError, match="finite"):
        ConstraintOptimizer().optimize(LLMSearchSpace(), evaluator, n_iterations=1, verbose=False)


def test_status_counts_only_current_running_and_queued_jobs(api):
    api.active_jobs.update({
        "running": {"status": "running", "type": "generation"},
        "batch": {"status": "queued", "type": "batch"},
        "nas": {"status": "queued", "type": "nas"},
        "done": {"status": "completed"},
        "failed": {"status": "failed"},
        "partial": {"status": "completed_with_errors"},
    })
    status = asyncio.run(api.get_status())
    assert status.active_jobs == 1
    assert status.queue_size == 2
    api.active_jobs["running"]["status"] = "completed"
    api.active_jobs["batch"]["status"] = "completed_with_errors"
    api.active_jobs["nas"]["status"] = "failed"
    status = asyncio.run(api.get_status())
    assert status.active_jobs == 0
    assert status.queue_size == 0


def test_batch_string_false_does_not_enable_remote_code(api, monkeypatch):
    captured = {}

    class Generator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def generate(self):
            return {"total_size": 0}

    monkeypatch.setattr("vitriol.core.generator.MinimalWeightGenerator", Generator)
    request = api.BatchGenerateRequest(models=[{"model_id": "demo/model", "trust_remote_code": "false"}])
    api.active_jobs["batch-trust"] = {"results": [], "completed": 0, "failed": 0}
    asyncio.run(api.process_batch_job("batch-trust", request))
    assert api.active_jobs["batch-trust"]["status"] == "completed"
    assert captured["config"].security.trust_remote_code is False


def test_concurrent_generation_remains_serialized(api, monkeypatch):
    first_started = threading.Event()
    release_first = threading.Event()
    starts = []

    class Generator:
        def __init__(self, model_id, **kwargs):
            starts.append(model_id)
            if model_id == "first":
                first_started.set()
                release_first.wait(timeout=2)

        def generate(self):
            return {}

    monkeypatch.setattr("vitriol.core.generator.MinimalWeightGenerator", Generator)
    monkeypatch.setattr(api, "build_generation_config", lambda **kwargs: SimpleNamespace(security_context={}))
    api.active_jobs.update({name: {"request": {"model_id": name}} for name in ("first", "second")})

    async def run():
        first = asyncio.create_task(api.process_generation_job("first"))
        second = None
        try:
            for _ in range(100):
                if first_started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert first_started.is_set()
            second = asyncio.create_task(api.process_generation_job("second"))
            await asyncio.sleep(0.05)
            assert starts == ["first"]
        finally:
            release_first.set()
            await first
            if second is not None:
                await second

    asyncio.run(run())
    assert starts == ["first", "second"]


def test_log_stream_continues_after_bounded_buffer_overflow(api, monkeypatch):
    import json
    import logging

    handler = api.LogStreamQueue(maxlen=2)
    monkeypatch.setattr(api, "_log_stream_handler", handler)
    original_sleep = asyncio.sleep

    async def poll_without_delay(_seconds):
        await original_sleep(0)

    monkeypatch.setattr(api.asyncio, "sleep", poll_without_delay)

    def emit(message):
        handler.emit(logging.LogRecord("test", logging.INFO, __file__, 1, message, (), None))

    async def run():
        emit("first")
        emit("second")
        response = await api.stream_logs(since=None)
        iterator = response.body_iterator
        try:
            first = json.loads((await iterator.__anext__()).removeprefix("data: "))
            second = json.loads((await iterator.__anext__()).removeprefix("data: "))
            emit("third")  # deque still has length 2, but contains a new event.
            third = json.loads((await asyncio.wait_for(iterator.__anext__(), timeout=0.5)).removeprefix("data: "))
            assert [first["message"], second["message"], third["message"]] == ["first", "second", "third"]
            assert first["sequence"] < second["sequence"] < third["sequence"]
            assert len(handler.get_logs()) == 2
        finally:
            await iterator.aclose()

    asyncio.run(run())
