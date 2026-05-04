"""Tests for patches and distributed coordinator modules."""


from vitriol.patches.dynamic_model_patches import (
    _set_missing as _dyn_set_missing
)
from vitriol.distributed.coordinator import (
    WorkerStatus, WorkerInfo, GenerationTask, DistributedCoordinator
)


# ─────────────────────────────────────────────────────────────────────────────
# dynamic_model_patches tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDynamicSetMissing:
    def test_sets_missing(self):
        class Obj:
            existing = "original"
        obj = Obj()
        _dyn_set_missing(obj, existing="new", missing="value")
        assert obj.missing == "value"
        assert obj.existing == "original"  # Should not overwrite


# ─────────────────────────────────────────────────────────────────────────────
# distributed coordinator tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWorkerStatus:
    def test_values(self):
        assert WorkerStatus.IDLE.value == "idle"
        assert WorkerStatus.BUSY.value == "busy"
        assert WorkerStatus.OFFLINE.value == "offline"
        assert WorkerStatus.ERROR.value == "error"


class TestWorkerInfo:
    def test_creation(self):
        worker = WorkerInfo(
            id="worker-1",
            host="localhost",
            port=5555,
            status=WorkerStatus.IDLE,
            capabilities={"gpu": True},
            last_heartbeat=0.0,
        )
        assert worker.id == "worker-1"
        assert worker.status == WorkerStatus.IDLE
        assert worker.completed_tasks == 0

    def test_defaults(self):
        worker = WorkerInfo(
            id="w1", host="h", port=1,
            status=WorkerStatus.IDLE,
            capabilities={}, last_heartbeat=0.0,
        )
        assert worker.current_task is None
        assert worker.completed_tasks == 0
        assert worker.failed_tasks == 0


class TestGenerationTask:
    def test_creation(self):
        task = GenerationTask(
            id="task-1",
            model_id="test/model",
            shard_indices=[0, 1],
            strategy="random",
            dtype="bfloat16",
        )
        assert task.id == "task-1"
        assert task.model_id == "test/model"
        assert task.status == "pending"
        assert task.created_at is not None

    def test_post_init_sets_created_at(self):
        task = GenerationTask(
            id="t1", model_id="m", shard_indices=[0],
            strategy="random", dtype="bfloat16",
        )
        assert task.created_at is not None
        assert task.retry_count == 0


class TestDistributedCoordinator:
    def test_init(self):
        coord = DistributedCoordinator(host="127.0.0.1", port=5556)
        assert coord.host == "127.0.0.1"
        assert coord.port == 5556
        assert coord.workers == {}

    def test_register_worker(self):
        coord = DistributedCoordinator()
        result = coord.register_worker("w1", "localhost", 5556, {"gpu": True})
        assert result is True
        assert "w1" in coord.workers
        assert coord.workers["w1"].status == WorkerStatus.IDLE

    def test_register_duplicate(self):
        coord = DistributedCoordinator()
        coord.register_worker("w1", "localhost", 5556, {})
        result = coord.register_worker("w1", "localhost", 5556, {})
        assert result is False

    def test_heartbeat_updates(self):
        coord = DistributedCoordinator()
        coord.register_worker("w1", "localhost", 5556, {})
        old_time = coord.workers["w1"].last_heartbeat
        coord.update_heartbeat("w1")
        assert coord.workers["w1"].last_heartbeat > old_time

    def test_heartbeat_unknown_worker(self):
        coord = DistributedCoordinator()
        coord.update_heartbeat("unknown")  # Should not raise

    def test_unregister_worker(self):
        coord = DistributedCoordinator()
        coord.register_worker("w1", "localhost", 5556, {})
        coord.unregister_worker("w1")
        assert "w1" not in coord.workers

