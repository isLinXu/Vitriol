"""Tests for vitriol.distributed.coordinator module."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from vitriol.distributed.coordinator import (
    WorkerStatus,
    WorkerInfo,
    GenerationTask,
    DistributedCoordinator,
    WorkerNode,
)


class TestWorkerStatus:
    """Tests for WorkerStatus enum."""

    def test_enum_values(self):
        """Test enum has expected values."""
        assert WorkerStatus.IDLE.value == "idle"
        assert WorkerStatus.BUSY.value == "busy"
        assert WorkerStatus.OFFLINE.value == "offline"
        assert WorkerStatus.ERROR.value == "error"


class TestWorkerInfo:
    """Tests for WorkerInfo dataclass."""

    def test_creation(self):
        """Test WorkerInfo creation."""
        worker = WorkerInfo(
            id="w1",
            host="localhost",
            port=8080,
            status=WorkerStatus.IDLE,
            capabilities={"cpu": 4},
            last_heartbeat=12345.0
        )
        assert worker.id == "w1"
        assert worker.host == "localhost"
        assert worker.port == 8080
        assert worker.status == WorkerStatus.IDLE
        assert worker.current_task is None
        assert worker.completed_tasks == 0
        assert worker.failed_tasks == 0


class TestGenerationTask:
    """Tests for GenerationTask dataclass."""

    def test_creation(self):
        """Test GenerationTask creation."""
        task = GenerationTask(
            id="t1",
            model_id="test/model",
            shard_indices=[0, 1],
            strategy="compact",
            dtype="bfloat16"
        )
        assert task.id == "t1"
        assert task.model_id == "test/model"
        assert task.status == "pending"
        assert task.retry_count == 0
        assert task.created_at is not None

    def test_post_init_timestamp(self):
        """Test timestamp is set in post_init."""
        task = GenerationTask(
            id="t1",
            model_id="test/model",
            shard_indices=[0],
            strategy="compact",
            dtype="bfloat16"
        )
        assert task.created_at > 0


class TestDistributedCoordinator:
    """Tests for DistributedCoordinator class."""

    def test_init(self):
        """Test initialization."""
        coord = DistributedCoordinator(host="127.0.0.1", port=5555)
        assert coord.host == "127.0.0.1"
        assert coord.port == 5555
        assert coord.heartbeat_interval == 30
        assert coord.task_timeout == 300
        assert coord.max_retries == 3
        assert coord.workers == {}
        assert coord.tasks == {}
        assert coord.running is False

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test start and stop."""
        coord = DistributedCoordinator()
        await coord.start()
        assert coord.running is True
        await coord.stop()
        assert coord.running is False

    def test_register_worker(self):
        """Test worker registration."""
        coord = DistributedCoordinator()
        result = coord.register_worker("w1", "host1", 8080, {"cpu": 4})
        assert result is True
        assert "w1" in coord.workers
        assert coord.workers["w1"].status == WorkerStatus.IDLE

    def test_register_duplicate_worker(self, caplog):
        """Test registering duplicate worker."""
        coord = DistributedCoordinator()
        coord.register_worker("w1", "host1", 8080, {})
        with caplog.at_level("WARNING"):
            result = coord.register_worker("w1", "host1", 8080, {})
        assert result is False
        assert "already registered" in caplog.text

    def test_unregister_worker(self):
        """Test worker unregistration."""
        coord = DistributedCoordinator()
        coord.register_worker("w1", "host1", 8080, {})
        coord.unregister_worker("w1")
        assert "w1" not in coord.workers

    def test_unregister_worker_with_task(self):
        """Test unregistering worker with running task."""
        coord = DistributedCoordinator()
        coord.register_worker("w1", "host1", 8080, {})
        coord.workers["w1"].current_task = "t1"
        coord.tasks["t1"] = GenerationTask(
            id="t1", model_id="test", shard_indices=[0], strategy="compact", dtype="bfloat16"
        )
        coord.tasks["t1"].status = "running"

        # Need to mock asyncio.create_task for the reassignment
        with patch("vitriol.distributed.coordinator.asyncio.create_task"):
            coord.unregister_worker("w1")

        assert "w1" not in coord.workers
        assert coord.tasks["t1"].status == "pending"

    def test_update_heartbeat(self):
        """Test heartbeat update."""
        coord = DistributedCoordinator()
        coord.register_worker("w1", "host1", 8080, {})
        old_time = coord.workers["w1"].last_heartbeat
        coord.update_heartbeat("w1")
        assert coord.workers["w1"].last_heartbeat > old_time

    def test_update_heartbeat_unknown_worker(self):
        """Test heartbeat for unknown worker doesn't crash."""
        coord = DistributedCoordinator()
        coord.update_heartbeat("unknown")

    @pytest.mark.asyncio
    async def test_submit_task(self):
        """Test task submission."""
        coord = DistributedCoordinator()
        await coord.start()

        task_id = await coord.submit_task("test/model", [0, 1])
        assert task_id is not None
        assert task_id in coord.tasks
        assert coord.tasks[task_id].model_id == "test/model"
        assert coord.stats["total_tasks"] == 1

        await coord.stop()

    @pytest.mark.asyncio
    async def test_get_task_status(self):
        """Test getting task status."""
        coord = DistributedCoordinator()
        await coord.start()

        task_id = await coord.submit_task("test/model", [0])
        status = await coord.get_task_status(task_id)

        assert status is not None
        assert status["id"] == task_id
        assert status["status"] == "pending"

        await coord.stop()

    @pytest.mark.asyncio
    async def test_get_task_status_unknown(self):
        """Test getting status for unknown task."""
        coord = DistributedCoordinator()
        status = await coord.get_task_status("unknown")
        assert status is None

    @pytest.mark.asyncio
    async def test_wait_for_task_completed(self):
        """Test waiting for task completion."""
        coord = DistributedCoordinator()
        await coord.start()

        task_id = await coord.submit_task("test/model", [0])
        # Manually mark as completed
        coord.tasks[task_id].status = "completed"

        result = await coord.wait_for_task(task_id, timeout=1.0)
        assert result is not None
        assert result.status == "completed"

        await coord.stop()

    @pytest.mark.asyncio
    async def test_wait_for_task_timeout(self):
        """Test wait_for_task timeout."""
        coord = DistributedCoordinator()
        await coord.start()

        task_id = await coord.submit_task("test/model", [0])

        result = await coord.wait_for_task(task_id, timeout=0.01)
        assert result is None

        await coord.stop()

    @pytest.mark.asyncio
    async def test_wait_for_task_unknown(self):
        """Test waiting for unknown task."""
        coord = DistributedCoordinator()
        result = await coord.wait_for_task("unknown", timeout=0.1)
        assert result is None

    def test_select_worker(self):
        """Test worker selection."""
        coord = DistributedCoordinator()
        coord.register_worker("w1", "host1", 8080, {})
        coord.register_worker("w2", "host2", 8080, {})

        worker = coord._select_worker()
        assert worker is not None
        assert worker.status == WorkerStatus.IDLE

    def test_select_worker_none_available(self):
        """Test selection when no workers available."""
        coord = DistributedCoordinator()
        coord.register_worker("w1", "host1", 8080, {})
        coord.workers["w1"].status = WorkerStatus.BUSY

        worker = coord._select_worker()
        assert worker is None

    def test_select_worker_load_balancing(self):
        """Test load balancing selects worker with fewer completed tasks."""
        coord = DistributedCoordinator()
        coord.register_worker("w1", "host1", 8080, {})
        coord.register_worker("w2", "host2", 8080, {})
        coord.workers["w1"].completed_tasks = 10
        coord.workers["w2"].completed_tasks = 2

        worker = coord._select_worker()
        assert worker.id == "w2"

    @pytest.mark.asyncio
    async def test_heartbeat_monitor(self):
        """Test heartbeat monitor removes dead workers."""
        coord = DistributedCoordinator(heartbeat_interval=0.001)
        coord.register_worker("w1", "host1", 8080, {})
        # Set heartbeat to very old
        coord.workers["w1"].last_heartbeat = 0

        coord.running = True
        # Run one iteration then stop
        original_sleep = asyncio.sleep
        async def mock_sleep(delay):
            coord.running = False
        with patch("vitriol.distributed.coordinator.asyncio.sleep", side_effect=mock_sleep):
            await coord._heartbeat_monitor()

        assert "w1" not in coord.workers

    @pytest.mark.asyncio
    async def test_task_scheduler_no_workers(self):
        """Test scheduler with no workers."""
        coord = DistributedCoordinator()
        task = GenerationTask(
            id="t1", model_id="test", shard_indices=[0], strategy="compact", dtype="bfloat16"
        )
        await coord.pending_tasks.put(task)
        coord.running = True

        call_count = 0
        original_sleep = asyncio.sleep
        async def controlled_sleep(delay):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                coord.running = False
        with patch("vitriol.distributed.coordinator.asyncio.sleep", side_effect=controlled_sleep):
            await coord._task_scheduler()

        coord.running = False

    @pytest.mark.asyncio
    async def test_execute_task(self):
        """Test task execution."""
        coord = DistributedCoordinator()
        coord.register_worker("w1", "host1", 8080, {})
        worker = coord.workers["w1"]
        worker.status = WorkerStatus.BUSY

        task = GenerationTask(
            id="t1", model_id="test", shard_indices=[0], strategy="compact", dtype="bfloat16"
        )

        await coord._execute_task(task, worker)

        assert task.status == "completed"
        assert task.result is not None
        assert worker.status == WorkerStatus.IDLE
        assert worker.completed_tasks == 1
        assert coord.stats["completed_tasks"] == 1

    @pytest.mark.asyncio
    async def test_handle_task_failure_retry(self):
        """Test task failure with retry."""
        coord = DistributedCoordinator(max_retries=3)
        coord.register_worker("w1", "host1", 8080, {})
        worker = coord.workers["w1"]

        task = GenerationTask(
            id="t1", model_id="test", shard_indices=[0], strategy="compact", dtype="bfloat16"
        )
        task.status = "running"

        await coord._handle_task_failure(task, worker, "Error")

        assert task.status == "pending"
        assert task.retry_count == 1
        assert coord.stats["retried_tasks"] == 1

    @pytest.mark.asyncio
    async def test_handle_task_failure_max_retries(self):
        """Test task failure after max retries."""
        coord = DistributedCoordinator(max_retries=2)
        coord.register_worker("w1", "host1", 8080, {})
        worker = coord.workers["w1"]

        task = GenerationTask(
            id="t1", model_id="test", shard_indices=[0], strategy="compact", dtype="bfloat16"
        )
        task.retry_count = 2

        await coord._handle_task_failure(task, worker, "Error")

        assert task.status == "failed"
        assert task.error == "Error"
        assert coord.stats["failed_tasks"] == 1

    @pytest.mark.asyncio
    async def test_task_timeout_monitor(self):
        """Test task timeout monitor."""
        coord = DistributedCoordinator(task_timeout=0.01)
        coord.register_worker("w1", "host1", 8080, {})
        worker = coord.workers["w1"]

        task = GenerationTask(
            id="t1", model_id="test", shard_indices=[0], strategy="compact", dtype="bfloat16"
        )
        task.status = "running"
        task.started_at = 0  # Very old start time
        task.assigned_worker = "w1"
        coord.tasks["t1"] = task

        coord.running = True
        call_count = 0
        async def controlled_sleep(delay):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                coord.running = False
        with patch("vitriol.distributed.coordinator.asyncio.sleep", side_effect=controlled_sleep):
            await coord._task_timeout_monitor()

        coord.running = False

    def test_get_stats_empty(self):
        """Test stats with no activity."""
        coord = DistributedCoordinator()
        stats = coord.get_stats()

        assert stats["total_tasks"] == 0
        assert stats["completed_tasks"] == 0
        assert stats["failed_tasks"] == 0
        assert stats["active_workers"] == 0
        assert stats["pending_tasks"] == 0
        assert stats["running_tasks"] == 0

    def test_get_stats_with_activity(self):
        """Test stats with some activity."""
        coord = DistributedCoordinator()
        coord.register_worker("w1", "host1", 8080, {})
        coord.stats["total_tasks"] = 5
        coord.stats["completed_tasks"] = 3

        stats = coord.get_stats()
        assert stats["active_workers"] == 1
        assert stats["workers"]["w1"]["status"] == "idle"


class TestWorkerNode:
    """Tests for WorkerNode class."""

    def test_init_with_id(self):
        """Test initialization with explicit ID."""
        worker = WorkerNode("localhost", 5555, worker_id="custom-id")
        assert worker.worker_id == "custom-id"
        assert worker.coordinator_host == "localhost"
        assert worker.coordinator_port == 5555

    def test_init_auto_id(self):
        """Test initialization with auto-generated ID."""
        worker = WorkerNode("localhost", 5555)
        assert worker.worker_id is not None
        assert len(worker.worker_id) > 0

    def test_default_capabilities(self):
        """Test default capabilities."""
        worker = WorkerNode("localhost", 5555)
        assert worker.capabilities["cpu_count"] == 4
        assert worker.capabilities["memory_gb"] == 16
        assert worker.capabilities["gpu_count"] == 0
        assert "compact" in worker.capabilities["strategies"]

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test start and stop."""
        worker = WorkerNode("localhost", 5555)
        await worker.start()
        assert worker.running is True
        await worker.stop()
        assert worker.running is False

    @pytest.mark.asyncio
    async def test_heartbeat_loop(self):
        """Test heartbeat loop."""
        worker = WorkerNode("localhost", 5555)
        worker.running = True
        call_count = 0
        async def controlled_sleep(delay):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                worker.running = False
        with patch("vitriol.distributed.coordinator.asyncio.sleep", side_effect=controlled_sleep):
            await worker._heartbeat_loop()

        worker.running = False

    @pytest.mark.asyncio
    async def test_task_processor(self):
        """Test task processor loop."""
        worker = WorkerNode("localhost", 5555)
        worker.running = True
        call_count = 0
        async def controlled_sleep(delay):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                worker.running = False
        with patch("vitriol.distributed.coordinator.asyncio.sleep", side_effect=controlled_sleep):
            await worker._task_processor()

        worker.running = False
