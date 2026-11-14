"""FIFO dispatch and wakeups when pending work or worker capacity changes."""

import asyncio

import pytest

import vitriol.distributed.coordinator as coordinator_module
from vitriol.distributed.coordinator import DistributedCoordinator, GenerationTask


@pytest.mark.asyncio
async def test_no_workers_preserves_pending_order_across_stop():
    coord = DistributedCoordinator()
    ids = [await coord.submit_task("model", []) for _ in range(3)]
    try:
        await coord.start()
        await asyncio.sleep(0)
        await coord.stop()
        assert coord.get_stats()["pending_tasks"] == 3
        assert [coord.pending_tasks.get_nowait().id for _ in ids] == ids
    finally:
        await coord.stop()


@pytest.mark.asyncio
async def test_busy_worker_preserves_fifo_dispatch(monkeypatch):
    coord = DistributedCoordinator()
    coord.register_worker("w1", "localhost", 1, {})
    ids = [await coord.submit_task("model", []) for _ in range(3)]
    dispatched = asyncio.Queue()
    completed = asyncio.Event()
    original_sleep = asyncio.sleep
    original_execute = coord._execute_task

    async def polling_sleep(delay):
        if delay == 1:
            await completed.wait()
            completed.clear()
        else:
            await original_sleep(delay)

    async def record(task, worker):
        dispatched.put_nowait(task.id)
        await original_execute(task, worker)
        completed.set()

    monkeypatch.setattr(asyncio, "sleep", polling_sleep)
    monkeypatch.setattr(coord, "_execute_task", record)
    try:
        await coord.start()
        actual = [await asyncio.wait_for(dispatched.get(), 2) for _ in ids]
        assert actual == ids
    finally:
        await coord.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("capacity_change", ["register", "complete", "fail"])
async def test_capacity_change_dispatches_without_polling_delay(capacity_change, monkeypatch):
    coord = DistributedCoordinator(max_retries=1)
    entered = asyncio.Queue()
    release = asyncio.Event()
    polling = asyncio.Event()
    original_sleep = asyncio.sleep
    original_execute = coord._execute_task

    async def controlled_sleep(delay):
        if delay == 1:
            # A legacy poll cannot finish during this test; state changes must wake dispatch.
            await polling.wait()
        elif delay == 2:
            await release.wait()
            if capacity_change == "fail":
                raise RuntimeError("worker failed")
        else:
            await original_sleep(delay)

    async def record(task, worker):
        entered.put_nowait(task.id)
        await original_execute(task, worker)

    monkeypatch.setattr(asyncio, "sleep", controlled_sleep)
    monkeypatch.setattr(coord, "_execute_task", record)
    if capacity_change != "register":
        coord.register_worker("w1", "localhost", 1, {})
        first = await coord.submit_task("model", [0])
    target = await coord.submit_task("model", [])
    try:
        await coord.start()
        if capacity_change != "register":
            assert await asyncio.wait_for(entered.get(), 2) == first
        await asyncio.sleep(0)
        if capacity_change == "register":
            coord.register_worker("w1", "localhost", 1, {})
        else:
            release.set()
        assert await asyncio.wait_for(entered.get(), 2) == target
    finally:
        release.set()
        await coord.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("insertion", ["submit", "queue_put", "queue_put_nowait"])
async def test_idle_scheduler_wakes_on_each_supported_queue_insertion(insertion):
    coord = DistributedCoordinator()
    coord.register_worker("w1", "localhost", 1, {})
    try:
        await coord.start()
        await asyncio.sleep(0)
        if insertion == "submit":
            task_id = await coord.submit_task("model", [])
        else:
            task = GenerationTask("direct", "model", [], "compact", "float32")
            task_id = task.id
            coord.tasks[task_id] = task
            if insertion == "queue_put":
                await coord.pending_tasks.put(task)
            else:
                coord.pending_tasks.put_nowait(task)
        result = await asyncio.wait_for(coord.wait_for_task(task_id), 2)
        assert result.status == "completed"
        assert coord.stats["completed_tasks"] == 1
    finally:
        await coord.stop()


@pytest.mark.asyncio
async def test_idle_scheduler_does_not_poll_when_workers_are_absent(monkeypatch):
    coord = DistributedCoordinator()
    await coord.submit_task("model", [])
    sleeps = []
    original_sleep = asyncio.sleep

    async def record_sleep(delay):
        sleeps.append(delay)
        await original_sleep(delay)

    monkeypatch.setattr(asyncio, "sleep", record_sleep)
    try:
        await coord.start()
        await original_sleep(0)
        assert 1 not in sleeps
    finally:
        await coord.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("first", ["task", "worker"])
async def test_work_and_capacity_notifications_are_not_lost(first):
    coord = DistributedCoordinator()
    try:
        await coord.start()
        await asyncio.sleep(0)
        if first == "task":
            task_id = await coord.submit_task("model", [])
        else:
            coord.register_worker("w1", "localhost", 1, {})
        await asyncio.sleep(0)  # Scheduler observes only one of its prerequisites.
        if first == "task":
            coord.register_worker("w1", "localhost", 1, {})
        else:
            task_id = await coord.submit_task("model", [])
        result = await asyncio.wait_for(coord.wait_for_task(task_id), 2)
        assert result.status == "completed"
    finally:
        await coord.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["retry", "disconnect"])
async def test_reassigned_work_goes_behind_existing_pending_tasks(reason, monkeypatch):
    coord = DistributedCoordinator()
    coord.register_worker("w1", "localhost", 1, {})
    first = await coord.submit_task("model", [0])
    second = await coord.submit_task("model", [])
    dispatched = asyncio.Queue()
    original_execute = coord._execute_task

    async def record(task, worker):
        dispatched.put_nowait(task.id)
        await original_execute(task, worker)

    monkeypatch.setattr(coord, "_execute_task", record)
    try:
        await coord.start()
        assert await asyncio.wait_for(dispatched.get(), 2) == first
        coord.tasks[first].shard_indices = []
        if reason == "retry":
            await coord._handle_task_failure(coord.tasks[first], coord.workers["w1"], "injected failure")
        else:
            coord.unregister_worker("w1")
            coord.register_worker("w1", "localhost", 2, {})
        assert [await asyncio.wait_for(dispatched.get(), 2) for _ in range(2)] == [second, first]
        assert (await asyncio.wait_for(coord.wait_for_task(first), 2)).status == "completed"
        assert coord.stats["completed_tasks"] == 2
    finally:
        await coord.stop()


@pytest.mark.asyncio
async def test_dispatch_setup_failure_restores_fifo_task_and_worker_state(monkeypatch, caplog):
    coord = DistributedCoordinator()
    coord.register_worker("w1", "localhost", 1, {})
    task_ids = [await coord.submit_task("model", []) for _ in range(2)]
    first_task = coord.tasks[task_ids[0]]
    launch_failed = asyncio.Event()
    execution_order = []
    original_create_task = asyncio.create_task
    original_execute = coord._execute_task
    injected = False

    async def record_execution(task, worker):
        execution_order.append(task.id)
        await original_execute(task, worker)

    def create_task(coro, *args, **kwargs):
        nonlocal injected
        if not injected and first_task.status == "running":
            injected = True
            coro.close()
            launch_failed.set()
            raise RuntimeError("injected dispatch setup failure")
        return original_create_task(coro, *args, **kwargs)

    monkeypatch.setattr(coordinator_module, "_BACKGROUND_RETRY_INITIAL", 0.05)
    monkeypatch.setattr(coord, "_execute_task", record_execution)
    monkeypatch.setattr(asyncio, "create_task", create_task)
    try:
        with caplog.at_level("ERROR"):
            await coord.start()
            await asyncio.wait_for(launch_failed.wait(), 1)
        assert first_task.status == "pending"
        assert first_task.assigned_worker is None
        assert coord.workers["w1"].status.value == "idle"
        assert coord.workers["w1"].current_task is None
        assert coord.get_stats()["pending_tasks"] == 2
        assert "injected dispatch setup failure" in caplog.text

        await asyncio.wait_for(
            coord.wait_for_task(task_ids[1]), 2
        )
        assert execution_order == task_ids
        assert first_task.retry_count == 0
        assert coord.stats["completed_tasks"] == 2
    finally:
        await coord.stop()


@pytest.mark.asyncio
async def test_scheduler_failures_use_supervised_exponential_backoff(monkeypatch, caplog):
    coord = DistributedCoordinator()
    monkeypatch.setattr(coordinator_module, "_BACKGROUND_RETRY_INITIAL", 0.1)
    monkeypatch.setattr(coordinator_module, "_BACKGROUND_RETRY_MAX", 0.4)
    calls = []
    third_attempt = asyncio.Event()

    def fail_worker_selection():
        calls.append(asyncio.get_running_loop().time())
        if len(calls) == 3:
            third_attempt.set()
        raise RuntimeError("injected persistent scheduler failure")

    monkeypatch.setattr(coord, "_select_worker", fail_worker_selection)
    try:
        with caplog.at_level("ERROR"):
            await coord.start()
            await asyncio.wait_for(third_attempt.wait(), 1.5)
        assert len(calls) == 3
        assert "restarting in 0.1s" in caplog.text
        assert "restarting in 0.2s" in caplog.text
        await coord.stop()
        assert len(calls) == 3  # Stop cancels the supervisor's next backoff wait.
    finally:
        await coord.stop()
