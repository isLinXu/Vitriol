"""FIFO dispatch and wakeups when pending work or worker capacity changes."""

import asyncio

import pytest

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
