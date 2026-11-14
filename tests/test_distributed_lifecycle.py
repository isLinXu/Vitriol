"""Regression tests for coordinator shutdown and retired execution attempts."""

import asyncio

import pytest

from vitriol.distributed.coordinator import DistributedCoordinator, WorkerNode, WorkerStatus


async def eventually(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["coordinator", "worker"])
async def test_start_is_idempotent_and_stop_joins_background_tasks(kind, monkeypatch):
    node = DistributedCoordinator() if kind == "coordinator" else WorkerNode("localhost", 5555)
    names = ("_heartbeat_monitor", "_task_scheduler", "_task_timeout_monitor") if kind == "coordinator" else (
        "_heartbeat_loop", "_task_processor"
    )
    entered = []
    exited = []

    async def background():
        entered.append(asyncio.current_task())
        try:
            await asyncio.Event().wait()
        finally:
            exited.append(asyncio.current_task())

    for name in names:
        monkeypatch.setattr(node, name, background)
    try:
        await node.start()
        await node.start()
        await asyncio.sleep(0)
        assert len(entered) == len(names)
        await asyncio.gather(node.stop(), node.stop())
        assert set(exited) == set(entered)
        assert all(task.done() for task in entered)
        await node.start()
        await asyncio.sleep(0)
        assert len(entered) == 2 * len(names)
        await node.stop()
        assert set(exited) == set(entered)
    finally:
        await node.stop()
        for task in entered:
            task.cancel()
        await asyncio.gather(*entered, return_exceptions=True)


@pytest.mark.asyncio
async def test_stop_requeues_running_task_and_restart_completes_once():
    coord = DistributedCoordinator()
    coord.register_worker("w1", "localhost", 1, {})
    task_id = await coord.submit_task("model", [0])
    task = coord.tasks[task_id]
    try:
        await coord.start()
        await eventually(lambda: task.status == "running")
        await coord.stop()
        assert task.status == "pending"
        assert task.assigned_worker is None
        assert task.started_at is None
        assert coord.workers["w1"].status == WorkerStatus.IDLE
        assert coord.get_stats()["pending_tasks"] == 1
        assert coord.stats["completed_tasks"] == 0
        assert task.retry_count == 0
        task.shard_indices = []  # Finish the resumed simulation without a delay.
        await coord.start()
        await eventually(lambda: task.status == "completed")
        await coord.stop()
        assert coord.stats["completed_tasks"] == 1
        assert coord.workers["w1"].completed_tasks == 1
        assert coord.get_stats()["pending_tasks"] == 0
    finally:
        await coord.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["disconnect", "timeout"])
@pytest.mark.parametrize("late_error", [False, True])
async def test_retired_attempt_cannot_complete_reassigned_task(reason, late_error, monkeypatch):
    coord = DistributedCoordinator()
    coord.register_worker("w1", "localhost", 1, {})
    task_id = await coord.submit_task("model", [0])
    task = coord.tasks[task_id]
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()
    original_sleep = asyncio.sleep

    async def delayed_work(delay):
        if delay != 2:
            return await original_sleep(delay)
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            # Model an operation whose result arrives after cancellation.
            cancelled.set()
            await release.wait()
            if late_error:
                raise RuntimeError("late failure from retired attempt") from None

    monkeypatch.setattr(asyncio, "sleep", delayed_work)
    try:
        await coord.start()
        await asyncio.wait_for(entered.wait(), 2)
        old_worker = coord.workers["w1"]
        if reason == "disconnect":
            coord.unregister_worker("w1")
            coord.register_worker("w1", "localhost", 2, {})
        else:
            await coord._handle_task_failure(task, old_worker, "Task timed out")
        await asyncio.wait_for(cancelled.wait(), 2)
        assert task.status in ("pending", "running")
        assert coord.stats["completed_tasks"] == 0
        task.shard_indices = []
        release.set()
        await eventually(lambda: task.status == "completed")
        await coord.stop()
        assert coord.stats["completed_tasks"] == 1
        assert coord.workers["w1"].completed_tasks == 1
        assert task.error is None
        assert coord.stats["retried_tasks"] == (reason == "timeout")
        if reason == "disconnect":
            assert old_worker.completed_tasks == 0
    finally:
        release.set()
        await coord.stop()


@pytest.mark.asyncio
async def test_duplicate_queue_entry_is_not_executed_twice():
    coord = DistributedCoordinator()
    for worker in ("w1", "w2"):
        coord.register_worker(worker, "localhost", 1, {})
    task_id = await coord.submit_task("model", [])
    task = coord.tasks[task_id]
    coord.pending_tasks.put_nowait(task)
    try:
        await coord.start()
        await eventually(lambda: task.status == "completed" and coord.pending_tasks.empty())
        await coord.stop()
        assert coord.stats["completed_tasks"] == 1
        assert sum(w.completed_tasks for w in coord.workers.values()) == 1
    finally:
        await coord.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["coordinator", "worker"])
async def test_restart_waits_for_cleanup_even_when_stop_caller_is_cancelled(kind, monkeypatch):
    node = DistributedCoordinator() if kind == "coordinator" else WorkerNode("localhost", 5555)
    name = "_heartbeat_monitor" if kind == "coordinator" else "_heartbeat_loop"
    entered = asyncio.Event()
    cancelling = asyncio.Event()
    release = asyncio.Event()
    exits = []

    async def background():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelling.set()
            await release.wait()
            exits.append(True)

    monkeypatch.setattr(node, name, background)
    await node.start()
    await asyncio.wait_for(entered.wait(), 2)
    stopping = asyncio.create_task(node.stop())
    restarting = None
    try:
        await asyncio.wait_for(cancelling.wait(), 2)
        stopping.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stopping
        restarting = asyncio.create_task(node.start())
        await asyncio.sleep(0)
        assert not restarting.done()
        assert not node.running
        release.set()
        await asyncio.wait_for(restarting, 2)
        assert exits == [True]
        assert node.running
    finally:
        release.set()
        if restarting is not None:
            await restarting
        await node.stop()


@pytest.mark.asyncio
async def test_stop_while_scheduler_waits_does_not_lose_queued_task():
    coord = DistributedCoordinator()
    coord.register_worker("w1", "localhost", 1, {})
    try:
        await coord.start()
        await asyncio.sleep(0)  # Scheduler is waiting on the empty queue.
        task_id = await coord.submit_task("model", [])
        await coord.stop()  # Stop before the awakened scheduler resumes.
        assert coord.tasks[task_id].status == "pending"
        assert coord.get_stats()["pending_tasks"] == 1
        await coord.start()
        await eventually(lambda: coord.tasks[task_id].status == "completed")
        assert coord.stats["completed_tasks"] == 1
    finally:
        await coord.stop()


@pytest.mark.asyncio
async def test_timeout_monitor_retires_attempt_and_exhausts_retry_budget(monkeypatch):
    coord = DistributedCoordinator(task_timeout=1, max_retries=0)
    coord.register_worker("w1", "localhost", 1, {})
    task_id = await coord.submit_task("model", [0])
    task = coord.tasks[task_id]
    execution_started = asyncio.Event()
    execution_cancelled = asyncio.Event()
    check_timeout = asyncio.Event()
    original_sleep = asyncio.sleep

    async def controlled_sleep(delay):
        if delay == coord.task_timeout:
            await check_timeout.wait()
            check_timeout.clear()
        elif delay == 2:
            execution_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                execution_cancelled.set()
        else:
            await original_sleep(delay)

    monkeypatch.setattr(asyncio, "sleep", controlled_sleep)
    try:
        await coord.start()
        await asyncio.wait_for(execution_started.wait(), 2)
        coord._task_start_times[task_id] = 0
        check_timeout.set()
        await asyncio.wait_for(execution_cancelled.wait(), 2)
        assert task.status == "failed"
        assert task.completed_at is not None
        assert task.error == "Task timed out"
        assert task.retry_count == 0
        assert coord.stats["failed_tasks"] == 1
        assert coord.stats["completed_tasks"] == 0
        assert coord.workers["w1"].status == WorkerStatus.IDLE
        await coord._handle_task_failure(task, coord.workers["w1"], "duplicate failure")
        assert coord.stats["failed_tasks"] == 1
    finally:
        await coord.stop()
