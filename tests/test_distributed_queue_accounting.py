"""Queue.join accounting across dispatch, retries, and coordinator restart."""

import asyncio

import pytest

from vitriol.distributed.coordinator import DistributedCoordinator


@pytest.mark.asyncio
async def test_queue_join_completes_when_task_is_dispatched():
    coord = DistributedCoordinator()
    coord.register_worker("w1", "localhost", 1, {})
    await coord.submit_task("model", [])
    try:
        await coord.start()
        await asyncio.wait_for(coord.pending_tasks.join(), 1)
        assert coord.pending_tasks.empty()
    finally:
        await coord.stop()


@pytest.mark.asyncio
async def test_queue_join_balances_a_failed_attempt_and_retry(monkeypatch):
    coord = DistributedCoordinator(max_retries=2)
    coord.register_worker("w1", "localhost", 1, {})
    task_id = await coord.submit_task("model", [])
    original_execute = coord._execute_task
    calls = 0

    async def fail_once(task, worker):
        nonlocal calls
        calls += 1
        if calls == 1:
            await coord._handle_task_failure(task, worker, "injected execution failure")
        else:
            await original_execute(task, worker)

    monkeypatch.setattr(coord, "_execute_task", fail_once)
    try:
        await coord.start()
        await asyncio.wait_for(coord.pending_tasks.join(), 1)
        result = await asyncio.wait_for(coord.wait_for_task(task_id), 1)
        assert result.status == "completed"
        assert calls == 2
        assert result.retry_count == 1
        assert coord.stats["retried_tasks"] == 1
    finally:
        await coord.stop()


@pytest.mark.asyncio
async def test_queue_join_recounts_a_task_requeued_by_stop():
    coord = DistributedCoordinator()
    coord.register_worker("w1", "localhost", 1, {})
    task_id = await coord.submit_task("model", [])
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    release = asyncio.Event()
    calls = 0
    original_execute = coord._execute_task

    async def hold_execution(task, worker):
        nonlocal calls
        calls += 1
        (first_started if calls == 1 else second_started).set()
        await release.wait()
        await original_execute(task, worker)

    coord._execute_task = hold_execution
    try:
        await coord.start()
        await asyncio.wait_for(first_started.wait(), 1)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(coord.pending_tasks.join(), 0.05)
        await coord.stop()
        assert coord.tasks[task_id].status == "pending"
        assert coord.pending_tasks.qsize() == 1

        release.set()
        await coord.start()
        await asyncio.wait_for(second_started.wait(), 1)
        await asyncio.wait_for(coord.pending_tasks.join(), 1)
        assert coord.pending_tasks.empty()
        assert coord.tasks[task_id].status == "completed"
    finally:
        release.set()
        await coord.stop()


@pytest.mark.asyncio
async def test_duplicate_stale_queue_entry_is_marked_done():
    coord = DistributedCoordinator()
    coord.register_worker("w1", "localhost", 1, {})
    task_id = await coord.submit_task("model", [])
    coord.pending_tasks.put_nowait(coord.tasks[task_id])
    try:
        await coord.start()
        result = await asyncio.wait_for(coord.wait_for_task(task_id), 1)
        assert result.status == "completed"
        assert coord.pending_tasks.empty()
        await asyncio.wait_for(coord.pending_tasks.join(), 1)
    finally:
        await coord.stop()
