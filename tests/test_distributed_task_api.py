"""Public task API deadlines, snapshot isolation, and waiter cancellation."""

import asyncio
from types import SimpleNamespace

import pytest

from vitriol.distributed import coordinator as module
from vitriol.distributed.coordinator import DistributedCoordinator


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [0, -1])
async def test_nonpositive_timeout_returns_without_sleeping(timeout, monkeypatch):
    coord = DistributedCoordinator()
    task_id = await coord.submit_task("model", [0])

    async def unexpected_sleep(delay):
        pytest.fail("An immediate poll must not sleep")

    monkeypatch.setattr(asyncio, "sleep", unexpected_sleep)
    assert await coord.wait_for_task(task_id, timeout) is None
    assert coord.tasks[task_id].status == "pending"
    assert coord.get_stats()["pending_tasks"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("wall_jump", [0, -10000, 10000])
@pytest.mark.parametrize("timeout", [0.025, 0.25])
async def test_wait_deadline_uses_elapsed_time_and_respects_remaining_budget(timeout, wall_jump, monkeypatch):
    coord = DistributedCoordinator()
    task_id = await coord.submit_task("model", [0])
    elapsed = 0.0
    wall = 1000.0
    sleeps = []

    async def advance_clock(delay):
        nonlocal elapsed, wall
        assert len(sleeps) < 10, "Wall-clock changes prevented the wait from expiring"
        sleeps.append(delay)
        elapsed += delay
        wall += delay + wall_jump

    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: wall, monotonic=lambda: elapsed))
    monkeypatch.setattr(asyncio, "sleep", advance_clock)
    assert await coord.wait_for_task(task_id, timeout) is None
    assert elapsed == pytest.approx(timeout)
    assert all(0 < delay <= 0.1 for delay in sleeps)
    assert coord.tasks[task_id].status == "pending"


@pytest.mark.asyncio
async def test_nan_timeout_is_rejected_without_waiting(monkeypatch):
    coord = DistributedCoordinator()
    task_id = await coord.submit_task("model", [])

    async def unexpected_sleep(delay):
        pytest.fail("NaN must not silently become an unlimited wait")

    monkeypatch.setattr(asyncio, "sleep", unexpected_sleep)
    with pytest.raises(ValueError, match="NaN"):
        await coord.wait_for_task(task_id, float("nan"))


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["completed", "failed"])
async def test_immediate_poll_returns_terminal_task(state):
    coord = DistributedCoordinator()
    task_id = await coord.submit_task("model", [])
    coord.tasks[task_id].status = state
    assert await coord.wait_for_task(task_id, 0) is coord.tasks[task_id]
    assert await coord.wait_for_task("unknown", 0) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [None, float("inf")])
async def test_unbounded_wait_allows_completion(timeout, monkeypatch):
    coord = DistributedCoordinator()
    task_id = await coord.submit_task("model", [])

    async def complete(delay):
        coord.tasks[task_id].status = "completed"

    monkeypatch.setattr(asyncio, "sleep", complete)
    assert await coord.wait_for_task(task_id, timeout) is coord.tasks[task_id]


@pytest.mark.asyncio
async def test_submitted_shards_are_copied():
    coord = DistributedCoordinator()
    shards = [2, 5]
    task_id = await coord.submit_task("model", shards)
    shards.clear()
    assert (await coord.get_task_status(task_id))["shard_indices"] == [2, 5]
    assert (await coord.pending_tasks.get()).shard_indices == [2, 5]


@pytest.mark.asyncio
async def test_status_nested_values_are_detached_in_both_directions():
    coord = DistributedCoordinator()
    task_id = await coord.submit_task("model", [2, 5])
    task = coord.tasks[task_id]
    task.result = {"artifacts": [{"path": "original"}]}
    status = await coord.get_task_status(task_id)
    status["shard_indices"].clear()
    status["result"]["artifacts"][0]["path"] = "modified"
    assert task.shard_indices == [2, 5]
    assert task.result["artifacts"][0]["path"] == "original"
    snapshot = await coord.get_task_status(task_id)
    task.shard_indices.append(8)
    task.result["artifacts"].append({"path": "later"})
    assert snapshot["shard_indices"] == [2, 5]
    assert snapshot["result"] == {"artifacts": [{"path": "original"}]}


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_cancel_queued_work():
    coord = DistributedCoordinator()
    task_id = await coord.submit_task("model", [])
    waiter = asyncio.create_task(coord.wait_for_task(task_id))
    try:
        await coord.start()  # No worker yet; the task must remain recoverable.
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert coord.tasks[task_id].status == "pending"
        coord.register_worker("w1", "localhost", 1, {})
        result = await asyncio.wait_for(coord.wait_for_task(task_id), 3)
        assert result.status == "completed"
        assert coord.stats["completed_tasks"] == 1
        assert coord.stats["failed_tasks"] == 0
    finally:
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        await coord.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("abandon", ["timeout", "cancel"])
async def test_abandoning_running_waiter_preserves_execution_and_other_waiters(abandon, monkeypatch):
    coord = DistributedCoordinator()
    coord.register_worker("w1", "localhost", 1, {})
    task_id = await coord.submit_task("model", [0])
    entered = asyncio.Event()
    release = asyncio.Event()
    original_sleep = asyncio.sleep

    async def work(delay):
        if delay == 2:
            entered.set()
            await release.wait()
        else:
            await original_sleep(delay)

    monkeypatch.setattr(asyncio, "sleep", work)
    other = asyncio.create_task(coord.wait_for_task(task_id))
    waiter = None
    try:
        await coord.start()
        await asyncio.wait_for(entered.wait(), 2)
        if abandon == "timeout":
            assert await coord.wait_for_task(task_id, 0.01) is None
        else:
            waiter = asyncio.create_task(coord.wait_for_task(task_id))
            await asyncio.sleep(0)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
        assert coord.tasks[task_id].status == "running"
        assert not other.done()
        release.set()
        assert (await asyncio.wait_for(other, 2)).status == "completed"
        assert coord.stats["completed_tasks"] == 1
        assert coord.stats["failed_tasks"] == 0
    finally:
        release.set()
        other.cancel()
        if waiter is not None:
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
        await asyncio.gather(other, return_exceptions=True)
        await coord.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("max_retries", "expected_attempts", "expected_status", "expected_retry_count"),
    [(0, 1, "failed", 0), (1, 2, "completed", 1), (2, 2, "completed", 1)],
)
async def test_execution_failure_reaches_waiter_with_correct_retry_accounting(
    max_retries, expected_attempts, expected_status, expected_retry_count, monkeypatch
):
    coord = DistributedCoordinator(max_retries=max_retries)
    coord.register_worker("w1", "localhost", 1, {})
    task_id = await coord.submit_task("model", [0])
    attempts = 0
    original_sleep = asyncio.sleep

    async def fail_once(delay):
        nonlocal attempts
        if delay == 2:
            attempts += 1
            if attempts == 1:
                raise RuntimeError("worker execution failed")
        else:
            await original_sleep(delay)

    monkeypatch.setattr(asyncio, "sleep", fail_once)
    try:
        await coord.start()
        result = await asyncio.wait_for(coord.wait_for_task(task_id), 2)
        assert attempts == expected_attempts
        assert result.status == expected_status
        assert result.retry_count == expected_retry_count
        assert result.completed_at is not None
        assert result.error == ("worker execution failed" if max_retries == 0 else None)
        assert coord.stats["completed_tasks"] == (max_retries > 0)
        assert coord.stats["failed_tasks"] == (max_retries == 0)
        assert coord.stats["retried_tasks"] == (max_retries > 0)
        assert coord.workers["w1"].failed_tasks == 1
    finally:
        await coord.stop()


@pytest.mark.asyncio
async def test_execution_failure_exhausts_retries_after_initial_attempt(monkeypatch):
    coord = DistributedCoordinator(max_retries=1)
    coord.register_worker("w1", "localhost", 1, {})
    task_id = await coord.submit_task("model", [0])
    attempts = 0
    original_sleep = asyncio.sleep

    async def always_fail(delay):
        nonlocal attempts
        if delay == 2:
            attempts += 1
            raise RuntimeError(f"worker execution failed {attempts}")
        await original_sleep(delay)

    monkeypatch.setattr(asyncio, "sleep", always_fail)
    try:
        await coord.start()
        result = await asyncio.wait_for(coord.wait_for_task(task_id), 2)
        assert attempts == 2
        assert result.status == "failed"
        assert result.retry_count == 1
        assert result.error == "worker execution failed 2"
        assert coord.stats["retried_tasks"] == 1
        assert coord.stats["failed_tasks"] == 1
        assert coord.workers["w1"].failed_tasks == 2
    finally:
        await coord.stop()
