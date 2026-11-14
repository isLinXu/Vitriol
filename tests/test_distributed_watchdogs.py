"""Watchdogs measure elapsed time independently of public wall-clock timestamps."""

import asyncio
from types import SimpleNamespace

import pytest

from vitriol.distributed import coordinator as module
from vitriol.distributed.coordinator import DistributedCoordinator


@pytest.fixture
def clock(monkeypatch):
    state = SimpleNamespace(wall=1000.0, elapsed=0.0)
    monkeypatch.setattr(module, "time", SimpleNamespace(
        time=lambda: state.wall, monotonic=lambda: state.elapsed,
    ))
    return state


async def heartbeat_tick(coord, clock, monkeypatch, elapsed, wall):
    async def tick(delay):
        clock.elapsed = elapsed
        clock.wall = wall
        coord.running = False

    with monkeypatch.context() as patch:
        patch.setattr(asyncio, "sleep", tick)
        coord.running = True
        await coord._heartbeat_monitor()


@pytest.mark.asyncio
@pytest.mark.parametrize("elapsed,wall,expired", [(1, 11001, False), (21, -8979, True), (21, 1021, True)])
async def test_heartbeat_expiry_ignores_wall_clock_jumps(elapsed, wall, expired, clock, monkeypatch):
    coord = DistributedCoordinator(heartbeat_interval=10)
    coord.register_worker("w1", "localhost", 1, {})
    assert coord.workers["w1"].last_heartbeat == 1000
    await heartbeat_tick(coord, clock, monkeypatch, elapsed, wall)
    assert ("w1" not in coord.workers) is expired


@pytest.mark.asyncio
@pytest.mark.parametrize("elapsed,expired", [(34, False), (36, True)])
async def test_heartbeat_refresh_resets_elapsed_budget(elapsed, expired, clock, monkeypatch):
    coord = DistributedCoordinator(heartbeat_interval=10)
    coord.register_worker("w1", "localhost", 1, {})
    clock.elapsed = 15
    clock.wall = 9000
    coord.update_heartbeat("w1")
    assert coord.workers["w1"].last_heartbeat == 9000
    await heartbeat_tick(coord, clock, monkeypatch, elapsed, -1000)
    assert ("w1" not in coord.workers) is expired


@pytest.mark.asyncio
async def test_reregistered_worker_gets_fresh_heartbeat_budget(clock, monkeypatch):
    coord = DistributedCoordinator(heartbeat_interval=10)
    coord.register_worker("w1", "localhost", 1, {})
    coord.unregister_worker("w1")
    clock.elapsed = 100
    coord.register_worker("w1", "localhost", 2, {})
    await heartbeat_tick(coord, clock, monkeypatch, 101, 9000)
    assert coord.workers["w1"].port == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("elapsed,wall,expired", [(1, 11001, False), (11, -8989, True), (10, 1010, True)])
async def test_execution_deadline_ignores_wall_clock_jumps(elapsed, wall, expired, clock, monkeypatch):
    coord = DistributedCoordinator(task_timeout=10, max_retries=0)
    coord.register_worker("w1", "localhost", 1, {})
    task_id = await coord.submit_task("model", [0])
    entered = asyncio.Event()
    release = asyncio.Event()

    async def controlled_sleep(delay):
        if delay == 2:
            entered.set()
            await release.wait()
        else:
            clock.elapsed = elapsed
            clock.wall = wall
            coord.running = False

    monkeypatch.setattr(asyncio, "sleep", controlled_sleep)
    coord.running = True
    scheduler = asyncio.create_task(coord._task_scheduler())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert (await coord.get_task_status(task_id))["started_at"] == 1000
        await coord._task_timeout_monitor()
        assert coord.tasks[task_id].status == ("failed" if expired else "running")
        assert coord.stats["failed_tasks"] == expired
        if expired:
            assert coord.tasks[task_id].completed_at == wall
    finally:
        scheduler.cancel()
        await asyncio.gather(scheduler, return_exceptions=True)
        release.set()
        await coord.stop()


@pytest.mark.asyncio
async def test_short_task_timeout_has_matching_watchdog_cadence(monkeypatch):
    coord = DistributedCoordinator(task_timeout=0.025)
    delays = []

    async def tick(delay):
        delays.append(delay)
        coord.running = False

    monkeypatch.setattr(asyncio, "sleep", tick)
    coord.running = True
    await coord._task_timeout_monitor()
    assert len(delays) == 1
    assert 0 < delays[0] <= coord.task_timeout


@pytest.mark.parametrize("setting", ["heartbeat_interval", "task_timeout"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_watchdog_intervals_fail_at_construction(setting, value):
    with pytest.raises(ValueError, match=setting):
        DistributedCoordinator(**{setting: value})


@pytest.mark.parametrize("max_retries", [-1, 1.5, True])
def test_invalid_retry_budgets_fail_at_construction(max_retries):
    with pytest.raises(ValueError, match="max_retries"):
        DistributedCoordinator(max_retries=max_retries)


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["retry", "disconnect", "restart"])
async def test_new_attempt_receives_fresh_deadline(reason, clock, monkeypatch):
    coord = DistributedCoordinator(task_timeout=10)
    coord.register_worker("w1", "localhost", 1, {})
    task_id = await coord.submit_task("model", [0])
    entered = asyncio.Queue()
    blocked = asyncio.Event()

    async def suspended_work(delay):
        if delay == 2:
            entered.put_nowait(True)
        await blocked.wait()

    monkeypatch.setattr(asyncio, "sleep", suspended_work)
    try:
        await coord.start()
        await asyncio.wait_for(entered.get(), 2)
        clock.elapsed = 100
        clock.wall = 1100
        if reason == "retry":
            await coord._handle_task_failure(coord.tasks[task_id], coord.workers["w1"], "injected")
        elif reason == "disconnect":
            coord.unregister_worker("w1")
            assert "w1" not in coord._heartbeat_times
            coord.register_worker("w1", "localhost", 2, {})
        else:
            await coord.stop()
            assert not coord._task_start_times
            await coord.start()
        await asyncio.wait_for(entered.get(), 2)
        assert coord.tasks[task_id].started_at == 1100

        async def tick(delay):
            clock.elapsed = 101
            clock.wall = 1101
            coord.running = False

        with monkeypatch.context() as patch:
            patch.setattr(asyncio, "sleep", tick)
            await coord._task_timeout_monitor()
        assert coord.tasks[task_id].status == "running"
        assert coord.tasks[task_id].retry_count == (reason == "retry")
        assert coord.stats["failed_tasks"] == 0
    finally:
        await coord.stop()
        assert not coord._task_start_times


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_terminal_attempt_releases_elapsed_clock(fail, clock, monkeypatch):
    coord = DistributedCoordinator(max_retries=1)
    coord.register_worker("w1", "localhost", 1, {})
    task_id = await coord.submit_task("model", [0])
    original_sleep = asyncio.sleep

    async def work(delay):
        if delay == 2:
            if fail:
                raise RuntimeError("injected failure")
        else:
            await original_sleep(delay)

    monkeypatch.setattr(asyncio, "sleep", work)
    try:
        await coord.start()
        result = await asyncio.wait_for(coord.wait_for_task(task_id), 2)
        assert result.status == ("failed" if fail else "completed")
        assert not coord._task_start_times
    finally:
        await coord.stop()
