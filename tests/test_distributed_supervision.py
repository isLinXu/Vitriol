"""Unexpected background-loop failures must recover without duplicate loops."""

import asyncio

import pytest

from vitriol.distributed.coordinator import DistributedCoordinator, WorkerNode


@pytest.mark.asyncio
@pytest.mark.parametrize("loop_name", ["_heartbeat_monitor", "_task_timeout_monitor"])
async def test_coordinator_restarts_failed_background_monitor(loop_name, monkeypatch, caplog):
    coord = DistributedCoordinator()
    calls = 0
    recovered = asyncio.Event()
    release = asyncio.Event()

    async def fail_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected monitor failure")
        recovered.set()
        await release.wait()

    monkeypatch.setattr(coord, loop_name, fail_once)
    try:
        await coord.start()
        with caplog.at_level("ERROR"):
            await asyncio.wait_for(recovered.wait(), 2)
        assert calls == 2
        assert coord.running
        assert "injected monitor failure" in caplog.text
    finally:
        release.set()
        await coord.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("loop_name", ["_heartbeat_loop", "_task_processor"])
async def test_worker_restarts_failed_background_loop(loop_name, monkeypatch, caplog):
    worker = WorkerNode("localhost", 5555)
    calls = 0
    recovered = asyncio.Event()
    release = asyncio.Event()

    async def fail_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected worker loop failure")
        recovered.set()
        await release.wait()

    monkeypatch.setattr(worker, loop_name, fail_once)
    try:
        await worker.start()
        with caplog.at_level("ERROR"):
            await asyncio.wait_for(recovered.wait(), 2)
        assert calls == 2
        assert worker.running
        assert "injected worker loop failure" in caplog.text
    finally:
        release.set()
        await worker.stop()


@pytest.mark.asyncio
async def test_failed_background_loop_observes_shutdown_during_backoff(monkeypatch):
    coord = DistributedCoordinator()
    calls = 0

    async def fail_forever():
        nonlocal calls
        calls += 1
        raise RuntimeError("persistent failure")

    monkeypatch.setattr(coord, "_heartbeat_monitor", fail_forever)
    await coord.start()
    await asyncio.sleep(0.01)
    await asyncio.wait_for(coord.stop(), 2)
    assert not coord.running
    assert calls == 1


@pytest.mark.asyncio
async def test_unexpected_normal_exit_is_restarted(monkeypatch, caplog):
    worker = WorkerNode("localhost", 5555)
    calls = 0
    recovered = asyncio.Event()
    release = asyncio.Event()

    async def exit_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            return
        recovered.set()
        await release.wait()

    monkeypatch.setattr(worker, "_task_processor", exit_once)
    try:
        await worker.start()
        with caplog.at_level("WARNING"):
            await asyncio.wait_for(recovered.wait(), 2)
        assert calls == 2
        assert "exited unexpectedly" in caplog.text
    finally:
        release.set()
        await worker.stop()
