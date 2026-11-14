"""Coordinator queues must bind to the loop that actually executes tasks."""

import subprocess
import sys


def run_script(script):
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr


def test_synchronous_construction_after_closed_loop():
    run_script('''
import asyncio
from vitriol.distributed.coordinator import DistributedCoordinator

async def noop():
    pass
asyncio.run(noop())
actual_queue = asyncio.Queue
def require_running_loop():
    asyncio.get_running_loop()
    return actual_queue()
asyncio.Queue = require_running_loop
coordinator = DistributedCoordinator()
coordinator.register_worker('worker', 'localhost', 5555, {})
assert coordinator.get_stats()['pending_tasks'] == 0
try:
    asyncio.get_event_loop()
except RuntimeError:
    pass
else:
    raise AssertionError('synchronous construction installed an event loop')
''')


def test_pending_getter_uses_execution_loop():
    run_script('''
import asyncio
from vitriol.distributed.coordinator import DistributedCoordinator

coordinator = DistributedCoordinator()
async def execute():
    waiting = asyncio.create_task(coordinator.pending_tasks.get())
    await asyncio.sleep(0)
    assert not waiting.done()
    task_id = await coordinator.submit_task('local/model', [0])
    task = await asyncio.wait_for(waiting, timeout=1)
    assert task.id == task_id
    assert coordinator.get_stats()['pending_tasks'] == 0
asyncio.run(execute())
''')


def test_synchronous_unregister_preserves_task_until_loop_starts():
    run_script('''
import asyncio
from vitriol.distributed.coordinator import DistributedCoordinator, GenerationTask

async def noop():
    pass
asyncio.run(noop())
coordinator = DistributedCoordinator()
coordinator.register_worker('worker', 'localhost', 5555, {})
task = GenerationTask('task', 'local/model', [0], 'compact', 'float32',
                      status='running', assigned_worker='worker')
coordinator.tasks[task.id] = task
coordinator.workers['worker'].current_task = task.id
coordinator.unregister_worker('worker')
assert coordinator.get_stats()['pending_tasks'] == 1
assert task.status == 'pending' and task.assigned_worker is None
async def execute():
    assert await asyncio.wait_for(coordinator.pending_tasks.get(), timeout=1) is task
    assert coordinator.pending_tasks.empty()
asyncio.run(execute())
''')


def test_unregister_requeues_immediately_on_existing_queue():
    run_script('''
import asyncio
from vitriol.distributed.coordinator import DistributedCoordinator

async def execute():
    coordinator = DistributedCoordinator()
    coordinator.register_worker('worker', 'localhost', 5555, {})
    task_id = await coordinator.submit_task('local/model', [0])
    task = await coordinator.pending_tasks.get()
    task.status = 'running'
    task.assigned_worker = 'worker'
    coordinator.workers['worker'].current_task = task_id
    coordinator.unregister_worker('worker')
    assert coordinator.pending_tasks.qsize() == 1
    assert coordinator.pending_tasks.get_nowait() is task
    assert coordinator.pending_tasks.empty()
asyncio.run(execute())
''')


def test_dispatch_event_is_created_on_execution_loop_after_synchronous_setup():
    run_script('''
import asyncio
from vitriol.distributed.coordinator import DistributedCoordinator

async def noop():
    pass
asyncio.run(noop())
actual_event = asyncio.Event
def require_running_loop():
    asyncio.get_running_loop()
    return actual_event()
asyncio.Event = require_running_loop
coordinator = DistributedCoordinator()
coordinator.register_worker('worker', 'localhost', 5555, {})
assert coordinator.get_stats()['pending_tasks'] == 0
async def execute():
    await coordinator.start()
    try:
        task_id = await coordinator.submit_task('model', [])
        task = await asyncio.wait_for(coordinator.wait_for_task(task_id), 2)
        assert task.status == 'completed'
    finally:
        await coordinator.stop()
asyncio.run(execute())
''')
