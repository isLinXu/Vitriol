"""
Distributed Weight Generation Coordinator.

Enables distributed model weight generation across multiple nodes:
- Master-worker architecture
- Task distribution and load balancing
- Fault tolerance and recovery
- Progress synchronization
"""

import asyncio
import logging
import math
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    """Worker node status."""
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"


@dataclass
class WorkerInfo:
    """Information about a worker node."""
    id: str
    host: str
    port: int
    status: WorkerStatus
    capabilities: Dict[str, Any]
    last_heartbeat: float
    current_task: Optional[str] = None
    completed_tasks: int = 0
    failed_tasks: int = 0


@dataclass
class GenerationTask:
    """Task for distributed generation."""
    id: str
    model_id: str
    shard_indices: List[int]
    strategy: str
    dtype: str
    status: str = "pending"
    assigned_worker: Optional[str] = None
    created_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict] = None
    error: Optional[str] = None
    retry_count: int = 0

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


class _PendingTaskQueue(asyncio.Queue):
    """Keep normal Queue insertion semantics while notifying the scheduler."""

    def __init__(self, on_put: Callable[[], None]):
        super().__init__()
        self._on_put = on_put

    def put_nowait(self, item):
        super().put_nowait(item)
        self._on_put()


class DistributedCoordinator:
    """
    Coordinator for distributed weight generation.

    Manages a cluster of worker nodes for parallel generation.
    Execution is currently simulated; network transport is not implemented.
    Async operations, including stop/restart, must share one event loop.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5555,
        heartbeat_interval: float = 30,
        task_timeout: float = 300,
        max_retries: int = 3
    ):
        for name, interval in (("heartbeat_interval", heartbeat_interval), ("task_timeout", task_timeout)):
            if not math.isfinite(interval) or interval <= 0:
                raise ValueError(f"{name} must be finite and greater than zero")
        self.host = host
        self.port = port
        self.heartbeat_interval = heartbeat_interval
        self.task_timeout = task_timeout
        self.max_retries = max_retries

        # State
        self.workers: Dict[str, WorkerInfo] = {}
        self.tasks: Dict[str, GenerationTask] = {}
        # Process-local elapsed clocks; public timestamps remain Unix time.
        self._heartbeat_times: Dict[str, float] = {}
        self._task_start_times: Dict[str, float] = {}
        # Python 3.9 binds Queue to the current loop at construction time.
        # The coordinator itself may be constructed before asyncio.run().
        self._pending_tasks: Optional[asyncio.Queue] = None
        self._deferred_tasks: List[GenerationTask] = []
        self._dispatch_ready: Optional[asyncio.Event] = None
        self.running = False
        self._background_tasks: Set[asyncio.Task] = set()
        self._executions: Dict[str, asyncio.Task] = {}
        # Retired attempts stay owned until cancellation has finished.
        self._execution_tasks: Set[asyncio.Task] = set()
        self._stop_task: Optional[asyncio.Task] = None

        # Statistics
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "retried_tasks": 0
        }

        logger.info(f"Coordinator initialized on {host}:{port}")

    @property
    def pending_tasks(self) -> asyncio.Queue:
        """Access the pending queue from the coordinator's execution loop."""
        if self._pending_tasks is None:
            asyncio.get_running_loop()
            queue = _PendingTaskQueue(self._wake_scheduler)
            for task in self._deferred_tasks:
                queue.put_nowait(task)
            self._deferred_tasks.clear()
            self._pending_tasks = queue
        return self._pending_tasks

    def _wake_scheduler(self) -> None:
        if self._dispatch_ready is not None:
            self._dispatch_ready.set()

    async def start(self) -> None:
        """Start the coordinator."""
        while self._stop_task is not None and not self._stop_task.done():
            await asyncio.shield(self._stop_task)
        if self.running:
            return
        self.running = True

        # Start background tasks
        self._background_tasks = {
            asyncio.create_task(self._heartbeat_monitor()),
            asyncio.create_task(self._task_scheduler()),
            asyncio.create_task(self._task_timeout_monitor()),
        }

        logger.info("Coordinator started")

    async def stop(self) -> None:
        """Cancel and join owned coroutines, preserving unfinished work for restart."""
        if self._stop_task is None or self._stop_task.done():
            self.running = False
            self._stop_task = asyncio.create_task(self._shutdown())
        # A cancelled caller must not interrupt shared shutdown cleanup.
        await asyncio.shield(self._stop_task)

    async def _shutdown(self) -> None:
        self.running = False
        owned = self._background_tasks | self._execution_tasks
        self._executions.clear()
        self._task_start_times.clear()
        for execution in owned:
            execution.cancel()
        for task in self.tasks.values():
            if task.status == "running":
                worker = self.workers.get(task.assigned_worker)
                if worker is not None and worker.current_task == task.id:
                    worker.status = WorkerStatus.IDLE
                    worker.current_task = None
                self._requeue(task)
        await asyncio.gather(*owned, return_exceptions=True)
        self._background_tasks.clear()
        self._execution_tasks.clear()
        logger.info("Coordinator stopped")

    def _requeue(self, task: GenerationTask) -> None:
        self._task_start_times.pop(task.id, None)
        task.status = "pending"
        task.assigned_worker = None
        task.started_at = None
        if self._pending_tasks is None:
            self._deferred_tasks.append(task)
        else:
            self._pending_tasks.put_nowait(task)

    def _retire_execution(self, task_id: str) -> None:
        self._task_start_times.pop(task_id, None)
        execution = self._executions.pop(task_id, None)
        if execution is not None and execution is not asyncio.current_task():
            execution.cancel()

    def register_worker(
        self,
        worker_id: str,
        host: str,
        port: int,
        capabilities: Dict[str, Any]
    ) -> bool:
        """
        Register a new worker.

        Args:
            worker_id: Unique worker ID
            host: Worker host
            port: Worker port
            capabilities: Worker capabilities

        Returns:
            True if registered successfully
        """
        if worker_id in self.workers:
            logger.warning(f"Worker {worker_id} already registered")
            return False

        worker = WorkerInfo(
            id=worker_id,
            host=host,
            port=port,
            status=WorkerStatus.IDLE,
            capabilities=capabilities,
            last_heartbeat=time.time()
        )

        self.workers[worker_id] = worker
        self._heartbeat_times[worker_id] = time.monotonic()
        self._wake_scheduler()
        logger.info(f"Worker registered: {worker_id} at {host}:{port}")
        return True

    def unregister_worker(self, worker_id: str) -> None:
        """Unregister a worker."""
        if worker_id in self.workers:
            worker = self.workers[worker_id]

            # Reassign any pending task
            if worker.current_task:
                task = self.tasks.get(worker.current_task)
                if task and task.status == "running":
                    self._retire_execution(task.id)
                    self._requeue(task)
                    logger.warning(f"Reassigned task {task.id} from disconnected worker")

            del self.workers[worker_id]
            self._heartbeat_times.pop(worker_id, None)
            logger.info(f"Worker unregistered: {worker_id}")

    def update_heartbeat(self, worker_id: str) -> None:
        """Update worker heartbeat."""
        if worker_id in self.workers:
            self.workers[worker_id].last_heartbeat = time.time()
            self._heartbeat_times[worker_id] = time.monotonic()

    async def submit_task(
        self,
        model_id: str,
        shard_indices: List[int],
        strategy: str = "compact",
        dtype: str = "bfloat16"
    ) -> str:
        """
        Submit a generation task.

        Args:
            model_id: Model to generate
            shard_indices: Shard indices to generate
            strategy: Generation strategy
            dtype: Data type

        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())

        task = GenerationTask(
            id=task_id,
            model_id=model_id,
            shard_indices=list(shard_indices),
            strategy=strategy,
            dtype=dtype
        )

        self.tasks[task_id] = task
        self.stats["total_tasks"] += 1

        await self.pending_tasks.put(task)
        logger.info(f"Task submitted: {task_id} for {model_id} shards {shard_indices}")

        return task_id

    async def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get a detached status snapshot, including shards and nested results."""
        if task_id not in self.tasks:
            return None

        task = self.tasks[task_id]
        return {
            "id": task.id,
            "status": task.status,
            "model_id": task.model_id,
            "shard_indices": list(task.shard_indices),
            "assigned_worker": task.assigned_worker,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "result": deepcopy(task.result),
            "error": task.error,
            "retry_count": task.retry_count
        }

    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Optional[GenerationTask]:
        """Wait for completion without cancelling the underlying task.

        None or positive infinity waits indefinitely. Nonpositive timeouts only
        inspect the current status. NaN is rejected. The returned terminal task
        is the live task object; use get_task_status() for a detached snapshot.
        """
        if timeout is not None and math.isnan(timeout):
            raise ValueError("timeout must not be NaN")
        start = time.monotonic()

        while True:
            task = self.tasks.get(task_id)
            if not task:
                return None

            if task.status in ["completed", "failed"]:
                return task

            delay = 0.1
            if timeout is not None:
                remaining = timeout - (time.monotonic() - start)
                if remaining <= 0:
                    return None
                delay = min(delay, remaining)

            await asyncio.sleep(delay)

    async def _heartbeat_monitor(self):
        """Monitor worker heartbeats."""
        while self.running:
            await asyncio.sleep(self.heartbeat_interval)

            now = time.monotonic()
            dead_workers = []

            for worker_id in self.workers:
                last_seen = self._heartbeat_times.get(worker_id)
                if last_seen is not None and now - last_seen > self.heartbeat_interval * 2:
                    logger.warning(f"Worker {worker_id} missed heartbeat")
                    dead_workers.append(worker_id)

            for worker_id in dead_workers:
                self.unregister_worker(worker_id)

    async def _task_scheduler(self):
        """Dispatch FIFO work when both a pending task and an idle worker exist."""
        # Like Queue, Event must be created on the execution loop for Python 3.9.
        if self._dispatch_ready is None:
            self._dispatch_ready = asyncio.Event()
        ready = self._dispatch_ready
        while self.running:
            try:
                # Clear before checking state. All checks and assignment below
                # are synchronous, so same-loop notifications cannot be lost.
                ready.clear()
                worker = self._select_worker()
                if worker is None:
                    await ready.wait()
                    continue
                try:
                    task = self.pending_tasks.get_nowait()
                except asyncio.QueueEmpty:
                    await ready.wait()
                    continue
                if task.status != "pending":
                    continue

                task.status = "running"
                task.assigned_worker = worker.id
                task.started_at = time.time()
                self._task_start_times[task.id] = time.monotonic()
                task.error = None
                task.completed_at = None
                task.result = None

                worker.status = WorkerStatus.BUSY
                worker.current_task = task.id
                logger.info(f"Task {task.id} assigned to worker {worker.id}")

                # Simulate task execution (in real implementation, send to worker)
                execution = asyncio.create_task(self._execute_task(task, worker))
                self._executions[task.id] = execution
                self._execution_tasks.add(execution)
                execution.add_done_callback(self._execution_tasks.discard)

            except Exception as e:
                logger.error(f"Task scheduler error: {e}")
                await asyncio.sleep(1)

    def _select_worker(self) -> Optional[WorkerInfo]:
        """Select best available worker."""
        available = [
            w for w in self.workers.values()
            if w.status == WorkerStatus.IDLE
        ]

        if not available:
            return None

        # Select worker with fewest completed tasks (load balancing)
        return min(available, key=lambda w: w.completed_tasks)

    async def _execute_task(self, task: GenerationTask, worker: WorkerInfo):
        """Execute task on worker (simulated)."""
        try:
            # Simulate work
            await asyncio.sleep(len(task.shard_indices) * 2)

            if self._executions.get(task.id) is not asyncio.current_task():
                return

            # Mark complete
            task.status = "completed"
            task.completed_at = time.time()
            task.result = {
                "shards_generated": len(task.shard_indices),
                "total_size_mb": len(task.shard_indices) * 100
            }

            worker.status = WorkerStatus.IDLE
            worker.current_task = None
            worker.completed_tasks += 1

            self.stats["completed_tasks"] += 1
            self._wake_scheduler()

            logger.info(f"Task {task.id} completed")

        except Exception as e:
            if self._executions.get(task.id) is asyncio.current_task():
                await self._handle_task_failure(task, worker, str(e))
        finally:
            if self._executions.get(task.id) is asyncio.current_task():
                self._retire_execution(task.id)

    async def _handle_task_failure(
        self,
        task: GenerationTask,
        worker: WorkerInfo,
        error: str
    ):
        """Handle task failure with retry."""
        if task.status != "running" or task.assigned_worker != worker.id:
            return
        self._retire_execution(task.id)
        task.error = error
        task.retry_count += 1

        worker.failed_tasks += 1
        worker.status = WorkerStatus.IDLE
        worker.current_task = None
        self._wake_scheduler()

        if task.retry_count < self.max_retries:
            # Retry
            self._requeue(task)

            self.stats["retried_tasks"] += 1
            logger.warning(f"Task {task.id} failed, retrying ({task.retry_count}/{self.max_retries})")
        else:
            # Max retries reached
            task.status = "failed"
            task.completed_at = time.time()
            self.stats["failed_tasks"] += 1
            logger.error(f"Task {task.id} failed permanently after {self.max_retries} retries")

    async def _task_timeout_monitor(self):
        """Monitor for timed out tasks."""
        while self.running:
            await asyncio.sleep(min(10, self.task_timeout))

            now = time.monotonic()

            for task in self.tasks.values():
                started = self._task_start_times.get(task.id)
                if task.status == "running" and started is not None:
                    if now - started >= self.task_timeout:
                        logger.warning(f"Task {task.id} timed out")

                        # Mark worker as failed
                        if task.assigned_worker and task.assigned_worker in self.workers:
                            worker = self.workers[task.assigned_worker]
                            await self._handle_task_failure(
                                task, worker, "Task timed out"
                            )

    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        return {
            **self.stats,
            "active_workers": len([w for w in self.workers.values() if w.status != WorkerStatus.OFFLINE]),
            "pending_tasks": self._pending_tasks.qsize() if self._pending_tasks is not None else len(self._deferred_tasks),
            "running_tasks": len([t for t in self.tasks.values() if t.status == "running"]),
            "workers": {
                wid: {
                    "status": w.status.value,
                    "completed": w.completed_tasks,
                    "failed": w.failed_tasks
                }
                for wid, w in self.workers.items()
            }
        }


class WorkerNode:
    """
    Worker node for distributed generation.

    Connects to coordinator and executes generation tasks.
    Currently a simulation without network transport. Start/stop share one loop.
    """

    def __init__(
        self,
        coordinator_host: str,
        coordinator_port: int,
        worker_id: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 0
    ):
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.worker_id = worker_id or str(uuid.uuid4())
        self.host = host
        self.port = port

        self.running = False
        self._background_tasks: Set[asyncio.Task] = set()
        self._stop_task: Optional[asyncio.Task] = None
        self.capabilities = {
            "cpu_count": 4,
            "memory_gb": 16,
            "gpu_count": 0,
            "strategies": ["compact", "random", "ultra"]
        }

    async def start(self) -> None:
        """Start worker node."""
        while self._stop_task is not None and not self._stop_task.done():
            await asyncio.shield(self._stop_task)
        if self.running:
            return
        self.running = True

        # Register with coordinator
        # In real implementation, use HTTP/gRPC
        logger.info(f"Worker {self.worker_id} started")

        self._background_tasks = {
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._task_processor()),
        }

    async def stop(self) -> None:
        """Stop worker node and join its background coroutines."""
        if self._stop_task is None or self._stop_task.done():
            self.running = False
            self._stop_task = asyncio.create_task(self._shutdown())
        await asyncio.shield(self._stop_task)

    async def _shutdown(self) -> None:
        self.running = False
        for task in self._background_tasks:
            task.cancel()
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()
        logger.info(f"Worker {self.worker_id} stopped")

    async def _heartbeat_loop(self):
        """Send heartbeats to coordinator."""
        while self.running:
            try:
                # In real implementation, send HTTP request
                logger.debug(f"Worker {self.worker_id} heartbeat")
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"Heartbeat failed: {e}")

    async def _task_processor(self):
        """Process tasks from coordinator."""
        while self.running:
            # In real implementation, poll for tasks
            await asyncio.sleep(1)
