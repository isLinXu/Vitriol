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
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

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


class DistributedCoordinator:
    """
    Coordinator for distributed weight generation.

    Manages a cluster of worker nodes for parallel generation.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5555,
        heartbeat_interval: int = 30,
        task_timeout: int = 300,
        max_retries: int = 3
    ):
        self.host = host
        self.port = port
        self.heartbeat_interval = heartbeat_interval
        self.task_timeout = task_timeout
        self.max_retries = max_retries

        # State
        self.workers: Dict[str, WorkerInfo] = {}
        self.tasks: Dict[str, GenerationTask] = {}
        self.pending_tasks: asyncio.Queue = asyncio.Queue()
        self.running = False

        # Statistics
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "retried_tasks": 0
        }

        logger.info(f"Coordinator initialized on {host}:{port}")

    async def start(self):
        """Start the coordinator."""
        self.running = True

        # Start background tasks
        asyncio.create_task(self._heartbeat_monitor())
        asyncio.create_task(self._task_scheduler())
        asyncio.create_task(self._task_timeout_monitor())

        logger.info("Coordinator started")

    async def stop(self):
        """Stop the coordinator."""
        self.running = False
        logger.info("Coordinator stopped")

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
        logger.info(f"Worker registered: {worker_id} at {host}:{port}")
        return True

    def unregister_worker(self, worker_id: str):
        """Unregister a worker."""
        if worker_id in self.workers:
            worker = self.workers[worker_id]

            # Reassign any pending task
            if worker.current_task:
                task = self.tasks.get(worker.current_task)
                if task and task.status == "running":
                    task.status = "pending"
                    task.assigned_worker = None
                    asyncio.create_task(self.pending_tasks.put(task))
                    logger.warning(f"Reassigned task {task.id} from disconnected worker")

            del self.workers[worker_id]
            logger.info(f"Worker unregistered: {worker_id}")

    def update_heartbeat(self, worker_id: str):
        """Update worker heartbeat."""
        if worker_id in self.workers:
            self.workers[worker_id].last_heartbeat = time.time()

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
            shard_indices=shard_indices,
            strategy=strategy,
            dtype=dtype
        )

        self.tasks[task_id] = task
        self.stats["total_tasks"] += 1

        await self.pending_tasks.put(task)
        logger.info(f"Task submitted: {task_id} for {model_id} shards {shard_indices}")

        return task_id

    async def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get task status."""
        if task_id not in self.tasks:
            return None

        task = self.tasks[task_id]
        return {
            "id": task.id,
            "status": task.status,
            "model_id": task.model_id,
            "shard_indices": task.shard_indices,
            "assigned_worker": task.assigned_worker,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "result": task.result,
            "error": task.error,
            "retry_count": task.retry_count
        }

    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Optional[GenerationTask]:
        """Wait for task completion."""
        start = time.time()

        while True:
            task = self.tasks.get(task_id)
            if not task:
                return None

            if task.status in ["completed", "failed"]:
                return task

            if timeout and (time.time() - start) > timeout:
                return None

            await asyncio.sleep(0.1)

    async def _heartbeat_monitor(self):
        """Monitor worker heartbeats."""
        while self.running:
            await asyncio.sleep(self.heartbeat_interval)

            now = time.time()
            dead_workers = []

            for worker_id, worker in self.workers.items():
                if now - worker.last_heartbeat > self.heartbeat_interval * 2:
                    logger.warning(f"Worker {worker_id} missed heartbeat")
                    dead_workers.append(worker_id)

            for worker_id in dead_workers:
                self.unregister_worker(worker_id)

    async def _task_scheduler(self):
        """Schedule tasks to available workers."""
        while self.running:
            try:
                # Get pending task
                task = await asyncio.wait_for(self.pending_tasks.get(), timeout=1)

                # Find available worker
                worker = self._select_worker()

                if worker:
                    # Assign task
                    task.status = "running"
                    task.assigned_worker = worker.id
                    task.started_at = time.time()

                    worker.status = WorkerStatus.BUSY
                    worker.current_task = task.id

                    logger.info(f"Task {task.id} assigned to worker {worker.id}")

                    # Simulate task execution (in real implementation, send to worker)
                    asyncio.create_task(self._execute_task(task, worker))
                else:
                    # No worker available, requeue
                    await self.pending_tasks.put(task)
                    await asyncio.sleep(1)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Task scheduler error: {e}")

    def _select_worker(self) -> Optional[WorkerInfo]:
        """Select best available worker."""
        available = [
            w for w in self.workers.values()
            if w.status == WorkerStatus.IDLE
        ]

        if not available:
            return None

        # Select worker with most completed tasks (load balancing)
        return min(available, key=lambda w: w.completed_tasks)

    async def _execute_task(self, task: GenerationTask, worker: WorkerInfo):
        """Execute task on worker (simulated)."""
        try:
            # Simulate work
            await asyncio.sleep(len(task.shard_indices) * 2)

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

            logger.info(f"Task {task.id} completed")

        except Exception as e:
            await self._handle_task_failure(task, worker, str(e))

    async def _handle_task_failure(
        self,
        task: GenerationTask,
        worker: WorkerInfo,
        error: str
    ):
        """Handle task failure with retry."""
        task.error = error
        task.retry_count += 1

        worker.failed_tasks += 1
        worker.status = WorkerStatus.IDLE
        worker.current_task = None

        if task.retry_count < self.max_retries:
            # Retry
            task.status = "pending"
            task.assigned_worker = None
            await self.pending_tasks.put(task)

            self.stats["retried_tasks"] += 1
            logger.warning(f"Task {task.id} failed, retrying ({task.retry_count}/{self.max_retries})")
        else:
            # Max retries reached
            task.status = "failed"
            self.stats["failed_tasks"] += 1
            logger.error(f"Task {task.id} failed permanently after {self.max_retries} retries")

    async def _task_timeout_monitor(self):
        """Monitor for timed out tasks."""
        while self.running:
            await asyncio.sleep(10)

            now = time.time()

            for task in self.tasks.values():
                if task.status == "running" and task.started_at:
                    if now - task.started_at > self.task_timeout:
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
            "pending_tasks": self.pending_tasks.qsize(),
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
        self.capabilities = {
            "cpu_count": 4,
            "memory_gb": 16,
            "gpu_count": 0,
            "strategies": ["compact", "random", "ultra"]
        }

    async def start(self):
        """Start worker node."""
        self.running = True

        # Register with coordinator
        # In real implementation, use HTTP/gRPC
        logger.info(f"Worker {self.worker_id} started")

        # Start heartbeat
        asyncio.create_task(self._heartbeat_loop())

        # Start task processor
        asyncio.create_task(self._task_processor())

    async def stop(self):
        """Stop worker node."""
        self.running = False
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
