"""
Checkpoint and Recovery System for Vitriol.

Provides fault tolerance through:
- Automatic checkpointing
- State persistence
- Recovery from failures
- Resume capabilities
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """Checkpoint data structure."""
    id: str
    operation: str
    state: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: float
    version: str = "1.0"

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "operation": self.operation,
            "state": self.state,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "version": self.version
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Checkpoint":
        return cls(**data)

    def compute_hash(self) -> str:
        """Compute checkpoint hash for integrity."""
        data = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class CheckpointManager:
    """
    Manages checkpoints for long-running operations.

    Features:
        - Automatic checkpointing at intervals
        - State serialization
        - Recovery from checkpoints
        - Checkpoint cleanup

    Example:
        >>> manager = CheckpointManager("./checkpoints")
        >>> with manager.checkpoint_context("generation", state) as ctx:
        ...     # Long running operation
        ...     pass
    """

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        auto_save_interval: int = 60,
        max_checkpoints: int = 10
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            auto_save_interval: Auto-save interval in seconds
            max_checkpoints: Maximum checkpoints to keep per operation
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.auto_save_interval = auto_save_interval
        self.max_checkpoints = max_checkpoints

        self._current_checkpoint: Optional[Checkpoint] = None
        self._last_save_time: float = 0

        logger.info(f"CheckpointManager initialized: {checkpoint_dir}")

    def create_checkpoint(
        self,
        operation: str,
        state: Dict[str, Any],
        metadata: Optional[Dict] = None
    ) -> Checkpoint:
        """
        Create a new checkpoint.

        Args:
            operation: Operation name
            state: State to checkpoint
            metadata: Additional metadata

        Returns:
            Checkpoint object
        """
        checkpoint_id = f"{operation}_{int(time.time())}"

        checkpoint = Checkpoint(
            id=checkpoint_id,
            operation=operation,
            state=state,
            metadata=metadata or {},
            timestamp=time.time()
        )

        self._save_checkpoint(checkpoint)
        self._cleanup_old_checkpoints(operation)

        logger.info(f"Checkpoint created: {checkpoint_id}")
        return checkpoint

    def _save_checkpoint(self, checkpoint: Checkpoint):
        """Save checkpoint to disk."""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint.id}.json"

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint.to_dict(), f, indent=2, default=str)

        # Also save state separately if large
        state_path = self.checkpoint_dir / f"{checkpoint.id}_state.pkl"
        with open(state_path, 'wb') as f:
            pickle.dump(checkpoint.state, f)

        self._last_save_time = time.time()

    def load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """
        Load a checkpoint.

        Args:
            checkpoint_id: Checkpoint ID

        Returns:
            Checkpoint or None if not found
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.json"

        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path) as f:
                data = json.load(f)

            # Load state from pickle if exists (restricted deserialization)
            state_path = self.checkpoint_dir / f"{checkpoint_id}_state.pkl"
            if state_path.exists():
                with open(state_path, 'rb') as f:
                    # Restrict unpickling to safe built-in types only
                    class _RestrictedUnpickler(pickle.Unpickler):
                        """Unpickler that only allows safe built-in types."""
                        ALLOWED_CLASSES = {
                            'builtins': ('dict', 'list', 'tuple', 'set', 'frozenset', 'str', 'int', 'float', 'bool', 'bytes', 'NoneType'),
                            'collections': ('OrderedDict', 'defaultdict'),
                            'numpy.core.multiarray': ('_reconstruct', 'scalar'),
                            'numpy': ('dtype', 'ndarray'),
                            'torch': ('Tensor',),
                            'torch._utils': ('_rebuild_tensor_v2',),
                        }

                        def find_class(self, module, name) -> Any:
                            if module in self.ALLOWED_CLASSES and name in self.ALLOWED_CLASSES[module]:
                                return super().find_class(module, name)
                            raise pickle.UnpicklingError(
                                f"Forbidden class: {module}.{name} — "
                                f"only allowlisted types can be deserialized"
                            )

                    data['state'] = _RestrictedUnpickler(f).load()

            checkpoint = Checkpoint.from_dict(data)
            logger.info(f"Checkpoint loaded: {checkpoint_id}")
            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None

    def find_latest_checkpoint(self, operation: str) -> Optional[Checkpoint]:
        """
        Find the latest checkpoint for an operation.

        Args:
            operation: Operation name

        Returns:
            Latest checkpoint or None
        """
        checkpoints = self.list_checkpoints(operation)

        if not checkpoints:
            return None

        # Sort by timestamp
        latest = max(checkpoints, key=lambda c: c.timestamp)
        return latest

    def list_checkpoints(self, operation: Optional[str] = None) -> list:
        """
        List available checkpoints.

        Args:
            operation: Filter by operation

        Returns:
            List of checkpoints
        """
        checkpoints = []

        for checkpoint_file in self.checkpoint_dir.glob("*.json"):
            try:
                with open(checkpoint_file) as f:
                    data = json.load(f)

                if operation is None or data.get('operation') == operation:
                    checkpoints.append(Checkpoint.from_dict(data))
            except Exception as e:
                logger.debug("Skipping corrupted checkpoint %s: %s", checkpoint_file, e)
                continue

        return checkpoints

    def delete_checkpoint(self, checkpoint_id: str) -> None:
        """Delete a checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.json"
        state_path = self.checkpoint_dir / f"{checkpoint_id}_state.pkl"

        if checkpoint_path.exists():
            checkpoint_path.unlink()

        if state_path.exists():
            state_path.unlink()

        logger.info(f"Checkpoint deleted: {checkpoint_id}")

    def _cleanup_old_checkpoints(self, operation: str):
        """Clean up old checkpoints for an operation."""
        checkpoints = self.list_checkpoints(operation)

        if len(checkpoints) <= self.max_checkpoints:
            return

        # Sort by timestamp and delete oldest
        checkpoints.sort(key=lambda c: c.timestamp)
        to_delete = checkpoints[:-self.max_checkpoints]

        for checkpoint in to_delete:
            self.delete_checkpoint(checkpoint.id)

    def checkpoint_context(
        self,
        operation: str,
        get_state_fn: Callable[[], Dict[str, Any]],
        metadata: Optional[Dict] = None
    ) -> Any:
        """
        Context manager for automatic checkpointing.

        Args:
            operation: Operation name
            get_state_fn: Function to get current state
            metadata: Additional metadata

        Example:
            >>> def get_state():
            ...     return {"progress": progress, "data": data}
            >>>
            >>> with manager.checkpoint_context("gen", get_state) as ctx:
            ...     # Do work
            ...     if ctx.should_checkpoint():
            ...         ctx.save()
        """
        return CheckpointContext(self, operation, get_state_fn, metadata)


class CheckpointContext:
    """Context for automatic checkpointing."""

    def __init__(
        self,
        manager: CheckpointManager,
        operation: str,
        get_state_fn: Callable[[], Dict[str, Any]],
        metadata: Optional[Dict]
    ):
        self.manager = manager
        self.operation = operation
        self.get_state_fn = get_state_fn
        self.metadata = metadata
        self.checkpoint: Optional[Checkpoint] = None

    def __enter__(self):
        """Enter context."""
        # Try to resume from checkpoint
        latest = self.manager.find_latest_checkpoint(self.operation)
        if latest:
            logger.info(f"Resuming from checkpoint: {latest.id}")
            self.checkpoint = latest

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if exc_type is None:
            # Success - clean up checkpoints
            for cp in self.manager.list_checkpoints(self.operation):
                self.manager.delete_checkpoint(cp.id)
            logger.info(f"Operation {self.operation} completed, checkpoints cleaned up")
        else:
            # Failure - save final checkpoint
            self.save()
            logger.error(f"Operation {self.operation} failed, state saved")

    def should_checkpoint(self) -> bool:
        """Check if it's time to save a checkpoint."""
        return (time.time() - self.manager._last_save_time) >= self.manager.auto_save_interval

    def save(self) -> Checkpoint:
        """Save current state as checkpoint."""
        state = self.get_state_fn()
        self.checkpoint = self.manager.create_checkpoint(
            self.operation,
            state,
            self.metadata
        )
        return self.checkpoint

    def get_resumed_state(self) -> Optional[Dict]:
        """Get state from resumed checkpoint."""
        if self.checkpoint:
            return self.checkpoint.state
        return None


class RecoveryManager:
    """
    Manages recovery from failures.

    Provides automatic retry with exponential backoff.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0
    ):
        """
        Initialize recovery manager.

        Args:
            max_retries: Maximum retry attempts
            base_delay: Initial delay between retries
            max_delay: Maximum delay between retries
            exponential_base: Exponential backoff multiplier
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

        self._retry_count: Dict[str, int] = {}

    async def execute_with_retry(
        self,
        operation_id: str,
        operation_fn: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute operation with retry logic.

        Args:
            operation_id: Unique operation identifier
            operation_fn: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Operation result

        Raises:
            Exception: If all retries fail
        """
        if operation_id not in self._retry_count:
            self._retry_count[operation_id] = 0

        while self._retry_count[operation_id] < self.max_retries:
            try:
                result = await operation_fn(*args, **kwargs)

                # Success - reset retry count
                if self._retry_count[operation_id] > 0:
                    logger.info(f"Operation {operation_id} succeeded after {self._retry_count[operation_id]} retries")

                self._retry_count[operation_id] = 0
                return result

            except Exception as e:
                self._retry_count[operation_id] += 1
                attempt = self._retry_count[operation_id]

                if attempt >= self.max_retries:
                    logger.error(f"Operation {operation_id} failed after {self.max_retries} retries: {e}")
                    raise

                # Calculate delay
                delay = min(
                    self.base_delay * (self.exponential_base ** (attempt - 1)),
                    self.max_delay
                )

                logger.warning(
                    f"Operation {operation_id} failed (attempt {attempt}/{self.max_retries}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )

                await asyncio.sleep(delay)

    def get_retry_count(self, operation_id: str) -> int:
        """Get retry count for an operation."""
        return self._retry_count.get(operation_id, 0)

    def reset_retry_count(self, operation_id: str) -> None:
        """Reset retry count for an operation."""
        self._retry_count[operation_id] = 0
