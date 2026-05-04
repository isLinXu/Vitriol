"""Tests for vitriol.resilience.checkpoint module."""

import pytest
import json
import pickle
import time
from unittest.mock import Mock, patch
from pathlib import Path

from vitriol.resilience.checkpoint import (
    Checkpoint,
    CheckpointManager,
    CheckpointContext,
    RecoveryManager,
)


class TestCheckpoint:
    """Tests for Checkpoint dataclass."""

    def test_creation(self):
        """Test Checkpoint creation."""
        cp = Checkpoint(
            id="cp1",
            operation="test_op",
            state={"key": "value"},
            metadata={"version": "1.0"},
            timestamp=12345.0
        )
        assert cp.id == "cp1"
        assert cp.operation == "test_op"
        assert cp.state == {"key": "value"}
        assert cp.version == "1.0"

    def test_to_dict(self):
        """Test to_dict method."""
        cp = Checkpoint(
            id="cp1", operation="test", state={"a": 1}, metadata={}, timestamp=0.0
        )
        d = cp.to_dict()
        assert d["id"] == "cp1"
        assert d["operation"] == "test"
        assert d["state"] == {"a": 1}

    def test_from_dict(self):
        """Test from_dict classmethod."""
        data = {
            "id": "cp1",
            "operation": "test",
            "state": {"a": 1},
            "metadata": {},
            "timestamp": 0.0,
            "version": "1.0"
        }
        cp = Checkpoint.from_dict(data)
        assert cp.id == "cp1"
        assert cp.state == {"a": 1}

    def test_compute_hash(self):
        """Test hash computation."""
        cp = Checkpoint(
            id="cp1", operation="test", state={"a": 1}, metadata={}, timestamp=0.0
        )
        hash1 = cp.compute_hash()
        hash2 = cp.compute_hash()
        assert len(hash1) == 16
        assert hash1 == hash2

    def test_compute_hash_different(self):
        """Test different checkpoints have different hashes."""
        cp1 = Checkpoint(
            id="cp1", operation="test", state={"a": 1}, metadata={}, timestamp=0.0
        )
        cp2 = Checkpoint(
            id="cp2", operation="test", state={"a": 2}, metadata={}, timestamp=0.0
        )
        assert cp1.compute_hash() != cp2.compute_hash()


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    def test_init_creates_directory(self, tmp_path):
        """Test initialization creates checkpoint directory."""
        cp_dir = tmp_path / "checkpoints"
        manager = CheckpointManager(str(cp_dir))
        assert cp_dir.exists()
        assert cp_dir.is_dir()

    def test_create_checkpoint(self, tmp_path):
        """Test checkpoint creation."""
        manager = CheckpointManager(str(tmp_path))
        cp = manager.create_checkpoint("test_op", {"progress": 50})

        assert cp.id.startswith("test_op_")
        assert cp.state == {"progress": 50}
        assert (tmp_path / f"{cp.id}.json").exists()

    def test_create_checkpoint_with_metadata(self, tmp_path):
        """Test checkpoint with metadata."""
        manager = CheckpointManager(str(tmp_path))
        cp = manager.create_checkpoint("test", {}, metadata={"tag": "v1"})

        assert cp.metadata == {"tag": "v1"}

    def test_load_checkpoint(self, tmp_path):
        """Test loading a checkpoint."""
        manager = CheckpointManager(str(tmp_path))
        original = manager.create_checkpoint("test", {"data": "value"})

        loaded = manager.load_checkpoint(original.id)
        assert loaded is not None
        assert loaded.id == original.id
        assert loaded.state == {"data": "value"}

    def test_load_checkpoint_not_found(self, tmp_path):
        """Test loading non-existent checkpoint."""
        manager = CheckpointManager(str(tmp_path))
        loaded = manager.load_checkpoint("nonexistent")
        assert loaded is None

    def test_load_checkpoint_with_pickle_state(self, tmp_path):
        """Test loading checkpoint with pickle state."""
        manager = CheckpointManager(str(tmp_path))
        original = manager.create_checkpoint("test", {"complex": [1, 2, 3]})

        loaded = manager.load_checkpoint(original.id)
        assert loaded.state == {"complex": [1, 2, 3]}

    def test_load_checkpoint_corrupted(self, tmp_path, caplog):
        """Test loading corrupted checkpoint."""
        manager = CheckpointManager(str(tmp_path))
        cp_file = tmp_path / "corrupted.json"
        cp_file.write_text("not valid json")

        loaded = manager.load_checkpoint("corrupted")
        assert loaded is None

    def test_find_latest_checkpoint(self, tmp_path):
        """Test finding latest checkpoint."""
        manager = CheckpointManager(str(tmp_path))

        cp1 = manager.create_checkpoint("test", {"v": 1})
        time.sleep(0.01)
        cp2 = manager.create_checkpoint("test", {"v": 2})

        latest = manager.find_latest_checkpoint("test")
        assert latest is not None
        assert latest.state == {"v": 2}

    def test_find_latest_checkpoint_none(self, tmp_path):
        """Test finding latest when none exist."""
        manager = CheckpointManager(str(tmp_path))
        latest = manager.find_latest_checkpoint("test")
        assert latest is None

    def test_list_checkpoints(self, tmp_path):
        """Test listing checkpoints."""
        manager = CheckpointManager(str(tmp_path))
        # Each create_checkpoint calls time.time() 3 times
        timestamps = [1000.0, 1000.0, 1000.0, 1001.0, 1001.0, 1001.0, 1002.0, 1002.0, 1002.0]
        with patch("vitriol.resilience.checkpoint.time.time", side_effect=timestamps):
            manager.create_checkpoint("op1", {})
            manager.create_checkpoint("op1", {})
            manager.create_checkpoint("op2", {})

        all_cps = manager.list_checkpoints()
        assert len(all_cps) == 3

        op1_cps = manager.list_checkpoints("op1")
        assert len(op1_cps) == 2

    def test_list_checkpoints_corrupted(self, tmp_path, caplog):
        """Test listing with corrupted files."""
        manager = CheckpointManager(str(tmp_path))
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("invalid")

        with caplog.at_level("DEBUG"):
            cps = manager.list_checkpoints()
        assert len(cps) == 0

    def test_delete_checkpoint(self, tmp_path):
        """Test deleting a checkpoint."""
        manager = CheckpointManager(str(tmp_path))
        cp = manager.create_checkpoint("test", {})

        manager.delete_checkpoint(cp.id)
        assert not (tmp_path / f"{cp.id}.json").exists()

    def test_delete_checkpoint_nonexistent(self, tmp_path, caplog):
        """Test deleting non-existent checkpoint."""
        manager = CheckpointManager(str(tmp_path))
        # Should not raise
        manager.delete_checkpoint("nonexistent")

    def test_cleanup_old_checkpoints(self, tmp_path):
        """Test automatic cleanup of old checkpoints."""
        manager = CheckpointManager(str(tmp_path), max_checkpoints=2)

        # Each create_checkpoint calls time.time() 3 times
        timestamps = [1000.0, 1000.0, 1000.0, 1001.0, 1001.0, 1001.0, 1002.0, 1002.0, 1002.0]
        with patch("vitriol.resilience.checkpoint.time.time", side_effect=timestamps):
            cp1 = manager.create_checkpoint("test", {"v": 1})
            cp2 = manager.create_checkpoint("test", {"v": 2})
            cp3 = manager.create_checkpoint("test", {"v": 3})

        # Old checkpoint should be cleaned up
        assert not (tmp_path / f"{cp1.id}.json").exists()
        assert (tmp_path / f"{cp2.id}.json").exists()
        assert (tmp_path / f"{cp3.id}.json").exists()

    def test_cleanup_not_needed(self, tmp_path):
        """Test cleanup when under limit."""
        manager = CheckpointManager(str(tmp_path), max_checkpoints=10)

        cp = manager.create_checkpoint("test", {})
        # Should still exist
        assert (tmp_path / f"{cp.id}.json").exists()

    def test_checkpoint_context_success(self, tmp_path):
        """Test context manager on success."""
        manager = CheckpointManager(str(tmp_path))

        def get_state():
            return {"progress": 100}

        with manager.checkpoint_context("test", get_state) as ctx:
            pass

        # Checkpoints should be cleaned up on success
        assert len(manager.list_checkpoints("test")) == 0

    def test_checkpoint_context_failure(self, tmp_path):
        """Test context manager on failure."""
        manager = CheckpointManager(str(tmp_path))

        def get_state():
            return {"progress": 50}

        try:
            with manager.checkpoint_context("test", get_state) as ctx:
                raise RuntimeError("Test error")
        except RuntimeError:
            pass

        # Checkpoint should be saved on failure
        assert len(manager.list_checkpoints("test")) == 1

    def test_checkpoint_context_resume(self, tmp_path):
        """Test context manager resumes from checkpoint."""
        manager = CheckpointManager(str(tmp_path))

        # Create existing checkpoint
        manager.create_checkpoint("test", {"progress": 50})

        def get_state():
            return {"progress": 100}

        with manager.checkpoint_context("test", get_state) as ctx:
            resumed = ctx.get_resumed_state()
            assert resumed == {"progress": 50}


class TestCheckpointContext:
    """Tests for CheckpointContext class."""

    def test_should_checkpoint(self, tmp_path):
        """Test should_checkpoint method."""
        manager = CheckpointManager(str(tmp_path), auto_save_interval=0)

        def get_state():
            return {}

        ctx = CheckpointContext(manager, "test", get_state, {})
        assert ctx.should_checkpoint() is True

    def test_should_checkpoint_not_yet(self, tmp_path):
        """Test should_checkpoint when not enough time passed."""
        manager = CheckpointManager(str(tmp_path), auto_save_interval=3600)
        manager._last_save_time = time.time()

        def get_state():
            return {}

        ctx = CheckpointContext(manager, "test", get_state, {})
        assert ctx.should_checkpoint() is False

    def test_save(self, tmp_path):
        """Test save method."""
        manager = CheckpointManager(str(tmp_path))

        def get_state():
            return {"data": "test"}

        ctx = CheckpointContext(manager, "test", get_state, {})
        cp = ctx.save()

        assert cp is not None
        assert cp.state == {"data": "test"}

    def test_get_resumed_state_none(self, tmp_path):
        """Test get_resumed_state with no checkpoint."""
        manager = CheckpointManager(str(tmp_path))

        def get_state():
            return {}

        ctx = CheckpointContext(manager, "test", get_state, {})
        assert ctx.get_resumed_state() is None


class TestRecoveryManager:
    """Tests for RecoveryManager class."""

    def test_init(self):
        """Test initialization."""
        rm = RecoveryManager(max_retries=5, base_delay=2.0)
        assert rm.max_retries == 5
        assert rm.base_delay == 2.0
        assert rm.max_delay == 60.0
        assert rm.exponential_base == 2.0

    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self):
        """Test successful execution."""
        rm = RecoveryManager()

        async def success_fn():
            return "success"

        result = await rm.execute_with_retry("op1", success_fn)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_execute_with_retry_eventual_success(self):
        """Test eventual success after failures."""
        rm = RecoveryManager(max_retries=3, base_delay=0.01)
        call_count = 0

        async def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("fail")
            return "success"

        result = await rm.execute_with_retry("op1", flaky_fn)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_execute_with_retry_exhausted(self):
        """Test all retries exhausted."""
        rm = RecoveryManager(max_retries=2, base_delay=0.01)

        async def always_fail():
            raise RuntimeError("always fails")

        with pytest.raises(RuntimeError):
            await rm.execute_with_retry("op1", always_fail)

    @pytest.mark.asyncio
    async def test_execute_with_retry_delay_increases(self):
        """Test delay increases with retries."""
        rm = RecoveryManager(max_retries=3, base_delay=0.01, max_delay=1.0)
        call_count = 0

        async def flaky_fn():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fail")

        start = time.time()
        with pytest.raises(RuntimeError):
            await rm.execute_with_retry("op1", flaky_fn)
        elapsed = time.time() - start

        # Should have waited at least a bit
        assert elapsed > 0.01

    def test_get_retry_count(self):
        """Test getting retry count."""
        rm = RecoveryManager()
        assert rm.get_retry_count("op1") == 0

    def test_reset_retry_count(self):
        """Test resetting retry count."""
        rm = RecoveryManager()
        rm._retry_count["op1"] = 3
        rm.reset_retry_count("op1")
        assert rm.get_retry_count("op1") == 0

    @pytest.mark.asyncio
    async def test_execute_with_retry_resets_count(self):
        """Test retry count resets after success."""
        rm = RecoveryManager(max_retries=3, base_delay=0.01)
        call_count = 0

        async def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("fail")
            return "success"

        await rm.execute_with_retry("op1", flaky_fn)
        assert rm.get_retry_count("op1") == 0
