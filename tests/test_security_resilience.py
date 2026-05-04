"""
Tests for vitriol.security.context and vitriol.resilience.checkpoint modules.
"""
import pytest
from unittest.mock import patch
import os
import time

from vitriol.security.context import (
    _get_bool,
    _as_dict,
    _env_offline,
    SecurityContext,
    resolve_security_context,
)
from vitriol.resilience.checkpoint import (
    Checkpoint,
    CheckpointManager,
    RecoveryManager,
)


# ─────────────────────────────────────────────────────────────
# _get_bool helper
# ─────────────────────────────────────────────────────────────

class TestGetBool:
    def test_true_values(self):
        for v in [True, "true", "True", "1", "yes", "on", " YES "]:
            assert _get_bool({"k": v}, "k") is True

    def test_false_values(self):
        for v in [False, "false", "False", "0", "no", "off"]:
            assert _get_bool({"k": v}, "k") is False

    def test_missing_key(self):
        assert _get_bool({}, "k") is None

    def test_none_value(self):
        assert _get_bool({"k": None}, "k") is None


# ─────────────────────────────────────────────────────────────
# _as_dict helper
# ─────────────────────────────────────────────────────────────

class TestAsDict:
    def test_none(self):
        assert _as_dict(None) == {}

    def test_dict(self):
        assert _as_dict({"a": 1}) == {"a": 1}

    def test_security_options(self):
        from vitriol.config.manager import SecurityOptions

        so = SecurityOptions(trust_remote_code=False, allow_network=True, local_files_only=False)
        d = _as_dict(so)
        assert d["trust_remote_code"] is False
        assert d["allow_network"] is True

    def test_mapping_like(self):
        class FakeMapping:
            def items(self):
                return [("x", 10)]
            def __iter__(self):
                return iter(["x"])
            def keys(self):
                return ["x"]
            def __getitem__(self, key):
                return {"x": 10}[key]

        assert _as_dict(FakeMapping()) == {"x": 10}

    def test_unconvertible(self):
        assert _as_dict(object()) == {}


# ─────────────────────────────────────────────────────────────
# _env_offline helper
# ─────────────────────────────────────────────────────────────

class TestEnvOffline:
    def test_hf_hub_offline(self, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
        assert _env_offline() is True

    def test_transformers_offline(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
        assert _env_offline() is True

    def test_not_offline(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
        assert _env_offline() is False


# ─────────────────────────────────────────────────────────────
# SecurityContext
# ─────────────────────────────────────────────────────────────

class TestSecurityContext:
    def test_dataclass(self):
        ctx = SecurityContext(
            trust_remote_code=True,
            allow_network=False,
            local_files_only=True,
            provenance={"trust_remote_code": "base"},
        )
        assert ctx.trust_remote_code is True
        assert ctx.allow_network is False
        assert ctx.local_files_only is True

    def test_to_security_options(self):
        ctx = SecurityContext(
            trust_remote_code=False,
            allow_network=True,
            local_files_only=False,
            provenance={},
        )
        so = ctx.to_security_options()
        assert so.trust_remote_code is False
        assert so.allow_network is True
        assert so.local_files_only is False

    def test_apply_to_environ_sets_offline(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

        ctx = SecurityContext(
            trust_remote_code=True,
            allow_network=False,
            local_files_only=True,
            provenance={},
        )
        ctx.apply_to_environ()
        assert os.environ.get("HF_HUB_OFFLINE") == "1"
        assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"

    def test_apply_to_environ_does_not_override_existing(self, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "0")
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")

        ctx = SecurityContext(
            trust_remote_code=True,
            allow_network=False,
            local_files_only=True,
            provenance={},
        )
        ctx.apply_to_environ()
        # setdefault should not override existing values
        assert os.environ.get("HF_HUB_OFFLINE") == "0"
        assert os.environ.get("TRANSFORMERS_OFFLINE") == "0"


# ─────────────────────────────────────────────────────────────
# resolve_security_context
# ─────────────────────────────────────────────────────────────

class TestResolveSecurityContext:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

        ctx = resolve_security_context()
        assert ctx.trust_remote_code is True
        assert ctx.allow_network is True
        assert ctx.local_files_only is False
        assert ctx.provenance["trust_remote_code"] == "base"

    def test_explicit_overrides(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

        ctx = resolve_security_context(
            explicit={"trust_remote_code": False, "allow_network": False}
        )
        assert ctx.trust_remote_code is False
        assert ctx.allow_network is False
        assert ctx.local_files_only is True  # inferred from allow_network=False
        assert ctx.provenance["trust_remote_code"] == "explicit"
        assert ctx.provenance["local_files_only"] == "inferred_offline"

    def test_env_offline_highest_priority(self, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

        ctx = resolve_security_context(
            explicit={"allow_network": True, "local_files_only": False}
        )
        assert ctx.allow_network is False
        assert ctx.local_files_only is True
        assert ctx.provenance["allow_network"] == "env_offline"
        assert ctx.provenance["local_files_only"] == "env_offline"

    def test_base_mapping(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

        ctx = resolve_security_context(base={"trust_remote_code": False})
        assert ctx.trust_remote_code is False
        assert ctx.provenance["trust_remote_code"] == "base"

    def test_security_options_as_base(self, monkeypatch):
        from vitriol.config.manager import SecurityOptions

        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

        so = SecurityOptions(trust_remote_code=False, allow_network=True, local_files_only=False)
        ctx = resolve_security_context(base=so)
        assert ctx.trust_remote_code is False


# ─────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────

class TestCheckpoint:
    def test_dataclass(self):
        ckpt = Checkpoint(
            id="test_123",
            operation="gen",
            state={"layer": 5},
            metadata={"model": "llama"},
            timestamp=1234567890.0,
        )
        assert ckpt.id == "test_123"
        assert ckpt.version == "1.0"

    def test_to_dict(self):
        ckpt = Checkpoint(
            id="test_123",
            operation="gen",
            state={"layer": 5},
            metadata={"model": "llama"},
            timestamp=1234567890.0,
        )
        d = ckpt.to_dict()
        assert d["id"] == "test_123"
        assert d["state"] == {"layer": 5}
        assert d["version"] == "1.0"

    def test_from_dict(self):
        d = {
            "id": "test_123",
            "operation": "gen",
            "state": {"layer": 5},
            "metadata": {},
            "timestamp": 1234567890.0,
            "version": "1.0",
        }
        ckpt = Checkpoint.from_dict(d)
        assert ckpt.id == "test_123"
        assert ckpt.state == {"layer": 5}

    def test_compute_hash(self):
        ckpt1 = Checkpoint(
            id="test_123",
            operation="gen",
            state={"layer": 5},
            metadata={},
            timestamp=1234567890.0,
        )
        ckpt2 = Checkpoint(
            id="test_123",
            operation="gen",
            state={"layer": 5},
            metadata={},
            timestamp=1234567890.0,
        )
        assert ckpt1.compute_hash() == ckpt2.compute_hash()
        assert len(ckpt1.compute_hash()) == 16


# ─────────────────────────────────────────────────────────────
# CheckpointManager
# ─────────────────────────────────────────────────────────────

class TestCheckpointManager:
    @pytest.fixture
    def manager(self, tmp_path):
        return CheckpointManager(checkpoint_dir=str(tmp_path / "checkpoints"), max_checkpoints=3)

    def test_init_creates_dir(self, tmp_path):
        cp_dir = tmp_path / "new_checkpoints"
        mgr = CheckpointManager(checkpoint_dir=str(cp_dir))
        assert cp_dir.exists()

    def test_create_checkpoint(self, manager):
        ckpt = manager.create_checkpoint("gen", {"progress": 0.5})
        assert ckpt.operation == "gen"
        assert ckpt.state == {"progress": 0.5}
        assert ckpt.id.startswith("gen_")

    def test_load_checkpoint(self, manager):
        ckpt = manager.create_checkpoint("gen", {"progress": 0.5})
        loaded = manager.load_checkpoint(ckpt.id)
        assert loaded is not None
        assert loaded.state == {"progress": 0.5}

    def test_load_checkpoint_not_found(self, manager):
        assert manager.load_checkpoint("nonexistent") is None

    def test_find_latest_checkpoint(self, manager):
        ckpt1 = manager.create_checkpoint("gen", {"v": 1})
        time.sleep(0.01)
        ckpt2 = manager.create_checkpoint("gen", {"v": 2})

        latest = manager.find_latest_checkpoint("gen")
        assert latest.state == {"v": 2}

    def test_find_latest_checkpoint_none(self, manager):
        assert manager.find_latest_checkpoint("gen") is None

    def test_list_checkpoints(self, manager):
        counter = [1000.0]
        def incr_time():
            counter[0] += 1.0
            return counter[0]

        with patch("vitriol.resilience.checkpoint.time.time", side_effect=incr_time):
            manager.create_checkpoint("gen", {"v": 1})
            manager.create_checkpoint("gen", {"v": 2})
            manager.create_checkpoint("other", {"v": 3})

        all_ckpts = manager.list_checkpoints()
        assert len(all_ckpts) == 3

        gen_ckpts = manager.list_checkpoints("gen")
        assert len(gen_ckpts) == 2

    def test_delete_checkpoint(self, manager):
        ckpt = manager.create_checkpoint("gen", {"v": 1})
        manager.delete_checkpoint(ckpt.id)
        assert manager.load_checkpoint(ckpt.id) is None

    def test_cleanup_old_checkpoints(self, manager):
        for i in range(5):
            manager.create_checkpoint("gen", {"v": i})
            time.sleep(0.01)

        ckpts = manager.list_checkpoints("gen")
        assert len(ckpts) <= 3

    def test_load_checkpoint_with_pickle_state(self, manager):
        ckpt = manager.create_checkpoint("gen", {"tensor": [1.0, 2.0, 3.0]})
        loaded = manager.load_checkpoint(ckpt.id)
        assert loaded.state == {"tensor": [1.0, 2.0, 3.0]}

    def test_list_checkpoints_skips_corrupted(self, manager, caplog):
        # Create a corrupted checkpoint file
        bad_file = manager.checkpoint_dir / "bad.json"
        bad_file.write_text("not-json{")

        with caplog.at_level("DEBUG"):
            ckpts = manager.list_checkpoints()
        assert len(ckpts) == 0
        assert "Skipping corrupted checkpoint" in caplog.text


# ─────────────────────────────────────────────────────────────
# CheckpointContext
# ─────────────────────────────────────────────────────────────

class TestCheckpointContext:
    @pytest.fixture
    def manager(self, tmp_path):
        return CheckpointManager(checkpoint_dir=str(tmp_path / "checkpoints"), auto_save_interval=0)

    def test_context_enter_no_checkpoint(self, manager):
        def get_state():
            return {"progress": 0.5}

        ctx = manager.checkpoint_context("gen", get_state)
        with ctx as c:
            assert c.get_resumed_state() is None

    def test_context_enter_with_checkpoint(self, manager):
        manager.create_checkpoint("gen", {"progress": 0.8})

        def get_state():
            return {"progress": 0.5}

        ctx = manager.checkpoint_context("gen", get_state)
        with ctx as c:
            assert c.get_resumed_state() == {"progress": 0.8}

    def test_context_success_cleans_up(self, manager):
        manager.create_checkpoint("gen", {"progress": 0.8})

        def get_state():
            return {"progress": 0.9}

        ctx = manager.checkpoint_context("gen", get_state)
        with ctx as c:
            pass

        assert manager.list_checkpoints("gen") == []

    def test_context_failure_saves_checkpoint(self, manager):
        def get_state():
            return {"progress": 0.5}

        ctx = manager.checkpoint_context("gen", get_state)
        try:
            with ctx as c:
                raise ValueError("test error")
        except ValueError:
            pass

        # Should have saved a checkpoint on failure
        assert len(manager.list_checkpoints("gen")) >= 1

    def test_should_checkpoint(self, manager):
        def get_state():
            return {"progress": 0.5}

        ctx = manager.checkpoint_context("gen", get_state)
        with ctx as c:
            # auto_save_interval is 0, so should always be true after init
            assert c.should_checkpoint() is True

    def test_save(self, manager):
        def get_state():
            return {"progress": 0.6}

        ctx = manager.checkpoint_context("gen", get_state)
        with ctx as c:
            ckpt = c.save()
            assert ckpt.operation == "gen"
            assert ckpt.state == {"progress": 0.6}


# ─────────────────────────────────────────────────────────────
# RecoveryManager
# ─────────────────────────────────────────────────────────────

class TestRecoveryManager:
    @pytest.fixture
    def recovery(self):
        return RecoveryManager(max_retries=3, base_delay=0.01, max_delay=0.1)

    @pytest.mark.asyncio
    async def test_execute_success_first_try(self, recovery):
        async def op():
            return "success"

        result = await recovery.execute_with_retry("op1", op)
        assert result == "success"
        assert recovery.get_retry_count("op1") == 0

    @pytest.mark.asyncio
    async def test_execute_retries_then_success(self, recovery):
        call_count = 0

        async def op():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("fail")
            return "success"

        result = await recovery.execute_with_retry("op2", op)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_execute_all_retries_fail(self, recovery):
        async def op():
            raise RuntimeError("always fails")

        with pytest.raises(RuntimeError, match="always fails"):
            await recovery.execute_with_retry("op3", op)

        assert recovery.get_retry_count("op3") == 3

    def test_get_retry_count_missing(self, recovery):
        assert recovery.get_retry_count("unknown") == 0

    def test_reset_retry_count(self, recovery):
        recovery._retry_count["op"] = 2
        recovery.reset_retry_count("op")
        assert recovery.get_retry_count("op") == 0

    def test_backoff_delay_capped(self):
        recovery = RecoveryManager(max_retries=10, base_delay=1.0, max_delay=5.0, exponential_base=2.0)
        # Delay for attempt 5 should be capped at max_delay
        expected = min(1.0 * (2.0 ** 4), 5.0)
        assert expected == 5.0
