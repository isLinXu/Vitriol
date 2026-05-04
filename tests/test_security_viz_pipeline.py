"""Tests for security, visualization utils, and core pipeline modules."""

import os
from unittest.mock import MagicMock, patch

from vitriol.security.context import (
    SecurityContext, _get_bool, _as_dict, _env_offline
)
from vitriol.visualization.utils import load_weights
from vitriol.core.pipeline.pipeline import GenerationPipeline
from vitriol.core.pipeline.context import GenerationContext


# ─────────────────────────────────────────────────────────────────────────────
# security context tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGetBool:
    def test_missing_key(self):
        assert _get_bool({}, "key") is None

    def test_none_value(self):
        assert _get_bool({"key": None}, "key") is None

    def test_true_bool(self):
        assert _get_bool({"key": True}, "key") is True

    def test_false_bool(self):
        assert _get_bool({"key": False}, "key") is False

    def test_string_true(self):
        assert _get_bool({"key": "true"}, "key") is True
        assert _get_bool({"key": "yes"}, "key") is True
        assert _get_bool({"key": "1"}, "key") is True
        assert _get_bool({"key": "on"}, "key") is True

    def test_string_false(self):
        assert _get_bool({"key": "false"}, "key") is False
        assert _get_bool({"key": "no"}, "key") is False
        assert _get_bool({"key": "0"}, "key") is False
        assert _get_bool({"key": "off"}, "key") is False


class TestAsDict:
    def test_none(self):
        assert _as_dict(None) == {}

    def test_dict(self):
        assert _as_dict({"a": 1}) == {"a": 1}

    def test_security_options(self):
        from vitriol.config.manager import SecurityOptions
        so = SecurityOptions(trust_remote_code=True, allow_network=False, local_files_only=True)
        d = _as_dict(so)
        assert d["trust_remote_code"] is True
        assert d["allow_network"] is False

    def test_unconvertible(self):
        assert _as_dict(object()) == {}


class TestEnvOffline:
    def test_no_env(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _env_offline() is False

    def test_hf_hub_offline(self):
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
            assert _env_offline() is True

    def test_transformers_offline(self):
        with patch.dict(os.environ, {"TRANSFORMERS_OFFLINE": "1"}):
            assert _env_offline() is True


class TestSecurityContext:
    def test_creation(self):
        ctx = SecurityContext(
            trust_remote_code=True,
            allow_network=False,
            local_files_only=True,
            provenance={"source": "test"},
        )
        assert ctx.trust_remote_code is True
        assert ctx.allow_network is False
        assert ctx.provenance["source"] == "test"

    def test_to_security_options(self):
        ctx = SecurityContext(
            trust_remote_code=True,
            allow_network=True,
            local_files_only=False,
            provenance={},
        )
        so = ctx.to_security_options()
        assert so.trust_remote_code is True
        assert so.allow_network is True

    def test_apply_to_environ_disables_network(self):
        ctx = SecurityContext(
            trust_remote_code=False,
            allow_network=False,
            local_files_only=True,
            provenance={},
        )
        with patch.dict(os.environ, {}, clear=True):
            ctx.apply_to_environ()
            assert os.environ.get("HF_HUB_OFFLINE") == "1"
            assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"

    def test_apply_to_environ_allows_network(self):
        ctx = SecurityContext(
            trust_remote_code=False,
            allow_network=True,
            local_files_only=False,
            provenance={},
        )
        with patch.dict(os.environ, {}, clear=True):
            ctx.apply_to_environ()
            assert "HF_HUB_OFFLINE" not in os.environ


# ─────────────────────────────────────────────────────────────────────────────
# visualization utils tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadWeights:
    def test_nonexistent_dir(self):
        weights = load_weights("/nonexistent/path")
        assert weights == {}

    def test_empty_dir(self, tmp_path):
        weights = load_weights(str(tmp_path))
        assert weights == {}

    def test_with_pattern(self, tmp_path):
        # Create a mock safetensors file
        import torch
        from safetensors.torch import save_file
        data = {"layer1.weight": torch.randn(10, 10), "layer2.weight": torch.randn(5, 5)}
        save_file(data, tmp_path / "model.safetensors")

        weights = load_weights(str(tmp_path), pattern="layer1")
        assert "layer1.weight" in weights
        assert "layer2.weight" not in weights

    def test_with_limit(self, tmp_path):
        import torch
        from safetensors.torch import save_file
        data = {"a": torch.randn(2, 2), "b": torch.randn(2, 2), "c": torch.randn(2, 2)}
        save_file(data, tmp_path / "model.safetensors")

        weights = load_weights(str(tmp_path), limit=2)
        assert len(weights) == 2


# ─────────────────────────────────────────────────────────────────────────────
# core pipeline tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerationContext:
    def test_creation(self):
        from vitriol.config.manager import GenerationConfig
        cfg = GenerationConfig()
        ctx = GenerationContext(model_id="test", output_dir="/tmp", config=cfg)
        assert ctx.model_id == "test"
        assert ctx.output_dir == "/tmp"

    def test_defaults(self):
        from vitriol.config.manager import GenerationConfig
        cfg = GenerationConfig()
        ctx = GenerationContext(model_id="test", output_dir="/tmp", config=cfg)
        assert ctx.generator is None
        assert ctx.shrink_config is None


class TestGenerationPipeline:
    def test_init(self):
        pipeline = GenerationPipeline(steps=[])
        assert pipeline.steps == []

    def test_run_single_step(self):
        step = MagicMock()
        step.name = "test_step"
        pipeline = GenerationPipeline(steps=[step])
        from vitriol.config.manager import GenerationConfig
        ctx = GenerationContext(model_id="test", output_dir="/tmp", config=GenerationConfig())
        pipeline.run(ctx)
        step.run.assert_called_once_with(ctx)

    def test_run_multiple_steps(self):
        steps = [MagicMock() for _ in range(3)]
        for i, s in enumerate(steps):
            s.name = f"step_{i}"
        pipeline = GenerationPipeline(steps=steps)
        from vitriol.config.manager import GenerationConfig
        ctx = GenerationContext(model_id="test", output_dir="/tmp", config=GenerationConfig())
        pipeline.run(ctx)
        for s in steps:
            s.run.assert_called_once_with(ctx)

    def test_run_empty(self):
        pipeline = GenerationPipeline(steps=[])
        from vitriol.config.manager import GenerationConfig
        ctx = GenerationContext(model_id="test", output_dir="/tmp", config=GenerationConfig())
        pipeline.run(ctx)  # Should not raise

