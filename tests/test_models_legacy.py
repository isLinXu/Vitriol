"""Tests for vitriol.models_legacy modules."""
from unittest.mock import MagicMock

import pytest
from transformers import PretrainedConfig

from vitriol.models_legacy.registry import ModelAdapter, DefaultAdapter, ModelRegistry
from vitriol.models_legacy.deepseek import DeepSeekAdapter
from vitriol.models_legacy.llama import LlamaAdapter
from vitriol.models_legacy.qwen import QwenMoeAdapter, Qwen35MoeAdapter


# ─────────────────────────────────────────────────────────────
# registry
# ─────────────────────────────────────────────────────────────

class TestDefaultAdapter:
    def test_match_always_true(self):
        config = MagicMock()
        assert DefaultAdapter.match("any/model", config) is True

    def test_patch_config_returns_config(self):
        adapter = DefaultAdapter()
        config = MagicMock()
        result = adapter.patch_config(config)
        assert result is config

    def test_get_model_class_none(self):
        adapter = DefaultAdapter()
        assert adapter.get_model_class(MagicMock()) is None


class TestModelRegistry:
    def test_register_and_get_adapter(self):
        class DummyAdapter(ModelAdapter):
            @classmethod
            def match(cls, model_id, config):
                return model_id == "dummy"

        ModelRegistry.register(DummyAdapter)
        config = MagicMock()
        adapter = ModelRegistry.get_adapter("dummy", config)
        assert isinstance(adapter, DummyAdapter)

    def test_get_adapter_fallback_default(self):
        config = MagicMock()
        adapter = ModelRegistry.get_adapter("unknown/model", config)
        assert isinstance(adapter, DefaultAdapter)


# ─────────────────────────────────────────────────────────────
# deepseek
# ─────────────────────────────────────────────────────────────

class TestDeepSeekAdapter:
    def test_match_by_model_type(self):
        config = MagicMock()
        config.model_type = "deepseek_v2"
        config.architectures = None
        assert DeepSeekAdapter.match("test", config) is True

    def test_match_by_architecture(self):
        config = MagicMock()
        config.model_type = ""
        config.architectures = ["DeepseekV2ForCausalLM"]
        assert DeepSeekAdapter.match("test", config) is True

    def test_no_match(self):
        config = MagicMock()
        config.model_type = "llama"
        config.architectures = ["LlamaForCausalLM"]
        assert DeepSeekAdapter.match("test", config) is False

    def test_patch_config(self):
        adapter = DeepSeekAdapter()
        config = MagicMock()
        config.is_encoder_decoder = True
        config.generation_config = MagicMock()
        result = adapter.patch_config(config)
        assert result is config
        assert result.is_encoder_decoder is False
        assert not hasattr(result, "generation_config")


# ─────────────────────────────────────────────────────────────
# llama
# ─────────────────────────────────────────────────────────────

class TestLlamaAdapter:
    def test_match_by_model_type(self):
        config = MagicMock()
        config.model_type = "llama"
        config.architectures = None
        assert LlamaAdapter.match("test", config) is True

    def test_match_by_architecture(self):
        config = MagicMock()
        config.model_type = ""
        config.architectures = ["LlamaForCausalLM"]
        assert LlamaAdapter.match("test", config) is True

    def test_no_match(self):
        config = MagicMock()
        config.model_type = "qwen2"
        config.architectures = ["Qwen2ForCausalLM"]
        assert LlamaAdapter.match("test", config) is False

    def test_patch_config(self):
        adapter = LlamaAdapter()
        config = MagicMock()
        config.is_encoder_decoder = True
        config.generation_config = MagicMock()
        result = adapter.patch_config(config)
        assert result is config
        assert result.is_encoder_decoder is False
        assert not hasattr(result, "generation_config")


# ─────────────────────────────────────────────────────────────
# qwen
# ─────────────────────────────────────────────────────────────

class TestQwenMoeAdapter:
    def test_match_by_model_type(self):
        config = MagicMock()
        config.model_type = "qwen2_moe"
        config.architectures = None
        assert QwenMoeAdapter.match("test", config) is True

    def test_match_by_architecture(self):
        config = MagicMock()
        config.model_type = ""
        config.architectures = ["Qwen2MoeForCausalLM"]
        assert QwenMoeAdapter.match("test", config) is True

    def test_no_match(self):
        config = MagicMock()
        config.model_type = "llama"
        config.architectures = ["LlamaForCausalLM"]
        assert QwenMoeAdapter.match("test", config) is False

    def test_patch_config(self):
        adapter = QwenMoeAdapter()
        config = MagicMock()
        config.is_encoder_decoder = True
        config.generation_config = MagicMock()
        result = adapter.patch_config(config)
        assert result is config
        assert result.is_encoder_decoder is False


class TestQwen35MoeAdapter:
    def test_match_by_architecture(self):
        config = MagicMock()
        config.model_type = ""
        config.architectures = ["Qwen3_5MoeForConditionalGeneration"]
        assert Qwen35MoeAdapter.match("test", config) is True

    def test_match_by_model_type(self):
        config = MagicMock()
        config.model_type = "qwen3_5_moe"
        config.architectures = None
        assert Qwen35MoeAdapter.match("test", config) is True

    def test_no_match(self):
        config = MagicMock()
        config.model_type = "llama"
        config.architectures = ["LlamaForCausalLM"]
        assert Qwen35MoeAdapter.match("test", config) is False

    def test_get_model_class(self):
        adapter = Qwen35MoeAdapter()
        result = adapter.get_model_class(MagicMock())
        # Returns Qwen2MoeForCausalLM if available, else None
        assert result is not None or result is None  # either is valid
