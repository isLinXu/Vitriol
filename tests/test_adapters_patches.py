"""Tests for adapter modules and model family patches."""

from unittest.mock import MagicMock

from vitriol.adapters.base import DefaultAdapter
from vitriol.adapters.registry import AdapterRegistry
from vitriol.adapters.llama import LlamaAdapter
from vitriol.adapters.qwen import QwenMoeAdapter, Qwen35MoeAdapter
from vitriol.patches.model_family_patches import (
    PatchRegistry, _set_missing, _ensure_rope_params,
    _fix_rope_theta, _fix_rms_norm
)


# ─────────────────────────────────────────────────────────────────────────────
# base adapter tests
# ─────────────────────────────────────────────────────────────────────────────

class TestModelAdapter:
    def test_default_adapter_match(self):
        config = MagicMock()
        config.model_type = "unknown"
        # DefaultAdapter is the fallback, always matches
        assert DefaultAdapter.match("test/model", config) is True

    def test_default_adapter_patch_config(self):
        adapter = DefaultAdapter()
        config = MagicMock()
        result = adapter.patch_config(config)
        assert result is config

    def test_default_adapter_validate(self):
        adapter = DefaultAdapter()
        config = MagicMock()
        assert adapter.validate_config(config) is True

    def test_default_adapter_get_model_class(self):
        adapter = DefaultAdapter()
        assert adapter.get_model_class(MagicMock()) is None


class TestLlamaAdapter:
    def test_match_llama(self):
        config = MagicMock()
        config.model_type = "llama"
        assert LlamaAdapter.match("test", config) is True

    def test_match_not_llama(self):
        config = MagicMock()
        config.model_type = "qwen"
        assert LlamaAdapter.match("test", config) is False

    def test_patch_config(self):
        adapter = LlamaAdapter()
        config = MagicMock()
        config.is_encoder_decoder = True
        result = adapter.patch_config(config)
        assert result.is_encoder_decoder is False


class TestQwenMoeAdapter:
    def test_match_qwen2_moe(self):
        config = MagicMock()
        config.model_type = "qwen2_moe"
        assert QwenMoeAdapter.match("test", config) is True

    def test_match_architecture(self):
        config = MagicMock()
        config.model_type = "other"
        config.architectures = ["Qwen2MoeForCausalLM"]
        assert QwenMoeAdapter.match("test", config) is True

    def test_match_false(self):
        config = MagicMock()
        config.model_type = "llama"
        config.architectures = []
        assert QwenMoeAdapter.match("test", config) is False

    def test_patch_config(self):
        adapter = QwenMoeAdapter()
        config = MagicMock()
        config.is_encoder_decoder = True
        config.generation_config = MagicMock()
        result = adapter.patch_config(config)
        assert result.is_encoder_decoder is False


class TestQwen35MoeAdapter:
    def test_match(self):
        config = MagicMock()
        config.model_type = "qwen3_5_moe"
        assert Qwen35MoeAdapter.match("test", config) is True

    def test_patch_config_promotes_text_config(self):
        adapter = Qwen35MoeAdapter()
        config = MagicMock()
        config.model_type = "qwen3_5_moe"
        config.text_config = {"hidden_size": 128, "num_hidden_layers": 12}
        config.vision_config = {"image_size": 224}
        config.hidden_size = None

        result = adapter.patch_config(config)
        # text_config fields should be promoted
        assert result.hidden_size == 128


# ─────────────────────────────────────────────────────────────────────────────
# registry tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAdapterRegistry:
    def test_discover_builtin(self):
        adapters = AdapterRegistry.discover_builtin_adapter_metadata()
        assert isinstance(adapters, list)
        assert len(adapters) > 0
        # Should include DefaultAdapter as fallback
        assert any(a.get("is_fallback") for a in adapters)

    def test_discover_has_names(self):
        adapters = AdapterRegistry.discover_builtin_adapter_metadata()
        for a in adapters:
            assert "name" in a
            assert "module" in a

    def test_iter_builtin_modules(self):
        modules = AdapterRegistry._iter_builtin_adapter_modules()
        assert isinstance(modules, list)
        # Should find actual adapter files
        names = [name for name, _ in modules]
        assert "llama" in names or "qwen" in names or len(names) == 0


# ─────────────────────────────────────────────────────────────────────────────
# patches tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSetMissing:
    def test_sets_missing(self):
        obj = MagicMock()
        del obj.new_attr  # Ensure it doesn't exist
        obj.new_attr = None
        _set_missing(obj, new_attr="value")
        assert obj.new_attr == "value"

    def test_skips_existing(self):
        obj = MagicMock()
        obj.existing = "original"
        _set_missing(obj, existing="new")
        assert obj.existing == "original"

    def test_sets_none(self):
        obj = MagicMock()
        obj.none_attr = None
        _set_missing(obj, none_attr="value")
        assert obj.none_attr == "value"


class TestEnsureRopeParams:
    def test_creates_dict(self):
        config = MagicMock()
        config.rope_parameters = None
        _ensure_rope_params(config)
        assert isinstance(config.rope_parameters, dict)
        assert "rope_type" in config.rope_parameters

    def test_merges_defaults(self):
        config = MagicMock()
        config.rope_parameters = {"rope_type": "custom"}
        _ensure_rope_params(config)
        assert config.rope_parameters["rope_type"] == "custom"
        assert "rope_theta" in config.rope_parameters


class TestFixRopeTheta:
    def test_fixes_list(self):
        config = MagicMock()
        config.rope_theta = [50000.0, 10000.0]
        config.rope_scaling = None
        _fix_rope_theta(config)
        assert config.rope_theta == 50000.0

    def test_fixes_none(self):
        config = MagicMock()
        config.rope_theta = None
        config.rope_scaling = None
        _fix_rope_theta(config)
        assert config.rope_theta == 10000.0

    def test_fixes_rope_scaling_list(self):
        config = MagicMock()
        config.rope_theta = 10000.0
        config.rope_scaling = [1.0, 2.0]
        _fix_rope_theta(config)
        assert config.rope_scaling is None


class TestFixRmsNorm:
    def test_fixes_list(self):
        config = MagicMock()
        config.rms_norm_eps = [1e-5, 1e-6]
        _fix_rms_norm(config)
        assert config.rms_norm_eps == 1e-5

    def test_leaves_float(self):
        config = MagicMock()
        config.rms_norm_eps = 1e-6
        _fix_rms_norm(config)
        assert config.rms_norm_eps == 1e-6


class TestPatchRegistry:
    def test_register_decorator(self):
        @PatchRegistry.register("test_family")
        def patch_test(config, model_id):
            config.patched = True

        assert len(PatchRegistry._entries) > 0

    def test_apply_matching(self):
        @PatchRegistry.register("testmodel")
        def patch_testmodel(config, model_id):
            config.patched = True

        config = MagicMock()
        config.__class__.__name__ = "TestModelConfig"
        PatchRegistry.apply(config, "org/testmodel-7b")
        assert config.patched is True

    def test_apply_no_match(self):
        config = MagicMock()
        config.__class__.__name__ = "UnknownConfig"
        # Should not raise
        PatchRegistry.apply(config, "org/unknown")

