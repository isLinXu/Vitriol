"""Regression tests for ExoBrain KV cache fusion boundaries."""

from types import SimpleNamespace

import pytest
import torch

from vitriol.kv.exobrain.pipeline import ExoBrainInferencePipeline


class _BrainBus:
    stats = {}

    def __init__(self, teacher_key, teacher_value):
        self.teacher_pair = (teacher_key, teacher_value)

    def retrieve(self, query, layer_idx):
        return self.teacher_pair


def _make_pipeline(fusion_mode, gate_temperature=1.0):
    pipeline = ExoBrainInferencePipeline(
        shell_model_path="unused",
        fusion_mode=fusion_mode,
        residual_alpha=0.5,
        gate_temperature=gate_temperature,
    )
    shell_key = torch.arange(8, dtype=torch.float32).reshape(1, 1, 4, 2)
    shell_value = shell_key + 10
    teacher_key = torch.full((1, 1, 2, 2), 20.0)
    teacher_value = torch.full((1, 1, 2, 2), 30.0)
    pipeline._brain_bus = _BrainBus(teacher_key, teacher_value)
    return pipeline, shell_key, shell_value, teacher_key, teacher_value


@pytest.mark.parametrize("fusion_mode", ["replace", "residual", "gated"])
def test_fusion_preserves_shell_cache_length_and_unmatched_suffix(fusion_mode):
    pipeline, shell_key, shell_value, teacher_key, teacher_value = _make_pipeline(fusion_mode)

    result = pipeline._inject_teacher_kv_into_cache(((shell_key, shell_value),))

    fused_key, fused_value = result[0]
    assert fused_key.shape == shell_key.shape
    assert fused_value.shape == shell_value.shape
    assert torch.equal(fused_key[:, :, 2:, :], shell_key[:, :, 2:, :])
    assert torch.equal(fused_value[:, :, 2:, :], shell_value[:, :, 2:, :])
    if fusion_mode == "replace":
        assert torch.equal(fused_key[:, :, :2, :], teacher_key)
        assert torch.equal(fused_value[:, :, :2, :], teacher_value)


def test_cache_injection_preserves_uninitialized_legacy_layers():
    pipeline, shell_key, shell_value, teacher_key, _ = _make_pipeline("replace")
    cache = ((shell_key, shell_value), None, (shell_key, shell_value))

    result = pipeline._inject_teacher_kv_into_cache(cache)

    assert result[1] is None
    expected_key = torch.cat([teacher_key, shell_key[:, :, 2:, :]], dim=2)
    assert torch.equal(result[0][0], expected_key)
    assert torch.equal(result[2][0], expected_key)


def test_dynamic_cache_injection_keeps_cache_type_and_sequence_length():
    from transformers.cache_utils import DynamicCache

    pipeline, shell_key, shell_value, teacher_key, _ = _make_pipeline("replace")
    cache = DynamicCache()
    cache.update(shell_key, shell_value, 0)

    result = pipeline._inject_teacher_kv_into_cache(cache)

    assert isinstance(result, DynamicCache)
    fused_key, fused_value = result[0]
    assert fused_key.shape == shell_key.shape
    assert fused_value.shape == shell_value.shape
    assert torch.equal(fused_key, torch.cat([teacher_key, shell_key[:, :, 2:, :]], dim=2))


def test_dynamic_cache_rebuild_preserves_sliding_layer_types():
    from transformers import PretrainedConfig
    from transformers.cache_utils import DynamicCache

    config = PretrainedConfig(num_hidden_layers=3, sliding_window=4)
    pipeline, shell_key, shell_value, _, _ = _make_pipeline("replace")
    pipeline._shell_model = SimpleNamespace(config=config)
    cache = DynamicCache(config=config)
    for layer_idx in range(config.num_hidden_layers):
        cache.update(shell_key + layer_idx, shell_value + layer_idx, layer_idx)

    result = pipeline._inject_teacher_kv_into_cache(cache)

    assert [type(layer) for layer in result.layers] == [type(layer) for layer in cache.layers]
    assert [layer.sliding_window for layer in result.layers] == [4, 4, 4]


def test_incompatible_teacher_head_count_falls_back_to_shell_layer():
    pipeline, shell_key, shell_value, _, _ = _make_pipeline("replace")
    pipeline._brain_bus = _BrainBus(
        torch.ones(1, 2, 2, 2),
        torch.ones(1, 2, 2, 2),
    )

    result = pipeline._inject_teacher_kv_into_cache(((shell_key, shell_value),))

    assert torch.equal(result[0][0], shell_key)
    assert torch.equal(result[0][1], shell_value)


def test_gated_fusion_uses_configured_temperature():
    cold_pipeline, shell_key, shell_value, _, _ = _make_pipeline("gated", gate_temperature=0.1)
    warm_pipeline, _, _, _, _ = _make_pipeline("gated", gate_temperature=100.0)
    cold_key, cold_value = cold_pipeline._inject_teacher_kv_into_cache(((shell_key, shell_value),))[0]
    warm_key, warm_value = warm_pipeline._inject_teacher_kv_into_cache(((shell_key, shell_value),))[0]

    assert not torch.allclose(cold_key[:, :, :2, :], warm_key[:, :, :2, :])
    assert not torch.allclose(cold_value[:, :, :2, :], warm_value[:, :, :2, :])


@pytest.mark.parametrize("temperature", [0.0, float("nan")])
def test_gate_temperature_must_be_positive_and_finite(temperature):
    with pytest.raises(ValueError, match="gate_temperature"):
        ExoBrainInferencePipeline("unused", gate_temperature=temperature)


def test_pipeline_rejects_unknown_fusion_mode():
    with pytest.raises(ValueError, match="fusion_mode"):
        ExoBrainInferencePipeline("unused", fusion_mode="typo")
