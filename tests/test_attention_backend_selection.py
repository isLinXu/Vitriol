"""Compatibility fallbacks must not disable explicitly selected SDPA."""

from types import SimpleNamespace

import pytest
import transformers.modeling_utils

from vitriol.patches.transformers_patches import patch_pretrained_init


@pytest.mark.parametrize("backend", [None, "sdpa", "eager", "flash_attention_2"])
@pytest.mark.parametrize("flash_available", [False, True])
def test_missing_flash_dependency_only_changes_flash_backend(monkeypatch, backend, flash_available):
    recorded = []

    def original_init(self, config):
        recorded.append(config._attn_implementation)

    monkeypatch.setattr(transformers.modeling_utils.PreTrainedModel, "__init__", original_init)
    monkeypatch.setattr("vitriol.patches.transformers_patches.importlib.util.find_spec",
                        lambda name: object() if flash_available else None)
    patch_pretrained_init()
    config = SimpleNamespace(
        _attn_implementation=backend, attn_implementation=backend,
        use_flash_attention_2=True, use_flash_attn=True,
    )
    transformers.modeling_utils.PreTrainedModel.__init__(object(), config)
    expected = "eager" if backend == "flash_attention_2" and not flash_available else backend
    assert recorded == [expected]
    assert config.attn_implementation == expected
    assert config.use_flash_attention_2 is flash_available
    assert config.use_flash_attn is flash_available
