"""Offline regression tests for persisted shard integrity and interrupted runs."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import load_file
from transformers import PretrainedConfig

from vitriol.config.manager import GenerationConfig, SecurityOptions
from vitriol.core.generator import MinimalWeightGenerator


def make_generator(tmp_path, monkeypatch, strategy="random", indexed=True):
    config = GenerationConfig(strategy=strategy, max_shard_size="16", security=SecurityOptions(allow_network=False))
    generator = MinimalWeightGenerator("local-test", str(tmp_path), config=config, shrink_config=False)
    model = torch.nn.Module()
    for name in ("first", "second", "third", "fourth", "fifth"):
        model.register_parameter(name, torch.nn.Parameter(torch.empty(4)))
    model.config = PretrainedConfig()
    model.tie_weights = lambda: None
    monkeypatch.setattr(generator, "_load_hf_config", lambda: model.config)
    monkeypatch.setattr(generator, "_initialize_model", lambda *a, **kw: model)
    monkeypatch.setattr("vitriol.core.generator.AdapterRegistry.get_adapter", lambda *a: SimpleNamespace(patch_config=lambda c: c))
    shard_map = {name: "model-00001-of-00001.safetensors" for name, _ in model.named_parameters()} if indexed else {}
    monkeypatch.setattr(generator, "_get_original_shard_map", lambda: shard_map)
    for name in ("_save_configs", "_copy_custom_code_files", "_save_tokenizer", "_add_readme_metadata", "_write_manifest"):
        monkeypatch.setattr(generator, name, lambda *a: None)
    return generator, model


def read_weights(tmp_path):
    index_path = next(tmp_path.glob("*.index.json"))
    index = json.loads(index_path.read_text())
    result = {}
    for filename in set(index["weight_map"].values()):
        path = tmp_path / filename
        tensors = load_file(str(path)) if path.suffix == ".safetensors" else torch.load(path, weights_only=True)
        for name, target in index["weight_map"].items():
            if target == filename:
                assert name in tensors, f"Index references missing tensor {name} in {filename}"
                result[name] = tensors[name]
    return index, result


@pytest.mark.parametrize("strategy", ["random", "ultra"])
def test_original_shard_retains_all_tensors_across_flush_threshold(tmp_path, monkeypatch, strategy):
    generator, model = make_generator(tmp_path, monkeypatch, strategy)
    # Force several flushes plus a residual tensor in the previous implementation.
    generator.max_shard_size = 4 if strategy == "ultra" else 16
    generator.generate()
    index, tensors = read_weights(tmp_path)
    assert set(tensors) == dict(model.named_parameters()).keys()
    assert len(set(index["weight_map"].values())) == 1
    assert all(tuple(tensors[n].shape) == tuple(p.shape) for n, p in model.named_parameters())


def test_without_source_index_respects_shard_byte_limit(tmp_path, monkeypatch):
    generator, model = make_generator(tmp_path, monkeypatch, indexed=False)
    generator.generate()
    index, tensors = read_weights(tmp_path)
    assert set(tensors) == dict(model.named_parameters()).keys()
    assert len(set(index["weight_map"].values())) == 3
    assert index["metadata"]["total_size"] == 40
    for filename in set(index["weight_map"].values()):
        assert sum(t.numel() * t.element_size() for t in load_file(str(tmp_path / filename)).values()) <= 16


def test_resume_keeps_completed_shards_and_generates_only_missing_tensors(tmp_path, monkeypatch):
    generator, model = make_generator(tmp_path, monkeypatch, indexed=False)
    original_generate = generator.strategy.generate_tensor
    def fail_on_fourth(shape, dtype, name):
        if name == "fourth":
            raise RuntimeError("injected interruption")
        return original_generate(shape, dtype, name)
    monkeypatch.setattr(generator.strategy, "generate_tensor", fail_on_fourth)
    with pytest.raises(RuntimeError, match="injected interruption"):
        generator.generate()
    checkpoint = generator.incremental.load_checkpoint()
    assert set(checkpoint["shard_map"]) == {"first", "second"}
    saved_file = tmp_path / checkpoint["shard_map"]["first"]
    saved_bytes = saved_file.read_bytes()
    resumed, _ = make_generator(tmp_path, monkeypatch, indexed=False)
    spy = MagicMock(wraps=resumed.strategy.generate_tensor)
    monkeypatch.setattr(resumed.strategy, "generate_tensor", spy)
    resumed.generate()
    index, tensors = read_weights(tmp_path)
    assert set(tensors) == dict(model.named_parameters()).keys()
    assert {call.args[2] for call in spy.call_args_list} == {"third", "fourth", "fifth"}
    assert (tmp_path / index["weight_map"]["first"]).read_bytes() == saved_bytes
    assert index["metadata"]["total_size"] == 40
    assert resumed.incremental.load_checkpoint() is None


def test_stale_checkpoint_never_suppresses_tensor_generation(tmp_path, monkeypatch):
    generator, model = make_generator(tmp_path, monkeypatch)
    generator.incremental.save_checkpoint({"generated_param_names": ["first", "second"], "shard_map": {"first": "missing.safetensors"}})
    generator.generate()
    _, tensors = read_weights(tmp_path)
    assert set(tensors) == dict(model.named_parameters()).keys()


def test_checkpoint_serialization_failure_preserves_last_valid_state(tmp_path):
    from vitriol.core.incremental import IncrementalGenerator
    checkpoint = IncrementalGenerator(str(tmp_path))
    checkpoint.save_checkpoint({"completed": ["first"]})
    checkpoint.save_checkpoint({"completed": object()})
    assert checkpoint.load_checkpoint() == {"completed": ["first"]}


def test_visualization_copy_failure_restores_active_config(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    from vitriol.core import generator_persistence
    module = ModuleType("vitriol.arch_viz.visualizer")
    module.ArchitectureVisualizer = MagicMock()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    active = '{"model_type": "llama", "hidden_size": 16}'
    (tmp_path / "config.json").write_text(active)
    (tmp_path / "meta-config.json").write_text('{"hidden_size": 4096}')
    generator = SimpleNamespace(output_dir=str(tmp_path), shrink_config=True)
    monkeypatch.setattr(generator_persistence.shutil, "copy2", MagicMock(side_effect=OSError("injected copy error")))
    assert generator_persistence.generate_visualization(generator, "unused") is False
    assert (tmp_path / "config.json").read_text() == active


def test_validator_rejects_size_mismatches_without_reinitializing_weights(monkeypatch):
    from vitriol.core import validator as module
    loader = MagicMock(side_effect=RuntimeError("size mismatch for model.weight"))
    fallback = MagicMock()
    monkeypatch.setattr(module, "hf_load_causallm", loader)
    monkeypatch.setattr(module, "hf_load_model", fallback)
    report = module.ModelValidator("local-model").validate(run_inference=False)
    assert report.success is False
    assert report.model_loadable is False
    assert "size mismatch" in report.errors[0]
    assert loader.call_count == 1
    fallback.assert_not_called()


def test_validator_reuse_does_not_retain_previous_failures(monkeypatch):
    from vitriol.core import validator as module
    loader = MagicMock(side_effect=[RuntimeError("bad weights"), torch.nn.Linear(2, 2)])
    monkeypatch.setattr(module, "hf_load_causallm", loader)
    validator = module.ModelValidator("local-model")
    first = validator.validate(run_inference=False)
    second = validator.validate(run_inference=False)
    assert first.success is False
    assert second.success is True
    assert second.errors == []
    assert second.model_loadable is True


def test_source_directory_is_never_cleaned_as_generation_output(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"original model")
    generator = MinimalWeightGenerator(str(tmp_path), str(tmp_path))
    with pytest.raises(ValueError, match="must differ"):
        generator.generate()
    assert (tmp_path / "model.safetensors").read_bytes() == b"original model"


def test_failed_atomic_shard_write_preserves_previous_file(tmp_path, monkeypatch):
    generator, _ = make_generator(tmp_path, monkeypatch)
    filename = "model-00001-of-00001.safetensors"
    path = tmp_path / filename
    path.write_bytes(b"previous valid artifact")
    def fail_save(data, target):
        from pathlib import Path
        Path(target).write_bytes(b"partial artifact")
        raise OSError("injected write failure")
    monkeypatch.setattr(generator.strategy, "save_shard", fail_save)
    with pytest.raises(OSError, match="injected write failure"):
        generator._save_shard({"first": torch.ones(4)}, filename)
    assert path.read_bytes() == b"previous valid artifact"
    assert generator.shard_map == {}
    assert not list(tmp_path.glob(".vitriol-shard-*"))


def test_auto_model_initialization_keeps_offline_security_policy(tmp_path, monkeypatch):
    from vitriol.utils import hf_loading
    generator, model = make_generator(tmp_path, monkeypatch)
    loader = MagicMock(return_value=model)
    monkeypatch.setattr(hf_loading, "load_causallm_from_config", loader)
    assert generator._try_load_auto(model.config) is model
    assert loader.call_args.kwargs["security"]["allow_network"] is False
    assert loader.call_args.kwargs["security"]["trust_remote_code"] is False


@pytest.mark.parametrize("filename", ["../escaped.safetensors", "/tmp/escaped.safetensors"])
def test_shard_path_cannot_escape_output_directory(tmp_path, monkeypatch, filename):
    generator, _ = make_generator(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="local basename"):
        generator._save_shard({"first": torch.ones(4)}, filename)


def test_corrupt_completed_shard_forces_safe_regeneration(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    generator, _ = make_generator(tmp_path, monkeypatch, indexed=False)
    original_generate = generator.strategy.generate_tensor
    def fail_on_fourth(shape, dtype, name):
        if name == "fourth":
            raise RuntimeError("injected interruption")
        return original_generate(shape, dtype, name)
    monkeypatch.setattr(generator.strategy, "generate_tensor", fail_on_fourth)
    with pytest.raises(RuntimeError, match="injected interruption"):
        generator.generate()
    checkpoint = generator.incremental.load_checkpoint()
    filename = checkpoint["shard_map"]["first"]
    # Same names/shapes but wrong dtype must not be accepted as valid progress.
    save_file({"first": torch.ones(4, dtype=torch.float64), "second": torch.ones(4, dtype=torch.float64)}, str(tmp_path / filename))
    resumed, model = make_generator(tmp_path, monkeypatch, indexed=False)
    spy = MagicMock(wraps=resumed.strategy.generate_tensor)
    monkeypatch.setattr(resumed.strategy, "generate_tensor", spy)
    resumed.generate()
    _, tensors = read_weights(tmp_path)
    assert set(tensors) == dict(model.named_parameters()).keys()
    assert spy.call_count == 5
    assert all(t.dtype == torch.bfloat16 for t in tensors.values())


@pytest.mark.parametrize("field", ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"])
def test_validator_rejects_loading_diagnostics(monkeypatch, field):
    from vitriol.core import validator as module
    loader = MagicMock(return_value=(torch.nn.Linear(2, 2), {field: ["model.weight"]}))
    fallback = MagicMock()
    monkeypatch.setattr(module, "hf_load_causallm", loader)
    monkeypatch.setattr(module, "hf_load_model", fallback)
    report = module.ModelValidator("local-model").validate(run_inference=False)
    assert report.success is False
    assert field in report.errors[0]
    assert loader.call_args.kwargs["output_loading_info"] is True
    fallback.assert_not_called()


def test_validator_detects_missing_tensor_in_real_local_llama(tmp_path):
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM

    from vitriol.core.validator import ModelValidator
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32))
    model.save_pretrained(tmp_path, safe_serialization=True)
    validator = ModelValidator(str(tmp_path))
    assert validator.validate(run_inference=False).success is True
    path = tmp_path / "model.safetensors"
    weights = load_file(str(path))
    del weights["model.norm.weight"]
    save_file(weights, str(path), metadata={"format": "pt"})
    report = validator.validate(run_inference=False)
    assert report.success is False
    assert report.model_loadable is False
    assert "model.norm.weight" in report.errors[0]


def test_partial_original_shard_checkpoint_is_not_merged_by_overwriting(tmp_path, monkeypatch):
    generator, _ = make_generator(tmp_path, monkeypatch)
    monkeypatch.setattr(generator, "_save_configs", MagicMock(side_effect=RuntimeError("metadata interruption")))
    with pytest.raises(RuntimeError, match="metadata interruption"):
        generator.generate()
    checkpoint = generator.incremental.load_checkpoint()
    del checkpoint["shard_map"]["third"]
    generator.incremental.save_checkpoint(checkpoint)
    resumed, model = make_generator(tmp_path, monkeypatch)
    spy = MagicMock(wraps=resumed.strategy.generate_tensor)
    monkeypatch.setattr(resumed.strategy, "generate_tensor", spy)
    resumed.generate()
    _, tensors = read_weights(tmp_path)
    assert set(tensors) == dict(model.named_parameters()).keys()
    assert spy.call_count == 5


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("strategy", ["random", "quantized", "structured_sparse", "lowrank", "binary", "ternary", "quantum", "sparse", "compact", "ultra", "hybrid_ultra"])
def test_interrupted_generation_matches_uninterrupted(tmp_path, monkeypatch, indexed, strategy):
    def prepare(path):
        generator, model = make_generator(path, monkeypatch, strategy=strategy, indexed=indexed)
        if indexed:
            names = list(dict(model.named_parameters()))
            mapping = {n: f"model-{1 if i < 2 else 2:05d}-of-00002.safetensors" for i, n in enumerate(names)}
            monkeypatch.setattr(generator, "_get_original_shard_map", lambda: mapping)
        return generator

    torch.manual_seed(1984)
    complete = prepare(tmp_path / "complete")
    complete.generate()
    _, expected = read_weights(tmp_path / "complete")
    torch.manual_seed(1984)
    interrupted = prepare(tmp_path / "resumed")
    original = interrupted.strategy.generate_tensor

    def fail(shape, dtype, name):
        if name == "fourth":
            torch.rand(13)  # A failed attempt can already have consumed randomness.
            raise RuntimeError("interrupt")
        return original(shape, dtype, name)

    monkeypatch.setattr(interrupted.strategy, "generate_tensor", fail)
    with pytest.raises(RuntimeError, match="interrupt"):
        interrupted.generate()
    torch.manual_seed(9001)  # Simulate unrelated work or a fresh process.
    resumed = prepare(tmp_path / "resumed")
    resumed.generate()
    _, actual = read_weights(tmp_path / "resumed")
    assert expected.keys() == actual.keys()
    for name in expected:
        assert torch.equal(actual[name], expected[name]), name


def test_same_shape_checkpoint_content_change_forces_regeneration(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    generator, _ = make_generator(tmp_path, monkeypatch)
    monkeypatch.setattr(generator, "_save_configs", MagicMock(side_effect=RuntimeError("interrupt")))
    with pytest.raises(RuntimeError, match="interrupt"):
        generator.generate()
    path = next(tmp_path.glob("*.safetensors"))
    tensors = load_file(str(path))
    tensors["first"].add_(42)
    save_file(tensors, str(path))
    resumed, _ = make_generator(tmp_path, monkeypatch)
    spy = MagicMock(wraps=resumed.strategy.generate_tensor)
    monkeypatch.setattr(resumed.strategy, "generate_tensor", spy)
    resumed.generate()
    assert spy.call_count == 5


@pytest.mark.parametrize("key,value", [
    ("total_size", -1), ("total_size", 999), ("total_size", "40"),
    ("shard_count", 0), ("shard_count", True), ("shard_hashes", {}),
    ("completed_groups", ["missing"]), ("completed_groups", []), ("rng_state", {}),
])
def test_invalid_checkpoint_metadata_regenerates(tmp_path, monkeypatch, key, value):
    generator, _ = make_generator(tmp_path, monkeypatch)
    monkeypatch.setattr(generator, "_save_configs", MagicMock(side_effect=RuntimeError("interrupt")))
    with pytest.raises(RuntimeError, match="interrupt"):
        generator.generate()
    state = generator.incremental.load_checkpoint()
    state[key] = value
    generator.incremental.save_checkpoint(state)
    resumed, _ = make_generator(tmp_path, monkeypatch)
    spy = MagicMock(wraps=resumed.strategy.generate_tensor)
    monkeypatch.setattr(resumed.strategy, "generate_tensor", spy)
    resumed.generate()
    index, _ = read_weights(tmp_path)
    assert spy.call_count == 5
    assert index["metadata"]["total_size"] == 40


def test_checkpoint_does_not_repeat_empty_original_shards(tmp_path, monkeypatch):
    generator, model = make_generator(tmp_path, monkeypatch)
    mapping = {n: "model-00002-of-00002.safetensors" for n, _ in model.named_parameters()}
    mapping["removed_by_shrinking"] = "model-00001-of-00002.safetensors"
    monkeypatch.setattr(generator, "_get_original_shard_map", lambda: mapping)
    monkeypatch.setattr(generator, "_save_configs", MagicMock(side_effect=RuntimeError("interrupt")))
    with pytest.raises(RuntimeError, match="interrupt"):
        generator.generate()
    resumed, _ = make_generator(tmp_path, monkeypatch)
    monkeypatch.setattr(resumed, "_get_original_shard_map", lambda: mapping)
    spy = MagicMock(wraps=resumed.strategy.generate_tensor)
    monkeypatch.setattr(resumed.strategy, "generate_tensor", spy)
    resumed.generate()
    assert spy.call_count == 0
    assert len(list(tmp_path.glob("*.safetensors"))) == 2


@pytest.mark.parametrize("strategy", ["learned", "hybrid_learned"])
def test_learned_strategy_without_network_state_regenerates(tmp_path, monkeypatch, strategy, caplog):
    generator, _ = make_generator(tmp_path, monkeypatch, strategy=strategy)
    monkeypatch.setattr(generator, "_save_configs", MagicMock(side_effect=RuntimeError("interrupt")))
    with pytest.raises(RuntimeError, match="interrupt"):
        generator.generate()
    resumed, _ = make_generator(tmp_path, monkeypatch, strategy=strategy)
    spy = MagicMock(wraps=resumed.strategy.generate_tensor)
    monkeypatch.setattr(resumed.strategy, "generate_tensor", spy)
    resumed.generate()
    assert spy.call_count == 5
    assert "no reproducible resume support" in caplog.text
