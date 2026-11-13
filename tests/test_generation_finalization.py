"""Failure injection at the generation-to-metadata boundary."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from test_core_generation_integrity import make_generator, read_weights


@pytest.mark.parametrize("strategy", ["random", "ultra", "sparse"])
@pytest.mark.parametrize("stage", ["_save_configs", "_copy_custom_code_files", "_save_tokenizer", "_add_readme_metadata", "_write_manifest"])
def test_metadata_failure_resumes_without_generating_weights(tmp_path, monkeypatch, strategy, stage):
    generator, _ = make_generator(tmp_path, monkeypatch, strategy=strategy, indexed=False)
    monkeypatch.setattr(generator, stage, MagicMock(side_effect=OSError("metadata interruption")))
    with pytest.raises(OSError, match="metadata interruption"):
        generator.generate()
    _, expected = read_weights(tmp_path)
    resumed, _ = make_generator(tmp_path, monkeypatch, strategy=strategy, indexed=False)
    monkeypatch.setattr(resumed.strategy, "generate_tensor", MagicMock(side_effect=AssertionError("regenerated")))
    resumed.generate()
    index, actual = read_weights(tmp_path)
    assert resumed.shard_map == index["weight_map"]
    assert all(torch.equal(actual[n], expected[n]) for n in expected)
    assert resumed.incremental.load_checkpoint() is None


def test_final_index_replaces_stale_destination(tmp_path, monkeypatch):
    generator, _ = make_generator(tmp_path, monkeypatch, indexed=False)
    save_index = generator._save_index

    def with_stale_destination(*args):
        (tmp_path / "model-00001-of-00003.safetensors").write_bytes(b"stale artifact")
        return save_index(*args)

    monkeypatch.setattr(generator, "_save_index", with_stale_destination)
    generator.generate()
    index, tensors = read_weights(tmp_path)
    assert len(tensors) == 5
    assert generator.shard_map == index["weight_map"]
    assert len(list(tmp_path.glob("*.safetensors"))) == 3


def test_final_result_references_existing_canonical_shards(tmp_path, monkeypatch):
    generator, _ = make_generator(tmp_path, monkeypatch, indexed=False)
    generator.generate()
    assert all((Path(generator.output_dir) / filename).is_file() for filename in generator.shard_map.values())


@pytest.mark.parametrize("strategy", ["random", "ultra"])
@pytest.mark.parametrize("stage", ["journal", "rename1", "rename2", "rename3", "index", "checkpoint"])
def test_finalization_interruption_keeps_weights_recoverable(tmp_path, monkeypatch, strategy, stage):
    import json

    import vitriol.core.generator as module

    def prepare():
        generator, _ = make_generator(tmp_path, monkeypatch, strategy=strategy, indexed=False)
        if strategy == "ultra":
            generator.max_shard_size = 4
        return generator

    generator = prepare()
    generate_tensor = generator.strategy.generate_tensor
    expected = {}

    def remember(shape, dtype, name):
        tensor = generate_tensor(shape, dtype, name)
        expected[name] = tensor.clone()
        return tensor

    monkeypatch.setattr(generator.strategy, "generate_tensor", remember)
    replace = module.os.replace
    prefix = generator.strategy.get_shard_prefix()
    suffix = generator.strategy.file_extension
    triggered = False

    def fail(source, destination):
        nonlocal triggered
        source, destination = Path(source), Path(destination)
        should_fail = False
        if stage.startswith("rename"):
            should_fail = source.parent == tmp_path and source.name == f"{prefix}-{int(stage[-1]):05d}.{suffix}"
        elif stage == "index":
            should_fail = destination.name.endswith(".index.json")
        elif destination.name == ".vitriol_checkpoint.json":
            state = json.loads(source.read_text())
            if stage == "journal":
                should_fail = bool(state.get("rename_plan"))
            elif stage == "checkpoint":
                should_fail = bool(state.get("shard_map")) and all("-of-" in f for f in state["shard_map"].values())
        if should_fail:
            triggered = True
            raise OSError("finalization interruption")
        return replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(module.os, "replace", fail)
        with pytest.raises(OSError, match="finalization interruption"):
            generator.generate()
    assert triggered
    assert generator.incremental.load_checkpoint() is not None
    resumed = prepare()
    monkeypatch.setattr(resumed.strategy, "generate_tensor", MagicMock(side_effect=AssertionError("regenerated")))
    resumed.generate()
    index, actual = read_weights(tmp_path)
    assert all(torch.equal(actual[n], expected[n]) for n in expected)
    assert resumed.shard_map == index["weight_map"]
    assert len(list(tmp_path.glob(f"*.{suffix}"))) == 3
    assert not list(tmp_path.glob(".vitriol-index-*"))


def test_failed_index_serialization_preserves_previous_index(tmp_path, monkeypatch):
    import vitriol.core.generator as module

    generator, _ = make_generator(tmp_path, monkeypatch, indexed=False)
    generator.generate()
    path = next(tmp_path.glob("*.index.json"))
    before = path.read_bytes()

    def fail(data, stream, **kwargs):
        stream.write('{"partial":')
        raise OSError("serialization interruption")

    monkeypatch.setattr(module.json, "dump", fail)
    with pytest.raises(OSError, match="serialization interruption"):
        generator._save_index(3)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".vitriol-index-*"))


def test_checkpoint_commit_failure_stops_generation(tmp_path, monkeypatch):
    import vitriol.core.incremental as module

    generator, _ = make_generator(tmp_path, monkeypatch, indexed=False)
    replace = module.os.replace

    def fail(source, destination):
        if Path(destination).name == ".vitriol_checkpoint.json":
            raise OSError("checkpoint commit failure")
        return replace(source, destination)

    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(OSError, match="checkpoint commit failure"):
        generator.generate()
    assert not list(tmp_path.glob("*.index.json"))


def test_legacy_sha256_checkpoint_still_resumes(tmp_path, monkeypatch):
    import hashlib

    generator, _ = make_generator(tmp_path, monkeypatch, indexed=False)
    monkeypatch.setattr(generator, "_save_configs", MagicMock(side_effect=OSError("interruption")))
    with pytest.raises(OSError, match="interruption"):
        generator.generate()
    state = generator.incremental.load_checkpoint()
    state["shard_hashes"] = {n: hashlib.sha256((tmp_path / n).read_bytes()).hexdigest() for n in state["shard_hashes"]}
    generator.incremental.save_checkpoint(state)
    resumed, _ = make_generator(tmp_path, monkeypatch, indexed=False)
    monkeypatch.setattr(resumed.strategy, "generate_tensor", MagicMock(side_effect=AssertionError("regenerated")))
    resumed.generate()
    assert len(read_weights(tmp_path)[1]) == 5


@pytest.mark.parametrize("destination", ["../outside.safetensors", "/outside.safetensors", ".", ".."])
def test_invalid_rename_journal_forces_regeneration(tmp_path, monkeypatch, destination):
    generator, _ = make_generator(tmp_path, monkeypatch, indexed=False)
    monkeypatch.setattr(generator, "_save_configs", MagicMock(side_effect=OSError("interruption")))
    with pytest.raises(OSError, match="interruption"):
        generator.generate()
    state = generator.incremental.load_checkpoint()
    state["rename_plan"] = {next(iter(state["shard_hashes"])): destination}
    generator.incremental.save_checkpoint(state)
    resumed, _ = make_generator(tmp_path, monkeypatch, indexed=False)
    spy = MagicMock(wraps=resumed.strategy.generate_tensor)
    monkeypatch.setattr(resumed.strategy, "generate_tensor", spy)
    resumed.generate()
    assert spy.call_count == 5
