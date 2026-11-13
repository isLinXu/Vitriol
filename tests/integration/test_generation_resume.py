"""Fresh-process resume must reproduce complete local model artifacts."""

import json
import os
import subprocess
import sys

import pytest
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

pytestmark = pytest.mark.integration

_SCRIPT = r'''
import os
import sys
from pathlib import Path
import torch
from vitriol.config.manager import GenerationConfig, SecurityOptions
from vitriol.core.generator import MinimalWeightGenerator

source, output, strategy, mode = sys.argv[1:]
torch.manual_seed(9001 if mode == "resume" else 1984)
generator = MinimalWeightGenerator(
    source, output,
    config=GenerationConfig(strategy=strategy, max_shard_size="128",
                            security=SecurityOptions(allow_network=False)),
    shrink_config=False,
)
if mode in {"interrupt", "metadata_interrupt", "kill_rename"}:
    original = generator.strategy.generate_tensor
    def fail(shape, dtype, name):
        if name == "model.layers.0.self_attn.v_proj.weight":
            torch.rand(19)
            raise RuntimeError("test interruption")
        return original(shape, dtype, name)
    if mode == "interrupt":
        generator.strategy.generate_tensor = fail
    elif mode == "metadata_interrupt":
        def fail_metadata():
            raise RuntimeError("test interruption")
        generator._write_manifest = fail_metadata
    else:
        replace = os.replace
        def kill_after_rename(source, destination):
            replace(source, destination)
            if Path(source).parent == Path(output) and "-of-" in Path(destination).name:
                os._exit(73)  # Abrupt process death: no Python cleanup runs.
        os.replace = kill_after_rename
    try:
        generator.generate()
    except RuntimeError as exc:
        if str(exc) != "test interruption":
            raise
        assert generator.incremental.load_checkpoint()["shard_map"]
    else:
        raise AssertionError("interruption was not reached")
else:
    if mode == "resume":
        completed = set(generator.incremental.load_checkpoint()["shard_map"])
        original = generator.strategy.generate_tensor
        def only_missing(shape, dtype, name):
            assert name not in completed, "persisted weights were regenerated"
            return original(shape, dtype, name)
        generator.strategy.generate_tensor = only_missing
    generator.generate()
'''


@pytest.mark.parametrize("strategy", ["random", "quantum"])
@pytest.mark.parametrize("interruption", ["interrupt", "metadata_interrupt", "kill_rename"])
def test_fresh_process_resume_matches_complete_llama(tiny_llama_model_dir, tmp_path, strategy, interruption):
    reference = tmp_path / "reference"
    resumed = tmp_path / "resumed"
    for seed, mode, destination in [(1, "complete", reference), (2, interruption, resumed), (3, "resume", resumed)]:
        process = subprocess.run(
            [sys.executable, "-c", _SCRIPT, str(tiny_llama_model_dir), str(destination), strategy, mode],
            env={**os.environ, "PYTHONHASHSEED": str(seed), "HF_HUB_OFFLINE": "1"},
            capture_output=True, text=True, timeout=45,
        )
        assert process.returncode == (73 if mode == "kill_rename" else 0), process.stdout + process.stderr
    expected_index = json.loads((reference / "model.safetensors.index.json").read_text())
    actual_index = json.loads((resumed / "model.safetensors.index.json").read_text())
    assert expected_index == actual_index
    for filename in set(expected_index["weight_map"].values()):
        expected = load_file(str(reference / filename))
        actual = load_file(str(resumed / filename))
        assert expected.keys() == actual.keys()
        for name in expected:
            assert torch.equal(expected[name], actual[name]), name
    assert not (resumed / ".vitriol_checkpoint.json").exists()
    model = AutoModelForCausalLM.from_pretrained(str(resumed), local_files_only=True).eval()
    with torch.no_grad():
        logits = model(torch.tensor([[1, 2, 3]])).logits
    assert logits.shape == (1, 3, 128)
    assert torch.isfinite(logits).all()
