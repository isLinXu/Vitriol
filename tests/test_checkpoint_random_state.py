"""Checkpoint RNG validation must not perturb unrelated default generators."""

import json
import random

import numpy as np
import pytest
import torch

from vitriol.core.checkpoint_state import capture_rng_state, inspect_shard, restore_rng_state
from vitriol.strategies.quantum import QuantumStrategy


@pytest.fixture(autouse=True)
def preserve_default_rngs():
    state = capture_rng_state()
    yield
    restore_rng_state(state)


def test_random_state_json_roundtrip_reproduces_all_cpu_streams(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: pytest.fail("must not initialize CUDA"))
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    random.seed(17)
    np.random.seed(23)
    torch.manual_seed(29)
    random.gauss(0, 1)  # Exercise cached Gaussian values as well as generator position.
    np.random.normal()
    state = json.loads(json.dumps(capture_rng_state()))
    expected = (random.gauss(0, 1), np.random.normal(size=7), torch.randn(7))
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    restore_rng_state(state)
    assert random.gauss(0, 1) == expected[0]
    np.testing.assert_array_equal(np.random.normal(size=7), expected[1])
    assert torch.equal(torch.randn(7), expected[2])


@pytest.mark.parametrize("key,value", [
    ("version", 99), ("runtime", []), ("python", [3, [0], None]),
    ("numpy", ["invalid", [], 0, 0, 0]), ("torch", "AA=="), ("torch", "not base64!"),
    ("cuda", ["AA=="]),
])
def test_invalid_random_state_leaves_global_streams_unchanged(key, value, monkeypatch):
    state = json.loads(json.dumps(capture_rng_state()))
    state[key] = value
    before = json.dumps(capture_rng_state())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises((ValueError, RuntimeError)):
        restore_rng_state(state)
    assert json.dumps(capture_rng_state()) == before


def test_quantum_scaling_does_not_reseed_default_generator(monkeypatch):
    strategy = QuantumStrategy()
    values = torch.ones(64, 8)
    torch.manual_seed(111)
    before = torch.get_rng_state().clone()
    scaled = strategy._apply_scaling(values, values.shape, "model.q_proj.weight")
    assert torch.equal(torch.get_rng_state(), before)
    # Process-randomized built-in hashes must not influence scaling.
    import vitriol.strategies.quantum as quantum_module

    monkeypatch.setattr(quantum_module, "hash", lambda _: 0, raising=False)
    other = strategy._apply_scaling(values, values.shape, "model.q_proj.weight")
    assert torch.equal(scaled, other)


def test_safetensors_inspection_uses_metadata_only(tmp_path, monkeypatch):
    import safetensors
    from safetensors.torch import save_file

    path = tmp_path / "weights.safetensors"
    save_file({"weight": torch.ones(20, 30, dtype=torch.bfloat16)}, str(path))
    actual_open = safetensors.safe_open

    class MetadataOnly:
        def __init__(self, *args, **kwargs):
            self.reader = actual_open(*args, **kwargs)

        def __enter__(self):
            self.reader.__enter__()
            return self

        def __exit__(self, *args):
            return self.reader.__exit__(*args)

        def keys(self):
            return self.reader.keys()

        def get_slice(self, name):
            return self.reader.get_slice(name)

        def get_tensor(self, name):
            pytest.fail("checkpoint inspection must not load tensor payloads")

    monkeypatch.setattr(safetensors, "safe_open", MetadataOnly)
    assert inspect_shard(str(path)) == ({"weight": (20, 30)}, {"weight": "torch.bfloat16"}, 1200)
