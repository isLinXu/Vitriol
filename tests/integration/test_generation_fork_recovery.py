"""A surviving fork worker must not prevent recovery of its dead generator."""

import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.skipif(sys.platform != "linux", reason="uses Linux child subreaper")]

_SCRIPT = r'''
import ctypes
import json
import os
import sys
from pathlib import Path

# Keep the controller free of ML threads until after forking the owner.
assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0
source, destination = sys.argv[1:]
output = Path(destination)
ready_r, ready_w = os.pipe()
exit_r, exit_w = os.pipe()
owner = os.fork()
if owner == 0:
    os.close(ready_r)
    os.close(exit_w)
    from vitriol.config.manager import GenerationConfig, SecurityOptions
    from vitriol.core.generator import MinimalWeightGenerator
    generator = MinimalWeightGenerator(
        source, destination, shrink_config=False,
        config=GenerationConfig(strategy="random", max_shard_size="16KB",
                                security=SecurityOptions(allow_network=False)),
    )
    def interrupt_before_manifest():
        worker = os.fork()
        if worker == 0:
            os.close(ready_w)
            os.read(exit_r, 1)
            os._exit(0)
        os.write(ready_w, str(worker).encode())
        os._exit(73)
    generator._write_manifest = interrupt_before_manifest
    generator.generate()
    os._exit(1)

os.close(ready_w)
os.close(exit_r)
worker = None
try:
    worker_message = os.read(ready_r, 64)
    _, status = os.waitpid(owner, 0)
    assert os.waitstatus_to_exitcode(status) == 73
    worker = int(worker_message)
    assert os.waitpid(worker, os.WNOHANG) == (0, 0), "worker must still be alive"
    index_before = (output / "model.safetensors.index.json").read_bytes()
    filenames = set(json.loads(index_before)["weight_map"].values())
    weights_before = {name: (output / name).read_bytes() for name in filenames}
    assert (output / ".vitriol_checkpoint.json").is_file()

    from vitriol.config.manager import GenerationConfig, SecurityOptions
    from vitriol.core.generator import MinimalWeightGenerator
    resumed = MinimalWeightGenerator(
        source, destination, shrink_config=False,
        config=GenerationConfig(strategy="random", max_shard_size="16KB",
                                security=SecurityOptions(allow_network=False)),
    )
    def never_regenerate(*args, **kwargs):
        raise AssertionError("completed weights regenerated")
    resumed.strategy.generate_tensor = never_regenerate
    resumed.generate()
    assert (output / "model.safetensors.index.json").read_bytes() == index_before
    assert {name: (output / name).read_bytes() for name in filenames} == weights_before
    assert (output / "vitriol-manifest.json").is_file()
    assert not (output / ".vitriol_checkpoint.json").exists()
    assert os.waitpid(worker, os.WNOHANG) == (0, 0), "worker exited before recovery"

    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(destination, local_files_only=True).eval()
    with torch.no_grad():
        logits = model(torch.tensor([[1, 2, 3]])).logits
    assert logits.shape == (1, 3, 128)
    assert torch.isfinite(logits).all()
finally:
    os.close(ready_r)
    os.close(exit_w)  # EOF releases the worker even if an assertion fails.
    if worker is not None:
        os.waitpid(worker, 0)
'''


def test_real_model_resumes_while_dead_owners_fork_worker_survives(tiny_llama_model_dir, tmp_path):
    process = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(tiny_llama_model_dir), str(tmp_path / "output")],
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
        capture_output=True, text=True, timeout=90,
    )
    assert process.returncode == 0, process.stdout + process.stderr
