"""Independent processes must not overwrite a live generation's artifacts."""

import os
import select
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.skipif(os.name == "nt", reason="POSIX process termination test")]

_SCRIPT = r'''
import sys
import tempfile
from pathlib import Path
from vitriol.config.manager import GenerationConfig, SecurityOptions
from vitriol.core.generator import MinimalWeightGenerator
from vitriol.utils.exceptions import OutputDirectoryBusyError

source, output, mode = sys.argv[1:]
generator = MinimalWeightGenerator(
    source, output, shrink_config=False,
    config=GenerationConfig(strategy="random", max_shard_size="16KB",
                            security=SecurityOptions(allow_network=False)),
)
if mode == "owner":
    write_manifest = generator._write_manifest
    def pause_before_manifest():
        with tempfile.TemporaryDirectory(prefix=".vitriol-shard-", dir=output) as temporary:
            (Path(temporary) / "active").write_text("owned")
            print("OWNER_READY", flush=True)
            if sys.stdin.readline().strip() != "release":
                raise RuntimeError("owner input closed")
        write_manifest()
    generator._write_manifest = pause_before_manifest
elif mode == "resume":
    def never_regenerate(*args, **kwargs):
        raise AssertionError("completed weights regenerated")
    generator.strategy.generate_tensor = never_regenerate
try:
    generator.generate()
except OutputDirectoryBusyError as exc:
    assert mode == "contender" and exc.recoverable
    print("BUSY", flush=True)
else:
    assert mode != "contender", "competing process entered generation"
    print("DONE", flush=True)
'''


@pytest.mark.parametrize("exit_mode", ["normal", "kill"])
def test_real_model_writer_excludes_contender_and_releases_after_exit(tiny_llama_model_dir, tmp_path, exit_mode):
    output = tmp_path / "output"
    environment = {**os.environ, "HF_HUB_OFFLINE": "1"}
    command = [sys.executable, "-c", _SCRIPT, str(tiny_llama_model_dir)]
    errors = tmp_path / "owner.stderr"
    with errors.open("w") as stderr:
        owner = subprocess.Popen(
            [*command, str(output), "owner"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=stderr, text=True, env=environment,
        )
        try:
            ready, _, _ = select.select([owner.stdout], [], [], 45)
            assert ready and owner.stdout.readline().strip() == "OWNER_READY", errors.read_text()
            before = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}
            alias = tmp_path / "alias"
            alias.symlink_to(output, target_is_directory=True)
            contender = subprocess.run(
                [*command, str(alias), "contender"], env=environment,
                capture_output=True, text=True, timeout=45,
            )
            assert contender.returncode == 0, contender.stdout + contender.stderr
            assert "BUSY" in contender.stdout
            after = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}
            assert after == before
            if exit_mode == "kill":
                owner.kill()
                owner.wait(timeout=10)
                assert list(output.glob(".vitriol-shard-*"))
                resumed = subprocess.run(
                    [*command, str(output), "resume"], env=environment,
                    capture_output=True, text=True, timeout=45,
                )
                assert resumed.returncode == 0, resumed.stdout + resumed.stderr
            else:
                owner.stdin.write("release\n")
                owner.stdin.flush()
                owner.communicate(timeout=45)
                assert owner.returncode == 0, errors.read_text()
            assert not list(output.glob(".vitriol-shard-*"))
            assert (output / "vitriol-manifest.json").is_file()
            assert not (output / ".vitriol_checkpoint.json").exists()
            from vitriol.core.output_lock import output_directory_lock

            with output_directory_lock(str(output)):
                assert (output / ".vitriol-generation.lock").is_file()
        finally:
            if owner.poll() is None:
                owner.kill()
                owner.wait(timeout=10)
            if owner.stdin:
                owner.stdin.close()
            if owner.stdout:
                owner.stdout.close()
