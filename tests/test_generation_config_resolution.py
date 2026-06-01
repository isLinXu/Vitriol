import asyncio

from click.testing import CliRunner

from vitriol.api import server
from vitriol.cli.commands.generate import generate
from vitriol.config.settings import init_config


def test_build_generation_config_uses_defaults_when_no_inputs(monkeypatch):
    from vitriol.config import manager as manager_module

    monkeypatch.delenv("VITRIOL_MAX_SHARD_SIZE", raising=False)
    monkeypatch.delenv("VITRIOL_DTYPE", raising=False)

    config = manager_module.build_generation_config()

    assert config.max_shard_size == "5GB"
    assert config.dtype == "bfloat16"
    assert config.strategy == "random"


def test_build_generation_config_prefers_env_over_defaults(monkeypatch):
    from vitriol.config import manager as manager_module

    monkeypatch.setenv("VITRIOL_MAX_SHARD_SIZE", "7GB")
    monkeypatch.setenv("VITRIOL_DTYPE", "float16")

    config = manager_module.build_generation_config()

    assert config.max_shard_size == "7GB"
    assert config.dtype == "float16"


def test_build_generation_config_prefers_yaml_over_env(tmp_path, monkeypatch):
    from vitriol.config import manager as manager_module

    config_file = tmp_path / "vitriol.yaml"
    config_file.write_text(
        "default:\n"
        "  strategy: compact\n"
        "  dtype: bfloat16\n"
        "  max_shard_size: 3GB\n"
    )
    monkeypatch.setenv("VITRIOL_MAX_SHARD_SIZE", "7GB")
    monkeypatch.setenv("VITRIOL_DTYPE", "float16")

    config = manager_module.build_generation_config(config_path=config_file)

    assert config.strategy == "compact"
    assert config.dtype == "bfloat16"
    assert config.max_shard_size == "3GB"


def test_build_generation_config_prefers_explicit_over_yaml_and_env(tmp_path, monkeypatch):
    from vitriol.config import manager as manager_module

    config_file = tmp_path / "vitriol.yaml"
    config_file.write_text(
        "default:\n"
        "  strategy: compact\n"
        "  dtype: bfloat16\n"
        "  max_shard_size: 3GB\n"
        "  n_bits: 4\n"
        "  rank: 8\n"
        "  sparsity: 0.25\n"
    )
    monkeypatch.setenv("VITRIOL_MAX_SHARD_SIZE", "7GB")
    monkeypatch.setenv("VITRIOL_DTYPE", "float16")

    config = manager_module.build_generation_config(
        config_path=config_file,
        overrides={
            "strategy": "ultra",
            "max_shard_size": "1GB",
            "n_bits": 2,
            "rank": 32,
            "sparsity": 0.9,
            "trust_remote_code": False,
        },
    )

    assert config.strategy == "ultra"
    assert config.max_shard_size == "1GB"
    assert config.n_bits == 2
    assert config.rank == 32
    assert config.sparsity == 0.9
    assert config.security.trust_remote_code is False


def test_cli_generate_uses_build_generation_config(monkeypatch, tmp_path):
    captured = {}

    def fake_build_generation_config(*, config_path=None, overrides=None):
        captured["config_path"] = config_path
        captured["overrides"] = overrides

        from vitriol.config.manager import GenerationConfig

        return GenerationConfig(strategy="compact")

    class StubGenerator:
        def __init__(self, model_id, output_dir, config, **kwargs):
            captured["model_id"] = model_id
            captured["output_dir"] = output_dir
            captured["config"] = config

        def generate(self):
            return None

    monkeypatch.setattr("vitriol.cli.commands.generate.build_generation_config", fake_build_generation_config)
    monkeypatch.setattr("vitriol.cli.commands.generate.MinimalWeightGenerator", StubGenerator)

    config_file = tmp_path / "vitriol.yaml"
    config_file.write_text("default:\n  strategy: random\n")

    runner = CliRunner()
    result = runner.invoke(
        generate,
        [
            "demo/model",
            "--output-dir",
            str(tmp_path / "out"),
            "--strategy",
            "ultra",
            "--dtype",
            "float16",
        ],
        obj={"config_path": config_file, "trust_remote_code": False},
    )

    assert result.exit_code == 0
    assert captured["config_path"] == config_file
    assert captured["overrides"]["strategy"] == "ultra"
    assert captured["overrides"]["dtype"] == "float16"
    assert captured["overrides"]["trust_remote_code"] is False


def test_cli_root_exposes_version():
    from vitriol.cli.main import cli
    from vitriol.version import __version__

    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.output


def test_cli_generate_propagates_global_offline_options(monkeypatch, tmp_path):
    from vitriol.cli.main import cli

    captured = {}

    def fake_build_generation_config(*, config_path=None, overrides=None):
        captured["config_path"] = config_path
        captured["overrides"] = overrides

        from vitriol.config.manager import GenerationConfig

        return GenerationConfig(strategy="ultra")

    class StubGenerator:
        def __init__(self, model_id, output_dir, config, **kwargs):
            captured["model_id"] = model_id
            captured["output_dir"] = output_dir
            captured["config"] = config

        def generate(self):
            return None

    monkeypatch.setattr("vitriol.cli.commands.generate.build_generation_config", fake_build_generation_config)
    monkeypatch.setattr("vitriol.cli.commands.generate.MinimalWeightGenerator", StubGenerator)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--no-trust-remote-code",
            "--offline",
            "generate",
            "demo/model",
            "--output-dir",
            str(tmp_path / "out"),
            "--strategy",
            "ultra",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["overrides"]["trust_remote_code"] is False
    assert captured["overrides"]["allow_network"] is False
    assert captured["overrides"]["local_files_only"] is True


def test_process_generation_job_uses_build_generation_config(monkeypatch):
    cfg = init_config()
    cfg.set("security.api_key_required", False)
    server.active_jobs.clear()
    server.active_jobs["job-1"] = {
        "id": "job-1",
        "type": "generation",
        "status": "queued",
        "request": {
            "model_id": "demo/model",
            "strategy": "compact",
            "dtype": "float16",
            "max_shard_size": "2GB",
            "output_dir": "/tmp/out",
        },
    }
    captured = {}

    def fake_build_generation_config(*, config_path=None, overrides=None):
        captured["config_path"] = config_path
        captured["overrides"] = overrides

        from vitriol.config.manager import GenerationConfig

        return GenerationConfig(
            strategy=overrides["strategy"],
            dtype=overrides["dtype"],
            max_shard_size=overrides["max_shard_size"],
        )

    class StubGenerator:
        def __init__(self, model_id, output_dir, config, **kwargs):
            captured["model_id"] = model_id
            captured["output_dir"] = output_dir
            captured["config"] = config

        def generate(self):
            class _Result:
                output_dir = "/tmp/out"
                manifest_path = "/tmp/out/vitriol-manifest.json"
                index_path = "/tmp/out/model.safetensors.index.json"
                total_size = 1024
                generated_at = "2026-04-04T00:00:00Z"

                def to_dict(self):
                    return {
                        "output_dir": self.output_dir,
                        "manifest_path": self.manifest_path,
                        "index_path": self.index_path,
                        "total_size": self.total_size,
                        "generated_at": self.generated_at,
                    }

            return _Result()

    monkeypatch.setattr("vitriol.api.server.build_generation_config", fake_build_generation_config)
    monkeypatch.setattr("vitriol.core.generator.MinimalWeightGenerator", StubGenerator)

    asyncio.run(server.process_generation_job("job-1"))

    assert captured["config_path"] is None
    assert captured["overrides"]["strategy"] == "compact"
    assert captured["overrides"]["dtype"] == "float16"
    assert captured["overrides"]["max_shard_size"] == "2GB"
    assert captured["overrides"]["trust_remote_code"] is False
