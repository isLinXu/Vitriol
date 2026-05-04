"""Tests for config/manager.py and config/settings.py."""

import os
import tempfile
from pathlib import Path

import pytest

from vitriol.config.manager import (
    GenerationConfig,
    SecurityOptions,
    _coerce_bool,
    build_generation_config,
    _load_generation_dict_from_env,
    _load_generation_dict_from_yaml,
)
from vitriol.config.settings import (
    ConfigManager,
    ConfigEnvironment,
    GenerationDefaults,
    NASConfig,
    SecurityConfig,
    SystemConfig,
    VitriolConfig,
    get_config,
    init_config,
)


class TestCoerceBool:
    """Tests for _coerce_bool."""

    def test_true_values(self):
        assert _coerce_bool(True) is True
        assert _coerce_bool("true") is True
        assert _coerce_bool("True") is True
        assert _coerce_bool("1") is True
        assert _coerce_bool("yes") is True
        assert _coerce_bool("on") is True

    def test_false_values(self):
        assert _coerce_bool(False) is False
        assert _coerce_bool("false") is False
        assert _coerce_bool("0") is False
        assert _coerce_bool("no") is False
        assert _coerce_bool("off") is False
        assert _coerce_bool("") is False


class TestSecurityOptions:
    """Tests for SecurityOptions."""

    def test_defaults(self):
        so = SecurityOptions()
        assert so.trust_remote_code is True
        assert so.allow_network is True
        assert so.local_files_only is False

    def test_custom(self):
        so = SecurityOptions(trust_remote_code=False, allow_network=False, local_files_only=True)
        assert so.trust_remote_code is False
        assert so.allow_network is False
        assert so.local_files_only is True


class TestGenerationConfig:
    """Tests for GenerationConfig."""

    def test_defaults(self):
        gc = GenerationConfig()
        assert gc.max_shard_size == "5GB"
        assert gc.dtype == "bfloat16"
        assert gc.strategy == "random"
        assert gc.auto_validate is True
        assert gc.n_bits == 8
        assert gc.rank == 16
        assert gc.sparsity == 0.5
        assert gc.security is not None

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="Invalid strategy"):
            GenerationConfig(strategy="invalid")

    def test_invalid_sparsity(self):
        with pytest.raises(ValueError, match="Sparsity must be in"):
            GenerationConfig(sparsity=1.5)
        with pytest.raises(ValueError, match="Sparsity must be in"):
            GenerationConfig(sparsity=-0.1)

    def test_invalid_n_bits(self):
        with pytest.raises(ValueError, match="n_bits must be in"):
            GenerationConfig(n_bits=0)
        with pytest.raises(ValueError, match="n_bits must be in"):
            GenerationConfig(n_bits=64)

    def test_valid_strategies(self):
        valid = ['random', 'sparse', 'compact', 'ultra', 'hybrid_ultra', 'ternary',
                 'binary', 'quantized', 'lowrank', 'structured_sparse',
                 'learned', 'hybrid_learned', 'quantum']
        for s in valid:
            gc = GenerationConfig(strategy=s)
            assert gc.strategy == s

    def test_from_yaml_not_exists(self):
        gc = GenerationConfig.from_yaml(Path("/nonexistent/path.yaml"))
        assert gc is not None
        assert gc.max_shard_size == "5GB"

    def test_from_env(self):
        gc = GenerationConfig.from_env()
        assert gc is not None


class TestLoadGenerationDictFromYaml:
    """Tests for _load_generation_dict_from_yaml."""

    def test_nonexistent_path(self):
        result = _load_generation_dict_from_yaml(Path("/nonexistent"))
        assert result == {}

    def test_valid_yaml(self):
        yaml_content = "default:\n  max_shard_size: 10GB\n  dtype: float16\n"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            result = _load_generation_dict_from_yaml(Path(path))
            assert result["max_shard_size"] == "10GB"
            assert result["dtype"] == "float16"
        finally:
            os.unlink(path)

    def test_yaml_no_default(self):
        yaml_content = "other:\n  key: value\n"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            result = _load_generation_dict_from_yaml(Path(path))
            assert result == {}
        finally:
            os.unlink(path)


class TestLoadGenerationDictFromEnv:
    """Tests for _load_generation_dict_from_env."""

    def test_with_env_vars(self, monkeypatch):
        monkeypatch.setenv("VITRIOL_MAX_SHARD_SIZE", "10GB")
        monkeypatch.setenv("VITRIOL_DTYPE", "float16")
        result = _load_generation_dict_from_env()
        assert result["max_shard_size"] == "10GB"
        assert result["dtype"] == "float16"

    def test_without_env_vars(self):
        result = _load_generation_dict_from_env()
        assert "max_shard_size" not in result
        assert "dtype" not in result


class TestBuildGenerationConfig:
    """Tests for build_generation_config."""

    def test_default_build(self):
        gc = build_generation_config()
        assert gc is not None
        assert gc.strategy in {'random','sparse','compact','ultra','hybrid_ultra','ternary',
                               'binary','quantized','lowrank','structured_sparse',
                               'learned','hybrid_learned','quantum'}

    def test_with_overrides(self):
        gc = build_generation_config(overrides={"strategy": "sparse", "dtype": "float16"})
        assert gc.strategy == "sparse"
        assert gc.dtype == "float16"

    def test_with_yaml_file(self):
        yaml_content = "default:\n  strategy: compact\n  dtype: float16\n"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            gc = build_generation_config(config_path=Path(path))
            assert gc.strategy == "compact"
            assert gc.dtype == "float16"
        finally:
            os.unlink(path)


class TestVitriolConfig:
    """Tests for VitriolConfig dataclass."""

    def test_defaults(self):
        vc = VitriolConfig()
        assert vc.environment == "development"
        assert isinstance(vc.generation, GenerationDefaults)
        assert isinstance(vc.nas, NASConfig)
        assert isinstance(vc.system, SystemConfig)
        assert isinstance(vc.security, SecurityConfig)
        assert vc.custom == {}


class TestConfigManager:
    """Tests for ConfigManager."""

    def test_creation(self):
        cm = ConfigManager()
        assert cm is not None

    def test_get_nested(self):
        cm = ConfigManager()
        result = cm.get("generation.default_strategy")
        assert result == "compact"

    def test_get_missing(self):
        cm = ConfigManager()
        result = cm.get("nonexistent.path", default="fallback")
        assert result == "fallback"

    def test_set_nested(self):
        cm = ConfigManager()
        cm.set("generation.default_strategy", "ultra")
        assert cm.get("generation.default_strategy") == "ultra"

    def test_set_new_key(self):
        cm = ConfigManager()
        cm.set("custom.new_key", "new_value")
        assert cm.get("custom.new_key") == "new_value"

    def test_to_dict(self):
        cm = ConfigManager()
        d = cm.to_dict()
        assert "version" in d
        assert "environment" in d
        assert "generation" in d
        assert "nas" in d
        assert "system" in d
        assert "security" in d

    def test_save_and_load(self):
        cm = ConfigManager()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            cm.save_to_file(str(path))
            assert path.exists()

            cm2 = ConfigManager()
            cm2.load_from_file(str(path))
            assert cm2.get("version") == cm.get("version")

    def test_save_json(self):
        cm = ConfigManager()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            cm.save_to_file(str(path), format="json")
            assert path.exists()

    def test_load_from_nonexistent(self):
        cm = ConfigManager()
        result = cm.load_from_file("/nonexistent/path.yaml")
        assert result is cm  # returns self

    def test_watchers(self):
        cm = ConfigManager()
        calls = []
        def watcher(key, value):
            calls.append((key, value))
        cm.watch(watcher)
        cm.set("system.log_level", "DEBUG")
        assert len(calls) >= 1
        cm.unwatch(watcher)

    def test_environment_methods(self):
        cm = ConfigManager()
        cm.set("environment", "production")
        assert cm.is_production() is True
        assert cm.is_development() is False
        assert cm.get_environment() == ConfigEnvironment.PRODUCTION

    def test_get_config_singleton(self):
        cm1 = get_config()
        cm2 = get_config()
        assert cm1 is cm2

    def test_init_config(self):
        cm = init_config()
        assert cm is not None


class TestGenerationDefaults:
    """Tests for GenerationDefaults."""

    def test_defaults(self):
        gd = GenerationDefaults()
        assert gd.default_strategy == "compact"
        assert gd.default_dtype == "bfloat16"
        assert gd.max_shard_size == "5GB"
        assert gd.parallel_workers == 4
        assert gd.use_memory_mapping is True
        assert gd.compression_level == 6
        assert gd.verify_checksums is True


class TestNASConfig:
    """Tests for NASConfig."""

    def test_defaults(self):
        nc = NASConfig()
        assert nc.default_algorithm == "evolutionary"
        assert nc.population_size == 20
        assert nc.n_iterations == 100
        assert nc.mutation_rate == 0.1
        assert nc.crossover_rate == 0.8
        assert nc.use_rl_agent is True


class TestSystemConfig:
    """Tests for SystemConfig."""

    def test_defaults(self):
        sc = SystemConfig()
        assert sc.log_level == "INFO"
        assert sc.max_memory_gb == 32.0
        assert sc.gpu_enabled is True
        assert sc.gpu_memory_fraction == 0.9


class TestSecurityConfig:
    """Tests for SecurityConfig."""

    def test_defaults(self):
        sc = SecurityConfig()
        assert sc.enable_encryption is False
        assert sc.verify_signatures is True
        assert sc.allowed_hosts == ["localhost"]
        assert sc.api_key_required is False
        assert sc.rate_limit_requests == 100
