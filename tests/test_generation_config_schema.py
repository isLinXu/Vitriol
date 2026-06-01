"""Tests for GenerationConfig validation and the JSON Schema contract."""

import pytest

from vitriol.config.manager import (
    GenerationConfig,
    build_generation_config,
    generation_config_schema,
    validate_generation_dict,
)
from vitriol.utils.exceptions import ConfigValidationError


def test_schema_has_expected_properties_and_rejects_extra():
    schema = generation_config_schema()
    props = schema["properties"]
    assert schema["additionalProperties"] is False
    for key in ("max_shard_size", "dtype", "strategy", "n_bits", "rank", "sparsity"):
        assert key in props
    assert set(props["dtype"]["enum"]) == {"float16", "bfloat16", "float32", "float64"}
    assert "random" in props["strategy"]["enum"]


def test_validate_accepts_known_keys():
    validate_generation_dict({"strategy": "compact", "dtype": "float16", "max_shard_size": "2GB"})


def test_validate_rejects_unknown_key():
    with pytest.raises(ConfigValidationError) as ei:
        validate_generation_dict({"strategy": "compact", "bogus_key": 1})
    msg = str(ei.value)
    assert "bogus_key" in msg
    assert "unknown key" in msg


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"dtype": "float8"}, "Invalid dtype"),
        ({"max_shard_size": "5 Gigabytes"}, "Invalid max_shard_size"),
        ({"max_shard_size": "abc"}, "Invalid max_shard_size"),
        ({"rank": 0}, "rank must be a positive integer"),
        ({"rank": -3}, "rank must be a positive integer"),
        ({"strategy": "nope"}, "Invalid strategy"),
        ({"sparsity": 1.5}, "Sparsity must be in"),
        ({"n_bits": 0}, "n_bits must be in"),
    ],
)
def test_post_init_rejects_invalid_values(kwargs, match):
    with pytest.raises(ValueError, match=match):
        GenerationConfig(**kwargs)


def test_post_init_accepts_valid_dtypes_and_sizes():
    for dtype in ("float16", "bfloat16", "float32", "float64"):
        assert GenerationConfig(dtype=dtype).dtype == dtype
    for size in ("5GB", "512 MB", "1024", "2.5GB"):
        assert GenerationConfig(max_shard_size=size).max_shard_size == size


def test_build_generation_config_rejects_unknown_yaml_key(tmp_path):
    config_file = tmp_path / "vitriol.yaml"
    config_file.write_text("default:\n  strategy: compact\n  not_a_real_key: 5\n")
    with pytest.raises(ConfigValidationError, match="not_a_real_key"):
        build_generation_config(config_path=config_file)


def test_jsonschema_catches_type_errors_when_available():
    pytest.importorskip("jsonschema")
    with pytest.raises(ConfigValidationError):
        validate_generation_dict({"n_bits": "not-an-int"})
