"""Tests for vitriol.core.shard_manager module."""
from unittest.mock import MagicMock, patch

import pytest

from vitriol.core.shard_manager import ShardManager


class TestShardManagerInit:
    def test_default_max_shard_size(self):
        manager = ShardManager()
        assert manager.max_shard_size == 5 * 1024 ** 3

    def test_custom_max_shard_size_gb(self):
        manager = ShardManager("2GB")
        assert manager.max_shard_size == 2 * 1024 ** 3

    def test_custom_max_shard_size_mb(self):
        manager = ShardManager("500MB")
        assert manager.max_shard_size == 500 * 1024 ** 2

    def test_custom_max_shard_size_kb(self):
        manager = ShardManager("1024KB")
        assert manager.max_shard_size == 1024 * 1024

    def test_custom_max_shard_size_bytes(self):
        manager = ShardManager("1048576")
        assert manager.max_shard_size == 1048576

    def test_parse_size_invalid(self):
        with pytest.raises(ValueError):
            ShardManager._parse_size("invalid")

    def test_parse_size_empty(self):
        with pytest.raises(ValueError):
            ShardManager._parse_size("")

    def test_parse_size_with_spaces(self):
        assert ShardManager._parse_size("  5 GB  ") == 5 * 1024 ** 3


class TestShardManagerPlanShards:
    def test_empty_params(self):
        manager = ShardManager("1GB")
        result = list(manager.plan_shards([], {}))
        assert result == []

    def test_single_param_fits(self):
        manager = ShardManager("1GB")
        result = list(manager.plan_shards(["layer1"], {"layer1": 100}))
        assert len(result) == 1
        assert result[0][0] == "pytorch_model-00001-of-{total:05d}.bin"
        assert result[0][1] == {"layer1": 100}

    def test_single_param_exceeds(self):
        manager = ShardManager("1GB")
        result = list(manager.plan_shards(["layer1"], {"layer1": 2 * 1024 ** 3}))
        assert len(result) == 1
        assert result[0][1] == {"layer1": 2 * 1024 ** 3}

    def test_multiple_params_single_shard(self):
        manager = ShardManager("1GB")
        result = list(manager.plan_shards(
            ["layer1", "layer2"],
            {"layer1": 100, "layer2": 200}
        ))
        assert len(result) == 1
        assert result[0][1] == {"layer1": 100, "layer2": 200}

    def test_multiple_params_multiple_shards(self):
        manager = ShardManager("1GB")
        result = list(manager.plan_shards(
            ["layer1", "layer2"],
            {"layer1": 600 * 1024 ** 2, "layer2": 600 * 1024 ** 2}
        ))
        assert len(result) == 2

    def test_exact_boundary(self):
        manager = ShardManager("1GB")
        result = list(manager.plan_shards(
            ["layer1", "layer2"],
            {"layer1": 1024 ** 3, "layer2": 1}
        ))
        assert len(result) == 2
        assert result[0][1] == {"layer1": 1024 ** 3}
        assert result[1][1] == {"layer2": 1}

    def test_shard_index_increment(self):
        manager = ShardManager("1GB")
        result = list(manager.plan_shards(
            ["a", "b", "c"],
            {"a": 600 * 1024 ** 2, "b": 600 * 1024 ** 2, "c": 600 * 1024 ** 2}
        ))
        assert len(result) == 3
        assert result[0][0] == "pytorch_model-00001-of-{total:05d}.bin"
        assert result[1][0] == "pytorch_model-00002-of-{total:05d}.bin"
        assert result[2][0] == "pytorch_model-00003-of-{total:05d}.bin"

    def test_param_size_zero(self):
        manager = ShardManager("1GB")
        result = list(manager.plan_shards(
            ["layer1", "layer2"],
            {"layer1": 0, "layer2": 100}
        ))
        assert len(result) == 1
        assert result[0][1] == {"layer1": 0, "layer2": 100}

    def test_param_not_in_sizes(self):
        manager = ShardManager("1GB")
        result = list(manager.plan_shards(
            ["layer1", "layer2"],
            {"layer1": 100}
        ))
        assert len(result) == 1
        assert result[0][1] == {"layer1": 100, "layer2": 0}


class TestShardManagerFormatShardName:
    def test_format_pytorch(self):
        manager = ShardManager()
        name = manager._format_shard_name(0, format="pytorch")
        assert "pytorch_model" in name
        assert name.endswith(".bin")

    def test_format_safetensors(self):
        manager = ShardManager()
        name = manager._format_shard_name(0, format="safetensors")
        assert "model" in name
        assert name.endswith(".safetensors")

    def test_format_shard_index(self):
        manager = ShardManager()
        name = manager._format_shard_name(5, format="pytorch")
        assert "00006" in name


class TestShardManagerEstimateTotalShards:
    def test_estimate_empty(self):
        manager = ShardManager("1GB")
        assert manager.estimate_total_shards([], {}) == 0

    def test_estimate_single_shard(self):
        manager = ShardManager("1GB")
        assert manager.estimate_total_shards(["layer1"], {"layer1": 100}) == 1

    def test_estimate_multiple_shards(self):
        manager = ShardManager("1GB")
        count = manager.estimate_total_shards(
            ["a", "b", "c"],
            {"a": 600 * 1024 ** 2, "b": 600 * 1024 ** 2, "c": 600 * 1024 ** 2}
        )
        assert count == 3


class TestShardManagerGetShardIndexMap:
    def test_index_map_empty(self):
        manager = ShardManager("1GB")
        assert manager.get_shard_index_map([], {}) == {}

    def test_index_map_single(self):
        manager = ShardManager("1GB")
        result = manager.get_shard_index_map(["layer1"], {"layer1": 100})
        assert "layer1" in result
        assert "pytorch_model" in result["layer1"]

    def test_index_map_multiple(self):
        manager = ShardManager("1GB")
        result = manager.get_shard_index_map(
            ["a", "b"],
            {"a": 600 * 1024 ** 2, "b": 600 * 1024 ** 2}
        )
        assert result["a"] != result["b"]


class TestShardManagerPlanShardsFromModel:
    def test_plan_from_model(self):
        manager = ShardManager("1GB")
        mock_param = MagicMock()
        mock_param.numel.return_value = 100
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = [("layer1", mock_param)]

        result = list(manager.plan_shards_from_model(mock_model, dtype_size=2))
        assert len(result) == 1
        assert result[0][1] == {"layer1": 200}

    def test_plan_from_model_default_dtype(self):
        manager = ShardManager("1GB")
        mock_param = MagicMock()
        mock_param.numel.return_value = 100
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = [("layer1", mock_param)]

        result = list(manager.plan_shards_from_model(mock_model))
        assert len(result) == 1
        assert result[0][1] == {"layer1": 200}

    def test_plan_from_model_empty(self):
        manager = ShardManager("1GB")
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = []

        result = list(manager.plan_shards_from_model(mock_model))
        assert result == []


class TestShardManagerLogging:
    @patch("vitriol.core.shard_manager.logger")
    def test_init_logs(self, mock_logger):
        ShardManager("2GB")
        mock_logger.info.assert_called_once()

    @patch("vitriol.core.shard_manager.logger")
    def test_plan_shards_logs(self, mock_logger):
        manager = ShardManager("1GB")
        list(manager.plan_shards(
            ["layer1"],
            {"layer1": 100}
        ))
        mock_logger.debug.assert_called()
