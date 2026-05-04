"""Tests for vitriol.core.parallel_generator module."""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from vitriol.core.parallel_generator import ParallelWeightGenerator, StreamingWeightGenerator


class MockStrategy:
    """Mock strategy for testing."""

    def __init__(self):
        self.call_count = 0

    def generate_tensor(self, shape, dtype, name):
        self.call_count += 1
        return torch.zeros(shape, dtype=dtype)

    def save_shard(self, data, path):
        pass


class TestParallelWeightGenerator:
    """Tests for ParallelWeightGenerator class."""

    def test_init_default_workers(self):
        """Test initialization with default workers."""
        strategy = MockStrategy()
        gen = ParallelWeightGenerator(strategy)
        assert gen.strategy == strategy
        assert gen.n_workers >= 1
        assert gen.use_processes is False

    def test_init_custom_workers(self):
        """Test initialization with custom workers."""
        strategy = MockStrategy()
        gen = ParallelWeightGenerator(strategy, n_workers=2, use_processes=True)
        assert gen.n_workers == 2
        assert gen.use_processes is True

    def test_generate_shard_parallel(self):
        """Test parallel shard generation."""
        strategy = MockStrategy()
        gen = ParallelWeightGenerator(strategy, n_workers=2)

        param_names = ["w1", "w2"]
        param_shapes = {"w1": (10, 10), "w2": (5, 5)}
        param_dtypes = {"w1": torch.float32, "w2": torch.float32}

        results = gen.generate_shard_parallel(
            param_names, param_shapes, param_dtypes, show_progress=False
        )

        assert len(results) == 2
        assert "w1" in results
        assert "w2" in results
        assert results["w1"].shape == (10, 10)
        assert results["w2"].shape == (5, 5)

    def test_generate_shard_parallel_with_progress(self):
        """Test parallel generation with progress bar."""
        strategy = MockStrategy()
        gen = ParallelWeightGenerator(strategy, n_workers=1)

        param_names = ["w1"]
        param_shapes = {"w1": (3, 3)}
        param_dtypes = {"w1": torch.float32}

        with patch("vitriol.core.parallel_generator.tqdm") as mock_tqdm:
            mock_pbar = Mock()
            mock_tqdm.return_value = mock_pbar
            results = gen.generate_shard_parallel(
                param_names, param_shapes, param_dtypes, show_progress=True
            )
            mock_tqdm.assert_called_once()
            mock_pbar.close.assert_called_once()

    def test_generate_shard_parallel_empty(self):
        """Test parallel generation with empty parameters."""
        strategy = MockStrategy()
        gen = ParallelWeightGenerator(strategy)

        results = gen.generate_shard_parallel([], {}, {}, show_progress=False)
        assert results == {}

    def test_generate_single_tensor(self):
        """Test _generate_single_tensor wrapper."""
        strategy = MockStrategy()
        gen = ParallelWeightGenerator(strategy)

        result = gen._generate_single_tensor("test", (4, 4), torch.float32)
        assert result.shape == (4, 4)
        assert strategy.call_count == 1

    def test_generate_batch_parallel(self):
        """Test batch parallel generation."""
        strategy = MockStrategy()
        gen = ParallelWeightGenerator(strategy, n_workers=2)

        batches = [
            {"w1": {"shape": (2, 2), "dtype": torch.float32}},
            {"w2": {"shape": (3, 3), "dtype": torch.float32}}
        ]

        results = gen.generate_batch_parallel(batches, show_progress=False)
        assert len(results) == 2

    def test_generate_batch_parallel_empty(self):
        """Test batch parallel with empty batches."""
        strategy = MockStrategy()
        gen = ParallelWeightGenerator(strategy)

        results = gen.generate_batch_parallel([], show_progress=False)
        assert results == []

    def test_generate_batch_parallel_single_batch(self):
        """Test batch parallel with single batch."""
        strategy = MockStrategy()
        gen = ParallelWeightGenerator(strategy, n_workers=1)

        batches = [
            {"w1": {"shape": (2, 2), "dtype": torch.float32}}
        ]

        results = gen.generate_batch_parallel(batches, show_progress=False)
        assert len(results) == 1


class TestStreamingWeightGenerator:
    """Tests for StreamingWeightGenerator class."""

    def test_init_with_parallel(self):
        """Test initialization with parallel generator."""
        strategy = MockStrategy()
        shard_manager = Mock()
        gen = StreamingWeightGenerator(strategy, shard_manager, n_workers=4)
        assert gen.strategy == strategy
        assert gen.shard_manager == shard_manager
        assert gen.parallel_gen is not None
        assert gen.parallel_gen.n_workers == 4

    def test_init_without_parallel(self):
        """Test initialization without parallel generator."""
        strategy = MockStrategy()
        shard_manager = Mock()
        gen = StreamingWeightGenerator(strategy, shard_manager)
        assert gen.parallel_gen is None

    def test_generate_streaming(self, tmp_path):
        """Test streaming weight generation."""
        strategy = MockStrategy()
        shard_manager = Mock()
        shard_manager.plan_shards.return_value = [
            ("shard1.bin", {"w1": 100, "w2": 200})
        ]

        gen = StreamingWeightGenerator(strategy, shard_manager)

        model = Mock()
        param1 = Mock()
        param1.shape = (10, 10)
        param1.dtype = torch.float32
        param1.numel.return_value = 100
        param2 = Mock()
        param2.shape = (20, 10)
        param2.dtype = torch.float32
        param2.numel.return_value = 200
        model.named_parameters.return_value = [("w1", param1), ("w2", param2)]

        output_dir = tmp_path / "output"
        result = gen.generate_streaming(model, str(output_dir), show_progress=False)

        assert len(result) == 1
        assert result[0] == "shard1.bin"
        assert output_dir.exists()
        shard_manager.plan_shards.assert_called_once()

    def test_get_param_info(self):
        """Test _get_param_info extraction."""
        strategy = MockStrategy()
        shard_manager = Mock()
        gen = StreamingWeightGenerator(strategy, shard_manager)

        model = Mock()
        param = Mock()
        param.shape = (5, 5)
        param.dtype = torch.float32
        param.numel.return_value = 25
        model.named_parameters.return_value = [("layer.weight", param)]

        info = gen._get_param_info(model)
        assert info["names"] == ["layer.weight"]
        assert info["shapes"]["layer.weight"] == (5, 5)
        assert info["dtypes"]["layer.weight"] == torch.float32
        assert info["sizes"]["layer.weight"] == 50  # 25 * 2

    def test_generate_shard_data_parallel(self):
        """Test _generate_shard_data with parallel generator."""
        strategy = MockStrategy()
        shard_manager = Mock()
        gen = StreamingWeightGenerator(strategy, shard_manager, n_workers=2)

        param_batch = {"w1": 100, "w2": 200}
        shapes = {"w1": (10, 10), "w2": (20, 10)}
        dtypes = {"w1": torch.float32, "w2": torch.float32}

        result = gen._generate_shard_data(param_batch, shapes, dtypes)
        assert len(result) == 2
        assert "w1" in result
        assert "w2" in result

    def test_generate_shard_data_sequential(self):
        """Test _generate_shard_data without parallel generator."""
        strategy = MockStrategy()
        shard_manager = Mock()
        gen = StreamingWeightGenerator(strategy, shard_manager)

        param_batch = {"w1": 100}
        shapes = {"w1": (10, 10)}
        dtypes = {"w1": torch.float32}

        result = gen._generate_shard_data(param_batch, shapes, dtypes)
        assert len(result) == 1
        assert result["w1"].shape == (10, 10)

    def test_generate_streaming_empty_model(self, tmp_path):
        """Test streaming generation with empty model."""
        strategy = MockStrategy()
        shard_manager = Mock()
        shard_manager.plan_shards.return_value = []

        gen = StreamingWeightGenerator(strategy, shard_manager)

        model = Mock()
        model.named_parameters.return_value = []

        output_dir = tmp_path / "output"
        result = gen.generate_streaming(model, str(output_dir), show_progress=False)
        assert result == []
