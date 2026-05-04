"""Tests for vitriol.core.pipeline.steps.bootstrap module."""
from unittest.mock import MagicMock, patch

import pytest

from vitriol.config.manager import GenerationConfig
from vitriol.core.pipeline.context import GenerationContext
from vitriol.core.pipeline.steps.bootstrap import BootstrapStep, _parse_size


class TestParseSize:
    def test_parse_size_bytes(self):
        assert _parse_size("1024") == 1024

    def test_parse_size_kb(self):
        assert _parse_size("10KB") == 10 * 1024

    def test_parse_size_mb(self):
        assert _parse_size("5MB") == 5 * 1024 * 1024

    def test_parse_size_gb(self):
        assert _parse_size("2GB") == 2 * 1024 * 1024 * 1024

    def test_parse_size_decimal(self):
        assert _parse_size("1.5GB") == int(1.5 * 1024 * 1024 * 1024)

    def test_parse_size_invalid_string(self):
        with pytest.raises(ValueError):
            _parse_size("invalid")


class TestBootstrapStep:
    def test_bootstrap_step_name(self):
        step = BootstrapStep()
        assert step.name == "bootstrap"

    def test_bootstrap_sets_incremental(self):
        config = GenerationConfig()
        ctx = GenerationContext(
            model_id="test/model",
            output_dir="/tmp/output",
            config=config,
        )
        step = BootstrapStep()
        step.run(ctx)
        assert ctx.incremental is not None

    def test_bootstrap_sets_strategy(self):
        config = GenerationConfig(strategy="random")
        ctx = GenerationContext(
            model_id="test/model",
            output_dir="/tmp/output",
            config=config,
        )
        step = BootstrapStep()
        step.run(ctx)
        assert ctx.strategy is not None

    def test_bootstrap_sets_max_shard_size(self):
        config = GenerationConfig(max_shard_size="2GB")
        ctx = GenerationContext(
            model_id="test/model",
            output_dir="/tmp/output",
            config=config,
        )
        step = BootstrapStep()
        step.run(ctx)
        assert ctx.max_shard_size == 2 * 1024 * 1024 * 1024

    def test_bootstrap_sets_shrink_config_true(self):
        config = GenerationConfig(strategy="ultra")
        ctx = GenerationContext(
            model_id="test/model",
            output_dir="/tmp/output",
            config=config,
        )
        step = BootstrapStep()
        step.run(ctx)
        assert ctx.shrink_config is True

    def test_bootstrap_sets_shrink_config_hybrid_ultra(self):
        config = GenerationConfig(strategy="hybrid_ultra")
        ctx = GenerationContext(
            model_id="test/model",
            output_dir="/tmp/output",
            config=config,
        )
        step = BootstrapStep()
        step.run(ctx)
        assert ctx.shrink_config is True

    def test_bootstrap_sets_shrink_config_false(self):
        config = GenerationConfig(strategy="random")
        ctx = GenerationContext(
            model_id="test/model",
            output_dir="/tmp/output",
            config=config,
        )
        step = BootstrapStep()
        step.run(ctx)
        assert ctx.shrink_config is False

    def test_bootstrap_preserves_existing_shrink_config(self):
        config = GenerationConfig(strategy="random")
        ctx = GenerationContext(
            model_id="test/model",
            output_dir="/tmp/output",
            config=config,
            shrink_config=True,
        )
        step = BootstrapStep()
        step.run(ctx)
        assert ctx.shrink_config is True

    def test_bootstrap_strategy_params(self):
        config = GenerationConfig(strategy="quantized", n_bits=4, rank=8, sparsity=0.3)
        ctx = GenerationContext(
            model_id="test/model",
            output_dir="/tmp/output",
            config=config,
        )
        step = BootstrapStep()
        step.run(ctx)
        assert ctx.strategy is not None

    @patch("vitriol.core.pipeline.steps.bootstrap.IncrementalGenerator")
    @patch("vitriol.core.pipeline.steps.bootstrap.get_strategy")
    def test_bootstrap_calls_get_strategy(self, mock_get_strategy, mock_incr_gen):
        mock_strategy = MagicMock()
        mock_get_strategy.return_value = mock_strategy

        config = GenerationConfig(strategy="random", n_bits=8, rank=16, sparsity=0.5)
        ctx = GenerationContext(
            model_id="test/model",
            output_dir="/tmp/output",
            config=config,
        )
        step = BootstrapStep()
        step.run(ctx)

        mock_get_strategy.assert_called_once_with(
            "random", n_bits=8, rank=16, sparsity=0.5
        )

    @patch("vitriol.core.pipeline.steps.bootstrap.IncrementalGenerator")
    def test_bootstrap_creates_incremental_generator(self, mock_incr_gen):
        mock_incr = MagicMock()
        mock_incr_gen.return_value = mock_incr

        config = GenerationConfig()
        ctx = GenerationContext(
            model_id="test/model",
            output_dir="/tmp/output",
            config=config,
        )
        step = BootstrapStep()
        step.run(ctx)

        mock_incr_gen.assert_called_once_with("/tmp/output")
