"""Tests for vitriol.core.pipeline.steps.legacy_generate module."""
from unittest.mock import MagicMock

import pytest

from vitriol.config.manager import GenerationConfig
from vitriol.core.pipeline.context import GenerationContext
from vitriol.core.pipeline.steps.legacy_generate import LegacyGenerateStep


class TestLegacyGenerateStep:
    def test_step_name(self):
        step = LegacyGenerateStep()
        assert step.name == "legacy_generate"

    def test_run_with_generator(self):
        config = GenerationConfig()
        ctx = GenerationContext(
            model_id="test/model",
            output_dir="/tmp/output",
            config=config,
        )
        mock_generator = MagicMock()
        ctx.generator = mock_generator

        step = LegacyGenerateStep()
        step.run(ctx)

        mock_generator._generate_legacy_impl.assert_called_once()

    def test_run_without_generator_raises(self):
        config = GenerationConfig()
        ctx = GenerationContext(
            model_id="test/model",
            output_dir="/tmp/output",
            config=config,
        )
        # ctx.generator is None by default

        step = LegacyGenerateStep()
        with pytest.raises(RuntimeError, match="GenerationContext.generator is required"):
            step.run(ctx)

    def test_run_preserves_context(self):
        config = GenerationConfig()
        ctx = GenerationContext(
            model_id="test/model",
            output_dir="/tmp/output",
            config=config,
        )
        mock_generator = MagicMock()
        ctx.generator = mock_generator
        ctx.total_size = 42

        step = LegacyGenerateStep()
        step.run(ctx)

        assert ctx.total_size == 42
        mock_generator._generate_legacy_impl.assert_called_once()

    def test_run_generator_exception_propagates(self):
        config = GenerationConfig()
        ctx = GenerationContext(
            model_id="test/model",
            output_dir="/tmp/output",
            config=config,
        )
        mock_generator = MagicMock()
        mock_generator._generate_legacy_impl.side_effect = RuntimeError("Generation failed")
        ctx.generator = mock_generator

        step = LegacyGenerateStep()
        with pytest.raises(RuntimeError, match="Generation failed"):
            step.run(ctx)
