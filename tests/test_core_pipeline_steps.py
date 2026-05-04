"""Tests for vitriol.core.pipeline.steps module."""
from vitriol.core.pipeline.steps import BootstrapStep, LegacyGenerateStep


class TestPipelineStepsImports:
    def test_bootstrap_step_importable(self):
        assert BootstrapStep is not None
        assert BootstrapStep.name == "bootstrap"

    def test_legacy_generate_step_importable(self):
        assert LegacyGenerateStep is not None
        assert LegacyGenerateStep.name == "legacy_generate"
