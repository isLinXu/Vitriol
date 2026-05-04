"""Tests for vitriol.cli.commands.batch module."""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from vitriol.cli.main import cli


class TestBatchCommandHelp:
    def test_batch_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["batch", "--help"])
        assert result.exit_code == 0
        assert "batch" in result.output.lower()


class TestBatchCommandMocked:
    @patch("vitriol.core.batch.BatchGenerator")
    def test_batch_success(self, mock_generator_cls):
        runner = CliRunner()
        mock_generator = MagicMock()
        mock_generator_cls.return_value = mock_generator

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text("models:\n  - model1\n")
            result = runner.invoke(cli, ["batch", str(config_file)])
            assert result.exit_code == 0
            mock_generator_cls.assert_called_once_with(str(config_file))
            mock_generator.generate_all.assert_called_once()

    @patch("vitriol.core.batch.BatchGenerator")
    def test_batch_failure(self, mock_generator_cls):
        runner = CliRunner()
        mock_generator = MagicMock()
        mock_generator.generate_all.side_effect = RuntimeError("Generation failed")
        mock_generator_cls.return_value = mock_generator

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text("models:\n  - model1\n")
            result = runner.invoke(cli, ["batch", str(config_file)])
            assert result.exit_code == 1
            assert "Batch generation failed" in result.output

    def test_batch_nonexistent_config(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["batch", "/nonexistent/config.yaml"])
        assert result.exit_code != 0


class TestBatchCommandEdgeCases:
    @patch("vitriol.core.batch.BatchGenerator")
    def test_batch_empty_config(self, mock_generator_cls):
        runner = CliRunner()
        mock_generator = MagicMock()
        mock_generator_cls.return_value = mock_generator

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text("")
            result = runner.invoke(cli, ["batch", str(config_file)])
            assert result.exit_code == 0
            mock_generator.generate_all.assert_called_once()
