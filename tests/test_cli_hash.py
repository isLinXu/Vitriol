"""Tests for vitriol.cli.commands.hash module."""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from vitriol.cli.main import cli


class TestHashCommandHelp:
    def test_hash_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["hash", "--help"])
        assert result.exit_code == 0
        assert "hash" in result.output.lower()


class TestHashCommandMocked:
    @patch("vitriol.core.hasher.ModelHasher")
    @patch("rich.console.Console")
    def test_hash_fast(self, mock_console_cls, mock_hasher_cls):
        runner = CliRunner()
        mock_hasher = MagicMock()
        mock_hasher.compute_architecture_hash.return_value = "arch123"
        mock_hasher.compute_activation_signature_hash.return_value = "beh456"
        mock_hasher_cls.return_value = mock_hasher
        mock_console_cls.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            result = runner.invoke(cli, ["hash", str(model_dir), "--fast"])
            assert result.exit_code == 0
            mock_hasher.compute_architecture_hash.assert_called_once()
            mock_hasher.compute_activation_signature_hash.assert_called_once()
            mock_hasher.compute_weight_distribution_hash.assert_not_called()

    @patch("vitriol.core.hasher.ModelHasher")
    @patch("rich.console.Console")
    def test_hash_full(self, mock_console_cls, mock_hasher_cls):
        runner = CliRunner()
        mock_hasher = MagicMock()
        mock_hasher.compute_architecture_hash.return_value = "arch123"
        mock_hasher.compute_activation_signature_hash.return_value = "beh456"
        mock_hasher.compute_weight_distribution_hash.return_value = "weight789"
        mock_hasher_cls.return_value = mock_hasher
        mock_console_cls.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            result = runner.invoke(cli, ["hash", str(model_dir)])
            assert result.exit_code == 0
            mock_hasher.compute_architecture_hash.assert_called_once()
            mock_hasher.compute_activation_signature_hash.assert_called_once()
            mock_hasher.compute_weight_distribution_hash.assert_called_once_with(max_tensors=50)

    @patch("vitriol.core.hasher.ModelHasher")
    @patch("rich.console.Console")
    def test_hash_combined_signature(self, mock_console_cls, mock_hasher_cls):
        runner = CliRunner()
        mock_hasher = MagicMock()
        mock_hasher.compute_architecture_hash.return_value = "arch123"
        mock_hasher.compute_activation_signature_hash.return_value = "beh456"
        mock_hasher.compute_weight_distribution_hash.return_value = "weight789"
        mock_hasher_cls.return_value = mock_hasher
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            result = runner.invoke(cli, ["hash", str(model_dir)])
            assert result.exit_code == 0
            # Verify the console was called to print the signature
            print_calls = [call for call in mock_console.method_calls if "print" in str(call)]
            assert len(print_calls) > 0

    @patch("vitriol.core.hasher.ModelHasher")
    @patch("rich.console.Console")
    def test_hash_skipped_signature(self, mock_console_cls, mock_hasher_cls):
        runner = CliRunner()
        mock_hasher = MagicMock()
        mock_hasher.compute_architecture_hash.return_value = "N/A"
        mock_hasher.compute_activation_signature_hash.return_value = "beh456"
        mock_hasher.compute_weight_distribution_hash.return_value = "N/A"
        mock_hasher_cls.return_value = mock_hasher
        mock_console_cls.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            result = runner.invoke(cli, ["hash", str(model_dir)])
            assert result.exit_code == 0


class TestHashCommandErrors:
    @patch("vitriol.core.hasher.ModelHasher")
    @patch("rich.console.Console")
    def test_hash_hasher_exception(self, mock_console_cls, mock_hasher_cls):
        runner = CliRunner()
        mock_hasher = MagicMock()
        mock_hasher.compute_architecture_hash.side_effect = RuntimeError("Hash failed")
        mock_hasher_cls.return_value = mock_hasher
        mock_console_cls.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            result = runner.invoke(cli, ["hash", str(model_dir), "--fast"])
            assert result.exit_code != 0
            assert "Hash failed" in str(result.exception)


class TestHashCommandEdgeCases:
    @patch("vitriol.core.hasher.ModelHasher")
    @patch("rich.console.Console")
    def test_hash_with_empty_model_dir(self, mock_console_cls, mock_hasher_cls):
        runner = CliRunner()
        mock_hasher = MagicMock()
        mock_hasher.compute_architecture_hash.return_value = "arch123"
        mock_hasher.compute_activation_signature_hash.return_value = "beh456"
        mock_hasher_cls.return_value = mock_hasher
        mock_console_cls.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            result = runner.invoke(cli, ["hash", str(model_dir), "--fast"])
            assert result.exit_code == 0

    def test_hash_nonexistent_path(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["hash", "/nonexistent/path"])
        assert result.exit_code != 0
