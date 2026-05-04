"""Tests for CLI export command."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from click.testing import CliRunner

from vitriol.cli.commands.export import export


class TestExportCommand:
    """Tests for export CLI command."""

    @patch("vitriol.core.exporter.ModelExporter")
    def test_export_json(self, mock_exporter_class):
        runner = CliRunner()
        mock_exporter = MagicMock()
        mock_exporter_class.return_value = mock_exporter

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "model"
            input_dir.mkdir()
            output_path = Path(tmpdir) / "structure.json"

            result = runner.invoke(export, [str(input_dir), "-o", str(output_path)])
            assert result.exit_code == 0
            mock_exporter_class.assert_called_once_with(str(input_dir))
            mock_exporter.export_structure.assert_called_once_with(str(output_path))

    @patch("vitriol.core.exporter.ModelExporter")
    def test_export_gguf(self, mock_exporter_class):
        runner = CliRunner()
        mock_exporter = MagicMock()
        mock_exporter_class.return_value = mock_exporter

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "model"
            input_dir.mkdir()
            output_path = Path(tmpdir) / "model.gguf"

            result = runner.invoke(export, [
                str(input_dir), "-o", str(output_path), "--format", "gguf"
            ])
            assert result.exit_code == 0
            mock_exporter.export_gguf_prep.assert_called_once_with(str(output_path))

    @patch("vitriol.core.exporter.ModelExporter")
    def test_export_default_format(self, mock_exporter_class):
        runner = CliRunner()
        mock_exporter = MagicMock()
        mock_exporter_class.return_value = mock_exporter

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "model"
            input_dir.mkdir()
            output_path = Path(tmpdir) / "out.json"

            result = runner.invoke(export, [str(input_dir), "-o", str(output_path)])
            assert result.exit_code == 0
            mock_exporter.export_structure.assert_called_once()

    def test_export_missing_input(self):
        runner = CliRunner()
        result = runner.invoke(export, ["/nonexistent/path", "-o", "out.json"])
        assert result.exit_code != 0
        assert "does not exist" in result.output or "Invalid value" in result.output

    def test_export_missing_output(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "model"
            input_dir.mkdir()
            result = runner.invoke(export, [str(input_dir)])
            assert result.exit_code != 0
            assert "required" in result.output.lower() or "Missing option" in result.output
