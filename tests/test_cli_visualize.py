"""Tests for vitriol.cli.commands.visualize module."""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from vitriol.cli.main import cli


class TestVisualizeCommandHelp:
    def test_visualize_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["visualize", "--help"])
        assert result.exit_code == 0
        assert "visualize" in result.output.lower()


class TestVisualizeCommandMocked:
    @patch("vitriol.visualization.utils.load_weights")
    @patch("vitriol.visualization.visualizer.WeightVisualizer")
    def test_visualize_default(self, mock_viz_cls, mock_load_weights):
        runner = CliRunner()
        mock_weights = {"layer1": MagicMock()}
        mock_load_weights.return_value = mock_weights
        mock_viz = MagicMock()
        mock_viz_cls.return_value = mock_viz

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, ["visualize", tmpdir])
            assert result.exit_code == 0
            mock_load_weights.assert_called_once_with(tmpdir, pattern=None, limit=None)
            mock_viz.generate_comprehensive_report.assert_called_once()

    @patch("vitriol.visualization.utils.load_weights")
    @patch("vitriol.visualization.visualizer.WeightVisualizer")
    def test_visualize_with_output_dir(self, mock_viz_cls, mock_load_weights):
        runner = CliRunner()
        mock_weights = {"layer1": MagicMock()}
        mock_load_weights.return_value = mock_weights
        mock_viz = MagicMock()
        mock_viz_cls.return_value = mock_viz

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = str(Path(tmpdir) / "viz")
            result = runner.invoke(cli, ["visualize", tmpdir, "--output-dir", out_dir])
            assert result.exit_code == 0
            mock_viz.generate_comprehensive_report.assert_called_once_with(mock_weights, out_dir)

    @patch("vitriol.visualization.utils.load_weights")
    @patch("vitriol.visualization.visualizer.WeightVisualizer")
    def test_visualize_with_layer_pattern(self, mock_viz_cls, mock_load_weights):
        runner = CliRunner()
        mock_weights = {"layer1": MagicMock()}
        mock_load_weights.return_value = mock_weights
        mock_viz = MagicMock()
        mock_viz_cls.return_value = mock_viz

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, ["visualize", tmpdir, "--layer-pattern", "layers.0"])
            assert result.exit_code == 0
            mock_load_weights.assert_called_once_with(tmpdir, pattern="layers.0", limit=None)

    @patch("vitriol.visualization.utils.load_weights")
    @patch("vitriol.visualization.visualizer.WeightVisualizer")
    def test_visualize_with_limit(self, mock_viz_cls, mock_load_weights):
        runner = CliRunner()
        mock_weights = {"layer1": MagicMock()}
        mock_load_weights.return_value = mock_weights
        mock_viz = MagicMock()
        mock_viz_cls.return_value = mock_viz

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, ["visualize", tmpdir, "--limit", "10"])
            assert result.exit_code == 0
            mock_load_weights.assert_called_once_with(tmpdir, pattern=None, limit=10)

    @patch("vitriol.visualization.utils.load_weights")
    def test_visualize_no_weights(self, mock_load_weights):
        runner = CliRunner()
        mock_load_weights.return_value = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, ["visualize", tmpdir])
            assert result.exit_code == 1
            assert "No weights loaded" in result.output

    @patch("vitriol.visualization.utils.load_weights")
    @patch("vitriol.visualization.visualizer.WeightVisualizer")
    def test_visualize_report_exception(self, mock_viz_cls, mock_load_weights):
        runner = CliRunner()
        mock_weights = {"layer1": MagicMock()}
        mock_load_weights.return_value = mock_weights
        mock_viz = MagicMock()
        mock_viz.generate_comprehensive_report.side_effect = RuntimeError("Report failed")
        mock_viz_cls.return_value = mock_viz

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, ["visualize", tmpdir])
            assert result.exit_code == 1
            assert "Visualization failed" in result.output


class TestVisualizeMissingDependency:
    @patch("vitriol.visualization.utils.load_weights", side_effect=ModuleNotFoundError("No module named 'numpy'"))
    def test_visualize_missing_dependency(self, mock_load_weights):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, ["visualize", tmpdir])
            assert result.exit_code != 0
            # The exception should be a ClickException wrapping the ModuleNotFoundError
            assert isinstance(result.exception, Exception)