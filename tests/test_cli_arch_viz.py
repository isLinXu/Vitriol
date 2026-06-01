"""Tests for vitriol.cli.commands.arch_viz module."""
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from vitriol.cli.main import cli


class TestArchVizCommandHelp:
    def test_arch_viz_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["arch-viz", "--help"])
        assert result.exit_code == 0
        assert "arch-viz" in result.output.lower() or "architecture" in result.output.lower()


class TestArchVizCommandMocked:
    @patch("vitriol.arch_viz.visualizer.ArchitectureVisualizer")
    def test_arch_viz_default_block(self, mock_viz_cls):
        runner = CliRunner()
        mock_viz = MagicMock()
        mock_viz_cls.return_value = mock_viz

        result = runner.invoke(cli, ["arch-viz", "test/model"])
        assert result.exit_code == 0
        mock_viz.generate_block_diagram.assert_called_once_with("architecture_block.png")
        mock_viz.generate_detailed_diagram.assert_not_called()
        mock_viz.generate_interactive_html.assert_not_called()
        mock_viz.generate_all.assert_not_called()

    @patch("vitriol.arch_viz.visualizer.ArchitectureVisualizer")
    def test_arch_viz_block_flag(self, mock_viz_cls):
        runner = CliRunner()
        mock_viz = MagicMock()
        mock_viz_cls.return_value = mock_viz

        result = runner.invoke(cli, ["arch-viz", "test/model", "--block"])
        assert result.exit_code == 0
        mock_viz.generate_block_diagram.assert_called_once_with("architecture_block.png")

    @patch("vitriol.arch_viz.visualizer.ArchitectureVisualizer")
    def test_arch_viz_detail_flag(self, mock_viz_cls):
        runner = CliRunner()
        mock_viz = MagicMock()
        mock_viz_cls.return_value = mock_viz

        result = runner.invoke(cli, ["arch-viz", "test/model", "--detail"])
        assert result.exit_code == 0
        mock_viz.generate_detailed_diagram.assert_called_once_with("architecture_detail.png")

    @patch("vitriol.arch_viz.visualizer.ArchitectureVisualizer")
    def test_arch_viz_html_flag(self, mock_viz_cls):
        runner = CliRunner()
        mock_viz = MagicMock()
        mock_viz_cls.return_value = mock_viz

        result = runner.invoke(cli, ["arch-viz", "test/model", "--html"])
        assert result.exit_code == 0
        mock_viz.generate_interactive_html.assert_called_once_with("architecture.html")

    @patch("vitriol.arch_viz.visualizer.ArchitectureVisualizer")
    def test_arch_viz_all_flags(self, mock_viz_cls):
        runner = CliRunner()
        mock_viz = MagicMock()
        mock_viz_cls.return_value = mock_viz

        result = runner.invoke(cli, ["arch-viz", "test/model", "--block", "--detail", "--html"])
        assert result.exit_code == 0
        mock_viz.generate_block_diagram.assert_called_once()
        mock_viz.generate_detailed_diagram.assert_called_once()
        mock_viz.generate_interactive_html.assert_called_once()

    @patch("vitriol.arch_viz.visualizer.ArchitectureVisualizer")
    def test_arch_viz_custom_output(self, mock_viz_cls):
        runner = CliRunner()
        mock_viz = MagicMock()
        mock_viz_cls.return_value = mock_viz

        result = runner.invoke(cli, ["arch-viz", "test/model", "--block", "--output", "custom.png"])
        assert result.exit_code == 0
        mock_viz.generate_block_diagram.assert_called_once_with("custom.png")

    @patch("vitriol.arch_viz.visualizer.ArchitectureVisualizer")
    def test_arch_viz_generate_all(self, mock_viz_cls):
        runner = CliRunner()
        mock_viz = MagicMock()
        mock_viz_cls.return_value = mock_viz

        result = runner.invoke(cli, ["arch-viz", "test/model", "--all"])
        assert result.exit_code == 0
        mock_viz.generate_all.assert_called_once()

    @patch("vitriol.arch_viz.visualizer.ArchitectureVisualizer")
    def test_arch_viz_generate_all_with_output(self, mock_viz_cls):
        runner = CliRunner()
        mock_viz = MagicMock()
        mock_viz_cls.return_value = mock_viz

        result = runner.invoke(cli, ["arch-viz", "test/model", "--all", "--output", "viz_dir"])
        assert result.exit_code == 0
        mock_viz.generate_all.assert_called_once_with("viz_dir")

    @patch("vitriol.arch_viz.visualizer.ArchitectureVisualizer")
    def test_arch_viz_with_style(self, mock_viz_cls):
        runner = CliRunner()
        mock_viz = MagicMock()
        mock_viz_cls.return_value = mock_viz

        result = runner.invoke(cli, ["--trust-remote-code", "arch-viz", "test/model", "--style", "dark"])
        assert result.exit_code == 0
        mock_viz_cls.assert_called_once_with("test/model", style="dark", trust_remote_code=True)

    @patch("vitriol.arch_viz.visualizer.ArchitectureVisualizer")
    def test_arch_viz_trust_remote_code_default(self, mock_viz_cls):
        runner = CliRunner()
        mock_viz = MagicMock()
        mock_viz_cls.return_value = mock_viz

        result = runner.invoke(cli, ["arch-viz", "test/model"])
        assert result.exit_code == 0
        mock_viz_cls.assert_called_once_with("test/model", style="default", trust_remote_code=False)

    @patch("vitriol.arch_viz.visualizer.ArchitectureVisualizer")
    def test_arch_viz_offline_forwards_local_files_only(self, mock_viz_cls):
        runner = CliRunner()
        mock_viz = MagicMock()
        mock_viz_cls.return_value = mock_viz

        result = runner.invoke(cli, ["--offline", "arch-viz", "test/model"])
        assert result.exit_code == 0
        mock_viz_cls.assert_called_once_with(
            "test/model",
            style="default",
            trust_remote_code=False,
            local_files_only=True,
        )


class TestArchVizCommandErrors:
    @patch("vitriol.arch_viz.visualizer.ArchitectureVisualizer")
    def test_arch_viz_exception(self, mock_viz_cls):
        runner = CliRunner()
        mock_viz_cls.side_effect = RuntimeError("Visualization failed")

        result = runner.invoke(cli, ["arch-viz", "test/model"])
        assert result.exit_code != 0
        assert "Error" in result.output or "Visualization failed" in result.output
