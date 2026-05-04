"""Tests for vitriol.cli.commands.webui module."""
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from vitriol.cli.main import cli


class TestWebuiCommandHelp:
    def test_webui_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["webui", "--help"])
        assert result.exit_code == 0
        assert "webui" in result.output.lower() or "Web UI" in result.output


class TestWebuiCommandMocked:
    @patch("vitriol.cli.commands.webui._load_webui_launch")
    def test_webui_default_port(self, mock_load_launch):
        runner = CliRunner()
        mock_launch = MagicMock()
        mock_load_launch.return_value = mock_launch

        result = runner.invoke(cli, ["webui"])
        assert result.exit_code == 0
        mock_launch.assert_called_once_with(share=False, port=7860, debug=False)

    @patch("vitriol.cli.commands.webui._load_webui_launch")
    def test_webui_custom_port(self, mock_load_launch):
        runner = CliRunner()
        mock_launch = MagicMock()
        mock_load_launch.return_value = mock_launch

        result = runner.invoke(cli, ["webui", "--port", "8080"])
        assert result.exit_code == 0
        mock_launch.assert_called_once_with(share=False, port=8080, debug=False)

    @patch("vitriol.cli.commands.webui._load_webui_launch")
    def test_webui_with_share(self, mock_load_launch):
        runner = CliRunner()
        mock_launch = MagicMock()
        mock_load_launch.return_value = mock_launch

        result = runner.invoke(cli, ["webui", "--share"])
        assert result.exit_code == 0
        mock_launch.assert_called_once_with(share=True, port=7860, debug=False)

    @patch("vitriol.cli.commands.webui._load_webui_launch")
    def test_webui_with_debug(self, mock_load_launch):
        runner = CliRunner()
        mock_launch = MagicMock()
        mock_load_launch.return_value = mock_launch

        result = runner.invoke(cli, ["webui", "--debug"])
        assert result.exit_code == 0
        mock_launch.assert_called_once_with(share=False, port=7860, debug=True)

    @patch("vitriol.cli.commands.webui._load_webui_launch")
    def test_webui_all_options(self, mock_load_launch):
        runner = CliRunner()
        mock_launch = MagicMock()
        mock_load_launch.return_value = mock_launch

        result = runner.invoke(cli, ["webui", "--port", "3000", "--share", "--debug"])
        assert result.exit_code == 0
        mock_launch.assert_called_once_with(share=True, port=3000, debug=True)


class TestWebuiCommandErrors:
    @patch("vitriol.cli.commands.webui._load_webui_launch")
    def test_webui_keyboard_interrupt(self, mock_load_launch):
        runner = CliRunner()
        mock_launch = MagicMock(side_effect=KeyboardInterrupt())
        mock_load_launch.return_value = mock_launch

        result = runner.invoke(cli, ["webui"])
        assert result.exit_code == 0
        assert "stopped" in result.output.lower() or result.output == ""

    @patch("vitriol.cli.commands.webui._load_webui_launch")
    def test_webui_launch_failure(self, mock_load_launch):
        runner = CliRunner()
        mock_launch = MagicMock(side_effect=RuntimeError("Launch failed"))
        mock_load_launch.return_value = mock_launch

        result = runner.invoke(cli, ["webui"])
        assert result.exit_code != 0
        assert "Launch failed" in result.output or "ClickException" in str(result.exception)


class TestWebuiLoadFunction:
    def test_load_webui_launch_imports(self):
        from vitriol.cli.commands.webui import _load_webui_launch
        # Should not raise
        try:
            launch = _load_webui_launch()
            assert callable(launch)
        except ImportError:
            pytest.skip("webui module not available")
