"""Tests for vitriol.webui.app module."""
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("gradio")
from vitriol.webui.app import (
    _ensure_cache_dirs,
    format_params,
    create_app,
)


class TestEnsureCacheDirs:
    def test_creates_directories(self):
        _ensure_cache_dirs()
        assert "MPLCONFIGDIR" in os.environ
        assert "XDG_CACHE_HOME" in os.environ
        mpl_dir = Path(os.environ["MPLCONFIGDIR"])
        xdg_dir = Path(os.environ["XDG_CACHE_HOME"])
        assert mpl_dir.exists()
        assert xdg_dir.exists()


class TestFormatParams:
    def test_trillions(self):
        assert format_params(1.5e12) == "1.5T"

    def test_billions(self):
        assert format_params(7.2e9) == "7.2B"

    def test_millions(self):
        assert format_params(125e6) == "125.0M"

    def test_small(self):
        assert format_params(5000) == "5,000"


class TestCreateApp:
    @patch("vitriol.webui.app.gr")
    def test_create_app_returns_blocks(self, mock_gr):
        mock_blocks = MagicMock()
        mock_gr.Blocks.return_value = mock_blocks
        mock_gr.themes.Soft.return_value = MagicMock()

        app = create_app(title="Test App")
        assert app is mock_blocks
        mock_gr.Blocks.assert_called_once()

    @patch("vitriol.webui.app.gr")
    def test_create_app_default_title(self, mock_gr):
        mock_blocks = MagicMock()
        mock_gr.Blocks.return_value = mock_blocks
        mock_gr.themes.Soft.return_value = MagicMock()

        create_app()
        _, kwargs = mock_gr.Blocks.call_args
        assert "Vitriol" in kwargs["title"]

    @patch("vitriol.webui.app.gr")
    def test_create_app_defaults_remote_code_checkbox_off(self, mock_gr):
        mock_blocks = MagicMock()
        mock_gr.Blocks.return_value = mock_blocks
        mock_gr.themes.Soft.return_value = MagicMock()

        create_app()

        trust_checkbox_values = [
            kwargs.get("value")
            for _, kwargs in mock_gr.Checkbox.call_args_list
            if "Trust Remote Code" in kwargs.get("label", "")
        ]
        assert trust_checkbox_values
        assert all(value is False for value in trust_checkbox_values)
