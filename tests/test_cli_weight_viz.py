"""Tests for vitriol.cli.commands.weight_viz module."""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from vitriol.cli.main import cli


class TestWeightVizCommandHelp:
    def test_weight_viz_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["weight-viz", "--help"])
        assert result.exit_code == 0
        assert "weight" in result.output.lower()


class TestWeightVizBuildLayerData:
    def test_build_layer_data_from_config_no_config(self):
        from vitriol.cli.commands.weight_viz import _build_layer_data_from_config
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _build_layer_data_from_config(Path(tmpdir), max_layers=12)
            assert result["model_name"] == Path(tmpdir).name
            assert result["layers"] == []

    def test_build_layer_data_from_config_with_config(self):
        from vitriol.cli.commands.weight_viz import _build_layer_data_from_config
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "hidden_size": 128,
                "num_hidden_layers": 2,
                "vocab_size": 1000,
                "intermediate_size": 512,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "model_type": "test_model",
            }
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps(config))

            result = _build_layer_data_from_config(Path(tmpdir), max_layers=12)
            assert result["model_name"] == "test_model"
            assert result["hidden_size"] == 128
            assert result["num_layers"] == 2
            assert result["vocab_size"] == 1000
            assert result["config_source"] == "config.json"
            assert result["weight_stats_available"] is False
            # embed_tokens + 2 layers * 7 sublayers + lm_head
            expected_layers = 1 + 2 * 7 + 1
            assert len(result["layers"]) == expected_layers

    def test_build_layer_data_from_config_with_meta(self):
        from vitriol.cli.commands.weight_viz import _build_layer_data_from_config
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"hidden_size": 128}
            meta = {
                "hidden_size": 256,
                "num_hidden_layers": 1,
                "vocab_size": 500,
                "intermediate_size": 1024,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "model_type": "meta_model",
            }
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps(config))
            meta_path = Path(tmpdir) / "meta-config.json"
            meta_path.write_text(json.dumps(meta))

            result = _build_layer_data_from_config(Path(tmpdir), max_layers=12)
            assert result["model_name"] == "meta_model"
            assert result["hidden_size"] == 256
            assert result["config_source"] == "meta-config.json"

    def test_build_layer_data_from_config_max_layers(self):
        from vitriol.cli.commands.weight_viz import _build_layer_data_from_config
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "hidden_size": 128,
                "num_hidden_layers": 100,
                "vocab_size": 1000,
                "intermediate_size": 512,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
            }
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps(config))

            result = _build_layer_data_from_config(Path(tmpdir), max_layers=5)
            # Only 5 layers + embed_tokens + lm_head
            expected_layers = 1 + 5 * 7 + 1
            assert len(result["layers"]) == expected_layers


class TestWeightVizCommandMocked:
    @patch("vitriol.cli.commands.weight_viz.serve_3d_weights")
    @patch("vitriol.cli.commands.weight_viz._build_layer_data_from_config")
    def test_weight_viz_config_only(self, mock_build, mock_serve):
        runner = CliRunner()
        mock_build.return_value = {
            "model_name": "test",
            "layers": [{"name": "layer1", "shape": [10, 10], "type": "Linear"}],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, ["weight-viz", "-m", tmpdir, "--config-only"])
            assert result.exit_code == 0
            mock_build.assert_called_once()
            mock_serve.assert_called_once()

    @patch("vitriol.cli.commands.weight_viz.serve_3d_weights")
    @patch("vitriol.cli.commands.weight_viz._build_layer_data_from_config")
    def test_weight_viz_no_weights_fallback(self, mock_build, mock_serve):
        runner = CliRunner()
        mock_build.return_value = {
            "model_name": "test",
            "layers": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, ["weight-viz", "-m", tmpdir])
            assert result.exit_code == 0
            assert "No weight files found" in result.output
            mock_build.assert_called_once()

    @patch("vitriol.cli.commands.weight_viz.serve_3d_weights")
    @patch("vitriol.cli.commands.weight_viz._build_layer_data_from_weights")
    def test_weight_viz_with_weights(self, mock_build, mock_serve):
        runner = CliRunner()
        mock_build.return_value = {
            "model_name": "test",
            "layers": [{"name": "layer1", "shape": [10, 10], "type": "Linear"}],
            "weight_stats_available": True,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake weight file
            weight_file = Path(tmpdir) / "model.safetensors"
            weight_file.write_text("fake")

            result = runner.invoke(cli, ["weight-viz", "-m", tmpdir])
            assert result.exit_code == 0
            mock_build.assert_called_once()

    def test_weight_viz_nonexistent_path(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["weight-viz", "-m", "/nonexistent/path"])
        assert result.exit_code == 0
        assert "not found" in result.output.lower()

    @patch("vitriol.cli.commands.weight_viz.serve_3d_weights")
    @patch("vitriol.cli.commands.weight_viz._build_layer_data_from_config")
    def test_weight_viz_custom_port(self, mock_build, mock_serve):
        runner = CliRunner()
        mock_build.return_value = {
            "model_name": "test",
            "layers": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, ["weight-viz", "-m", tmpdir, "--port", "9999"])
            assert result.exit_code == 0
            mock_serve.assert_called_once()
            args, kwargs = mock_serve.call_args
            assert kwargs.get("port") == 9999

    @patch("vitriol.cli.commands.weight_viz.serve_3d_weights")
    @patch("vitriol.cli.commands.weight_viz._build_layer_data_from_config")
    def test_weight_viz_no_open(self, mock_build, mock_serve):
        runner = CliRunner()
        mock_build.return_value = {
            "model_name": "test",
            "layers": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, ["weight-viz", "-m", tmpdir, "--no-open"])
            assert result.exit_code == 0
            mock_serve.assert_called_once()
            args, kwargs = mock_serve.call_args
            assert kwargs.get("no_open") is True

    @patch("vitriol.cli.commands.weight_viz.serve_3d_weights")
    @patch("vitriol.cli.commands.weight_viz._build_layer_data_from_config")
    def test_weight_viz_max_layers(self, mock_build, mock_serve):
        runner = CliRunner()
        mock_build.return_value = {
            "model_name": "test",
            "layers": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, ["weight-viz", "-m", tmpdir, "--max-layers", "24"])
            assert result.exit_code == 0
            mock_build.assert_called_once()
            args, kwargs = mock_build.call_args
            assert args[1] == 24


class TestWeightVizServeFunction:
    @patch("vitriol.cli.commands.weight_viz.HTTPServer")
    @patch("vitriol.cli.commands.weight_viz.threading.Thread")
    def test_serve_3d_weights(self, mock_thread, mock_server_cls):
        from vitriol.cli.commands.weight_viz import serve_3d_weights
        mock_httpd = MagicMock()
        mock_server_cls.return_value = mock_httpd

        with tempfile.TemporaryDirectory() as tmpdir:
            serve_3d_weights(tmpdir, port=8888, no_open=True)
            mock_server_cls.assert_called_once()
            mock_httpd.serve_forever.assert_called_once()

    @patch("vitriol.cli.commands.weight_viz.HTTPServer")
    def test_serve_3d_weights_keyboard_interrupt(self, mock_server_cls):
        from vitriol.cli.commands.weight_viz import serve_3d_weights
        mock_httpd = MagicMock()
        mock_httpd.serve_forever.side_effect = KeyboardInterrupt()
        mock_server_cls.return_value = mock_httpd

        with tempfile.TemporaryDirectory() as tmpdir:
            serve_3d_weights(tmpdir, port=8888, no_open=True)
            mock_httpd.server_close.assert_called_once()
