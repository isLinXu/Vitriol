"""Tests for core/exporter module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from vitriol.core.exporter import ModelExporter


class TestModelExporterInit:
    """Tests for ModelExporter initialization."""

    def test_init(self):
        exporter = ModelExporter("/tmp/model")
        assert str(exporter.input_dir) == "/tmp/model"
        assert exporter.trust_remote_code is True

    def test_init_custom_trust(self):
        exporter = ModelExporter("/tmp/model", trust_remote_code=False)
        assert exporter.trust_remote_code is False


class TestLoadBestConfig:
    """Tests for _load_best_config method."""

    @patch("vitriol.core.exporter.load_config_or_raw")
    def test_load_meta_config(self, mock_load_config):
        exporter = ModelExporter("/tmp/model")
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter.input_dir = Path(tmpdir)
            meta_config = {"hidden_size": 128, "model_type": "test"}
            (Path(tmpdir) / "meta-config.json").write_text(json.dumps(meta_config))

            mock_config = MagicMock()
            mock_load_config.return_value = mock_config

            result = exporter._load_best_config()
            assert result is mock_config
            mock_load_config.assert_called()

    @patch("vitriol.core.exporter.load_config_or_raw")
    def test_load_config_meta_fallback(self, mock_load_config):
        exporter = ModelExporter("/tmp/model")
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter.input_dir = Path(tmpdir)
            meta_config = {"hidden_size": 256}
            (Path(tmpdir) / "config_meta.json").write_text(json.dumps(meta_config))

            mock_config = MagicMock()
            mock_load_config.return_value = mock_config

            result = exporter._load_best_config()
            assert result is mock_config

    @patch("vitriol.core.exporter.load_config_or_raw")
    def test_load_config_json_fallback(self, mock_load_config):
        exporter = ModelExporter("/tmp/model")
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter.input_dir = Path(tmpdir)
            config = {"hidden_size": 512}
            (Path(tmpdir) / "config.json").write_text(json.dumps(config))

            mock_config = MagicMock()
            mock_load_config.return_value = mock_config

            result = exporter._load_best_config()
            assert result is mock_config
            mock_load_config.assert_called_once_with(
                str(tmpdir),
                security={
                    "trust_remote_code": True,
                    "allow_network": False,
                    "local_files_only": True,
                },
            )

    @patch("vitriol.core.exporter.load_config_or_raw")
    def test_meta_config_load_failure(self, mock_load_config):
        exporter = ModelExporter("/tmp/model")
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter.input_dir = Path(tmpdir)
            # Create invalid meta-config
            (Path(tmpdir) / "meta-config.json").write_text("not json")
            config = {"hidden_size": 512}
            (Path(tmpdir) / "config.json").write_text(json.dumps(config))

            mock_config = MagicMock()
            mock_load_config.return_value = mock_config

            result = exporter._load_best_config()
            assert result is mock_config


class TestExportStructure:
    """Tests for export_structure method."""

    @patch("vitriol.core.exporter.ModelExporter._load_best_config")
    def test_export_structure(self, mock_load):
        exporter = ModelExporter("/tmp/model")
        mock_config = MagicMock()
        mock_config.model_type = "llama"
        mock_config.architectures = ["LlamaForCausalLM"]
        mock_config.hidden_size = 4096
        mock_config.num_hidden_layers = 32
        mock_config.num_attention_heads = 32
        mock_config.vocab_size = 32000
        mock_config.to_dict.return_value = {"model_type": "llama"}
        mock_load.return_value = mock_config

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "structure.json"
            exporter.export_structure(str(output_file))

            assert output_file.exists()
            data = json.loads(output_file.read_text())
            assert data["model_type"] == "llama"
            assert data["architectures"] == ["LlamaForCausalLM"]
            assert data["hidden_size"] == 4096
            assert data["num_layers"] == 32
            assert data["num_heads"] == 32
            assert data["vocab_size"] == 32000
            assert data["config"] == {"model_type": "llama"}

    @patch("vitriol.core.exporter.ModelExporter._load_best_config")
    def test_export_structure_missing_attrs(self, mock_load):
        exporter = ModelExporter("/tmp/model")
        # Use a simple object to test getattr defaults
        class MinimalConfig:
            architectures = []
            hidden_size = None
            num_hidden_layers = None
            num_attention_heads = None
            vocab_size = None
            def to_dict(self):
                return {}
        mock_config = MinimalConfig()
        mock_load.return_value = mock_config

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "structure.json"
            exporter.export_structure(str(output_file))

            data = json.loads(output_file.read_text())
            assert data["model_type"] == "unknown"

    @patch("vitriol.core.exporter.ModelExporter._load_best_config")
    def test_export_structure_error(self, mock_load):
        exporter = ModelExporter("/tmp/model")
        mock_load.side_effect = Exception("Load error")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "structure.json"
            with pytest.raises(Exception, match="Load error"):
                exporter.export_structure(str(output_file))

    @patch("vitriol.core.exporter.ModelExporter._load_best_config")
    def test_export_structure_creates_parent_directories(self, mock_load):
        exporter = ModelExporter("/tmp/model")
        mock_config = MagicMock()
        mock_config.model_type = "llama"
        mock_config.architectures = []
        mock_config.hidden_size = 4096
        mock_config.num_hidden_layers = 32
        mock_config.num_attention_heads = 32
        mock_config.vocab_size = 32000
        mock_config.to_dict.return_value = {"model_type": "llama"}
        mock_load.return_value = mock_config

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "nested" / "export" / "structure.json"
            exporter.export_structure(str(output_file))
            assert output_file.exists()


class TestExportGgufPrep:
    """Tests for export_gguf_prep method."""

    @patch("subprocess.run")
    def test_export_gguf_success(self, mock_run):
        exporter = ModelExporter("/tmp/model")
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter.export_gguf_prep(tmpdir)
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "llama_cpp.convert_hf_to_gguf" in cmd

    @patch("subprocess.run")
    def test_export_gguf_failure(self, mock_run):
        exporter = ModelExporter("/tmp/model")
        mock_run.side_effect = Exception("Command not found")
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should not raise, just log warning
            exporter.export_gguf_prep(tmpdir)
            mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_export_gguf_uses_explicit_output_file_path(self, mock_run):
        exporter = ModelExporter("/tmp/model")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "exports" / "custom-name.gguf"
            exporter.export_gguf_prep(str(output_file))
            cmd = mock_run.call_args[0][0]
            assert cmd[-1] == str(output_file)
            assert output_file.parent.exists()
