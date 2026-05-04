"""Tests for vitriol.core.batch module."""

import pytest
import yaml
from unittest.mock import Mock, patch, mock_open
from pathlib import Path

from vitriol.core.batch import BatchGenerator
from vitriol.config.manager import GenerationConfig


class TestBatchGenerator:
    """Tests for BatchGenerator class."""

    def test_init_loads_config(self, tmp_path):
        """Test that __init__ loads YAML config file."""
        config_data = {
            "models": [
                {"id": "model1", "output": "/tmp/out1", "options": {"strategy": "compact"}}
            ]
        }
        config_file = tmp_path / "batch_config.yaml"
        config_file.write_text(yaml.dump(config_data))

        generator = BatchGenerator(str(config_file))
        assert generator.config == config_data

    def test_generate_all_empty_models(self, tmp_path, caplog):
        """Test generate_all with empty models list."""
        config_data = {"models": []}
        config_file = tmp_path / "batch_config.yaml"
        config_file.write_text(yaml.dump(config_data))

        generator = BatchGenerator(str(config_file))
        with caplog.at_level("INFO"):
            generator.generate_all()
        assert "Starting batch generation for 0 models" in caplog.text

    @patch("vitriol.core.batch.MinimalWeightGenerator")
    def test_generate_all_success(self, mock_gen_class, tmp_path, caplog):
        """Test successful batch generation."""
        config_data = {
            "models": [
                {
                    "id": "test/model1",
                    "output": str(tmp_path / "out1"),
                    "options": {
                        "max_shard_size": "2GB",
                        "strategy": "compact",
                        "dtype": "float16"
                    }
                }
            ]
        }
        config_file = tmp_path / "batch_config.yaml"
        config_file.write_text(yaml.dump(config_data))

        mock_gen = Mock()
        mock_gen_class.return_value = mock_gen

        generator = BatchGenerator(str(config_file))
        with caplog.at_level("INFO"):
            generator.generate_all()

        mock_gen_class.assert_called_once()
        args, kwargs = mock_gen_class.call_args
        assert kwargs["model_id"] == "test/model1"
        assert kwargs["output_dir"] == str(tmp_path / "out1")
        assert isinstance(kwargs["config"], GenerationConfig)
        mock_gen.generate.assert_called_once()
        assert "Successfully generated test/model1" in caplog.text

    @patch("vitriol.core.batch.MinimalWeightGenerator")
    def test_generate_all_default_options(self, mock_gen_class, tmp_path):
        """Test generation with default options."""
        config_data = {
            "models": [
                {"id": "test/model2", "output": str(tmp_path / "out2")}
            ]
        }
        config_file = tmp_path / "batch_config.yaml"
        config_file.write_text(yaml.dump(config_data))

        mock_gen = Mock()
        mock_gen_class.return_value = mock_gen

        generator = BatchGenerator(str(config_file))
        generator.generate_all()

        _, kwargs = mock_gen_class.call_args
        assert kwargs["config"].max_shard_size == "5GB"
        assert kwargs["config"].strategy == "random"
        assert kwargs["config"].dtype == "bfloat16"

    @patch("vitriol.core.batch.MinimalWeightGenerator")
    def test_generate_all_failure_continues(self, mock_gen_class, tmp_path, caplog):
        """Test that one model failure doesn't stop batch."""
        config_data = {
            "models": [
                {"id": "test/fail", "output": str(tmp_path / "fail")},
                {"id": "test/success", "output": str(tmp_path / "success")}
            ]
        }
        config_file = tmp_path / "batch_config.yaml"
        config_file.write_text(yaml.dump(config_data))

        def side_effect(*args, **kwargs):
            mock = Mock()
            if kwargs["model_id"] == "test/fail":
                mock.generate.side_effect = RuntimeError("Generation failed")
            return mock

        mock_gen_class.side_effect = side_effect

        generator = BatchGenerator(str(config_file))
        with caplog.at_level("ERROR"):
            generator.generate_all()

        assert "Failed to generate test/fail" in caplog.text
        assert mock_gen_class.call_count == 2

    @patch("vitriol.core.batch.MinimalWeightGenerator")
    def test_generate_all_multiple_models(self, mock_gen_class, tmp_path):
        """Test batch generation with multiple models."""
        config_data = {
            "models": [
                {"id": "test/model1", "output": str(tmp_path / "out1")},
                {"id": "test/model2", "output": str(tmp_path / "out2")},
                {"id": "test/model3", "output": str(tmp_path / "out3")}
            ]
        }
        config_file = tmp_path / "batch_config.yaml"
        config_file.write_text(yaml.dump(config_data))

        mock_gen = Mock()
        mock_gen_class.return_value = mock_gen

        generator = BatchGenerator(str(config_file))
        generator.generate_all()

        assert mock_gen_class.call_count == 3
        assert mock_gen.generate.call_count == 3

    def test_generate_all_missing_models_key(self, tmp_path, caplog):
        """Test with config missing 'models' key."""
        config_data = {"other_key": "value"}
        config_file = tmp_path / "batch_config.yaml"
        config_file.write_text(yaml.dump(config_data))

        generator = BatchGenerator(str(config_file))
        with caplog.at_level("INFO"):
            generator.generate_all()
        assert "Starting batch generation for 0 models" in caplog.text
