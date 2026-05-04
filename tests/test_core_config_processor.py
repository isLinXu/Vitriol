"""Tests for vitriol.core.config_processor module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from vitriol.core.config_processor import ConfigProcessor, ConfigLoadError


class TestConfigProcessor:
    """Tests for ConfigProcessor class."""

    def test_init(self):
        """Test initialization."""
        processor = ConfigProcessor("test/model", cache_dir="/tmp/cache")
        assert processor.model_id == "test/model"
        assert processor.cache_dir == "/tmp/cache"
        assert processor.raw_config is None
        assert processor.processed_config is None

    @patch("vitriol.core.config_processor.Path.exists")
    @patch("vitriol.core.config_processor.hf_load_config")
    def test_load_local_path(self, mock_hf_load, mock_exists):
        """Test loading from local path."""
        mock_exists.return_value = True
        mock_config = Mock()
        mock_config.__class__.__name__ = "TestConfig"
        mock_hf_load.return_value = mock_config

        processor = ConfigProcessor("/local/path")
        result = processor.load()

        assert result == mock_config
        assert processor.raw_config == mock_config
        mock_hf_load.assert_called_once()
        _, kwargs = mock_hf_load.call_args
        assert kwargs["security"]["local_files_only"] is True
        assert kwargs["security"]["trust_remote_code"] is False

    @patch("vitriol.core.config_processor.Path.exists")
    @patch("vitriol.core.config_processor.hf_load_config")
    def test_load_remote_model(self, mock_hf_load, mock_exists):
        """Test loading remote model config."""
        mock_exists.return_value = False
        mock_config = Mock()
        mock_hf_load.return_value = mock_config

        processor = ConfigProcessor("org/model")
        result = processor.load()

        assert result == mock_config
        mock_hf_load.assert_called_once()
        _, kwargs = mock_hf_load.call_args
        assert kwargs["security"]["local_files_only"] is False
        assert kwargs["security"]["trust_remote_code"] is True

    @patch("vitriol.core.config_processor.Path.exists")
    @patch("vitriol.core.config_processor.hf_load_config")
    def test_load_with_kwargs(self, mock_hf_load, mock_exists):
        """Test load with additional kwargs."""
        mock_exists.return_value = True
        mock_hf_load.return_value = Mock()

        processor = ConfigProcessor("test/model")
        processor.load(trust_remote_code=True, allow_network=False, revision="v1.0")

        _, kwargs = mock_hf_load.call_args
        assert kwargs["security"]["trust_remote_code"] is True
        assert kwargs["security"]["allow_network"] is False

    @patch("vitriol.core.config_processor.Path.exists")
    @patch("vitriol.core.config_processor.hf_load_config")
    def test_load_failure(self, mock_hf_load, mock_exists):
        """Test load failure raises ConfigLoadError."""
        mock_exists.return_value = False
        mock_hf_load.side_effect = Exception("Network error")

        processor = ConfigProcessor("org/model")
        with pytest.raises(ConfigLoadError) as exc_info:
            processor.load()

        assert "Failed to load config" in str(exc_info.value)
        assert "Network error" in str(exc_info.value)

    @patch("vitriol.core.config_processor.PatchRegistry")
    @patch("vitriol.core.config_processor.AdapterRegistry")
    def test_process(self, mock_adapter_registry, mock_patch_registry):
        """Test config processing."""
        mock_config = Mock()
        mock_config.__class__.__name__ = "TestConfig"
        mock_adapter = Mock()
        mock_adapter.patch_config.return_value = mock_config
        mock_adapter_registry.get_adapter.return_value = mock_adapter

        processor = ConfigProcessor("test/model")
        processor.raw_config = mock_config
        result = processor.process()

        assert result == mock_config
        mock_patch_registry.apply.assert_called_once_with(mock_config, "test/model")
        mock_adapter_registry.get_adapter.assert_called_once_with("test/model", mock_config)
        mock_adapter.patch_config.assert_called_once_with(mock_config)

    @patch("vitriol.core.config_processor.PatchRegistry")
    @patch("vitriol.core.config_processor.AdapterRegistry")
    def test_process_with_adapter_register(self, mock_adapter_registry, mock_patch_registry):
        """Test process with adapter that has register_classes."""
        mock_config = Mock()
        mock_adapter = Mock()
        mock_adapter.patch_config.return_value = mock_config
        mock_adapter.register_classes = Mock()
        mock_adapter_registry.get_adapter.return_value = mock_adapter

        processor = ConfigProcessor("test/model")
        processor.raw_config = mock_config
        processor.process()

        mock_adapter.register_classes.assert_called_once()

    def test_process_no_config(self):
        """Test process without loaded config."""
        processor = ConfigProcessor("test/model")
        with pytest.raises(ConfigLoadError) as exc_info:
            processor.process()
        assert "No config loaded" in str(exc_info.value)

    def test_process_with_explicit_config(self):
        """Test process with explicit config parameter."""
        mock_config = Mock()
        processor = ConfigProcessor("test/model")

        with patch("vitriol.core.config_processor.PatchRegistry") as mock_patch:
            with patch("vitriol.core.config_processor.AdapterRegistry") as mock_adapter:
                mock_adapter.get_adapter.return_value = None
                result = processor.process(mock_config)
                assert result == mock_config

    @patch("vitriol.core.config_processor.Path.exists")
    @patch("vitriol.core.config_processor.hf_load_config")
    def test_load_and_process(self, mock_hf_load, mock_exists):
        """Test load_and_process convenience method."""
        mock_exists.return_value = False
        mock_config = Mock()
        mock_hf_load.return_value = mock_config

        processor = ConfigProcessor("org/model")

        with patch("vitriol.core.config_processor.PatchRegistry") as mock_patch:
            with patch("vitriol.core.config_processor.AdapterRegistry") as mock_adapter:
                mock_adapter.get_adapter.return_value = None
                result = processor.load_and_process()
                assert result == mock_config

    def test_validate_with_defaults(self):
        """Test validate with default config."""
        mock_config = Mock()
        mock_config.hidden_size = 4096
        mock_config.num_hidden_layers = 32

        processor = ConfigProcessor("test/model")
        processor.processed_config = mock_config
        assert processor.validate() is True

    def test_validate_with_raw_config(self):
        """Test validate falls back to raw_config."""
        mock_config = Mock()
        mock_config.hidden_size = 2048
        mock_config.num_hidden_layers = 16

        processor = ConfigProcessor("test/model")
        processor.raw_config = mock_config
        assert processor.validate() is True

    def test_validate_no_config(self):
        """Test validate with no config raises error."""
        processor = ConfigProcessor("test/model")
        with pytest.raises(ValueError) as exc_info:
            processor.validate()
        assert "No config to validate" in str(exc_info.value)

    def test_validate_suspicious_hidden_size(self, caplog):
        """Test validate warns on suspiciously large hidden_size."""
        mock_config = Mock()
        mock_config.hidden_size = 200000
        mock_config.num_hidden_layers = 32

        processor = ConfigProcessor("test/model")
        processor.raw_config = mock_config
        with caplog.at_level("WARNING"):
            processor.validate()
        assert "Suspiciously large hidden_size" in caplog.text

    def test_validate_suspicious_layer_count(self, caplog):
        """Test validate warns on suspiciously large layer count."""
        mock_config = Mock()
        mock_config.hidden_size = 4096
        mock_config.num_hidden_layers = 2000

        processor = ConfigProcessor("test/model")
        processor.raw_config = mock_config
        with caplog.at_level("WARNING"):
            processor.validate()
        assert "Suspiciously large layer count" in caplog.text

    def test_validate_missing_required_attr(self):
        """Test validate raises on missing required attribute."""
        mock_config = Mock(spec=[])

        processor = ConfigProcessor("test/model")
        processor.raw_config = mock_config
        with pytest.raises(ValueError) as exc_info:
            processor.validate()
        assert "Missing required attribute" in str(exc_info.value)

    def test_validate_with_explicit_config(self):
        """Test validate with explicit config parameter."""
        mock_config = Mock()
        mock_config.hidden_size = 1024
        mock_config.num_hidden_layers = 12

        processor = ConfigProcessor("test/model")
        assert processor.validate(mock_config) is True


class TestConfigLoadError:
    """Tests for ConfigLoadError exception."""

    def test_is_exception(self):
        """Test ConfigLoadError is an Exception."""
        assert issubclass(ConfigLoadError, Exception)

    def test_raise_and_catch(self):
        """Test raising and catching ConfigLoadError."""
        with pytest.raises(ConfigLoadError):
            raise ConfigLoadError("test error")
