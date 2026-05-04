"""Tests for CLI validate command."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from click.testing import CliRunner

from vitriol.cli.commands.validate import validate


class TestValidateCommand:
    """Tests for validate CLI command."""

    @patch("vitriol.core.validator.ModelValidator")
    def test_validate_success(self, mock_validator_class):
        runner = CliRunner()
        mock_report = MagicMock()
        mock_report.success = True
        mock_report.model_loadable = True
        mock_report.tokenizer_loadable = True
        mock_report.inference_test = True
        mock_report.memory_usage_gb = 2.5
        mock_report.errors = []
        mock_report.warnings = []

        mock_validator = MagicMock()
        mock_validator.validate.return_value = mock_report
        mock_validator_class.return_value = mock_validator

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                validate,
                [str(tmpdir)],
                obj={"trust_remote_code": True, "allow_network": True, "local_files_only": False}
            )
            assert result.exit_code == 0
            assert f"Validation Report for {tmpdir}" in result.output
            assert "Success: True" in result.output
            assert "2.50 GB" in result.output
            mock_validator_class.assert_called_once_with(str(tmpdir), trust_remote_code=True)
            mock_validator.validate.assert_called_once_with(run_inference=True)

    @patch("vitriol.core.validator.ModelValidator")
    def test_validate_no_inference(self, mock_validator_class):
        runner = CliRunner()
        mock_report = MagicMock()
        mock_report.success = True
        mock_report.model_loadable = True
        mock_report.tokenizer_loadable = True
        mock_report.inference_test = False
        mock_report.memory_usage_gb = None
        mock_report.errors = []
        mock_report.warnings = []

        mock_validator = MagicMock()
        mock_validator.validate.return_value = mock_report
        mock_validator_class.return_value = mock_validator

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                validate,
                [str(tmpdir), "--no-inference"],
                obj={"trust_remote_code": True, "allow_network": True, "local_files_only": False}
            )
            assert result.exit_code == 0
            mock_validator.validate.assert_called_once_with(run_inference=False)

    @patch("vitriol.core.validator.ModelValidator")
    def test_validate_with_errors(self, mock_validator_class):
        runner = CliRunner()
        mock_report = MagicMock()
        mock_report.success = False
        mock_report.model_loadable = False
        mock_report.tokenizer_loadable = True
        mock_report.inference_test = False
        mock_report.memory_usage_gb = None
        mock_report.errors = ["Model file missing", "Config invalid"]
        mock_report.warnings = ["Deprecated format"]

        mock_validator = MagicMock()
        mock_validator.validate.return_value = mock_report
        mock_validator_class.return_value = mock_validator

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                validate,
                [str(tmpdir)],
                obj={"trust_remote_code": True, "allow_network": True, "local_files_only": False}
            )
            assert result.exit_code == 1
            assert "Errors:" in result.output
            assert "Model file missing" in result.output
            assert "Warnings:" in result.output
            assert "Deprecated format" in result.output

    @patch("vitriol.core.validator.ModelValidator")
    def test_validate_with_ctx_trust_remote(self, mock_validator_class):
        runner = CliRunner()
        mock_report = MagicMock()
        mock_report.success = True
        mock_report.model_loadable = True
        mock_report.tokenizer_loadable = True
        mock_report.inference_test = True
        mock_report.memory_usage_gb = None
        mock_report.errors = []
        mock_report.warnings = []

        mock_validator = MagicMock()
        mock_validator.validate.return_value = mock_report
        mock_validator_class.return_value = mock_validator

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                validate,
                [str(tmpdir)],
                obj={"trust_remote_code": False, "allow_network": True, "local_files_only": False}
            )
            assert result.exit_code == 0
            mock_validator_class.assert_called_once_with(str(tmpdir), trust_remote_code=False)

    @patch("vitriol.core.validator.ModelValidator")
    def test_validate_trust_remote_warning(self, mock_validator_class):
        runner = CliRunner()
        mock_report = MagicMock()
        mock_report.success = True
        mock_report.model_loadable = True
        mock_report.tokenizer_loadable = True
        mock_report.inference_test = True
        mock_report.memory_usage_gb = None
        mock_report.errors = []
        mock_report.warnings = []

        mock_validator = MagicMock()
        mock_validator.validate.return_value = mock_report
        mock_validator_class.return_value = mock_validator

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                validate,
                [str(tmpdir)],
                obj={"trust_remote_code": True}
            )
            assert result.exit_code == 0
            assert "SECURITY WARNING" in result.output
            assert "trust_remote_code is enabled" in result.output

    def test_validate_missing_dir(self):
        runner = CliRunner()
        result = runner.invoke(validate, ["/nonexistent/path"])
        assert result.exit_code != 0
        assert "does not exist" in result.output or "Invalid value" in result.output
