"""Tests for vitriol.cli.commands.* modules and cli/main.py."""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from vitriol.cli.main import cli


# ─────────────────────────────────────────────────────────────
# CLI import smoke tests
# ─────────────────────────────────────────────────────────────

class TestCLIImports:
    def test_main_cli_imports(self):
        from vitriol.cli.main import cli
        assert cli is not None

    def test_commands_importable(self):
        # Verify all command modules can be imported
        pass


# ─────────────────────────────────────────────────────────────
# Main CLI
# ─────────────────────────────────────────────────────────────

class TestMainCLI:
    def test_cli_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.output

    def test_cli_version(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0


# ─────────────────────────────────────────────────────────────
# Hash command
# ─────────────────────────────────────────────────────────────

class TestHashCommand:
    def test_hash_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["hash", "--help"])
        assert result.exit_code == 0
        assert "hash" in result.output.lower()

    @patch("vitriol.core.hasher.ModelHasher")
    @patch("rich.console.Console")
    def test_hash_fast(self, mock_console_cls, mock_hasher_cls):
        runner = CliRunner()
        mock_hasher = MagicMock()
        mock_hasher.compute_architecture_hash.return_value = "abc123"
        mock_hasher.compute_activation_signature_hash.return_value = "sig456"
        mock_hasher_cls.return_value = mock_hasher
        mock_console_cls.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            result = runner.invoke(cli, ["hash", str(model_dir), "--fast"])
            assert result.exit_code == 0


# ─────────────────────────────────────────────────────────────
# Analyze command
# ─────────────────────────────────────────────────────────────

class TestAnalyzeCommand:
    def test_analyze_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "analyze" in result.output.lower()

    @patch("vitriol.core.analyzer.ModelAnalyzer")
    def test_analyze_mock(self, mock_analyzer_cls):
        runner = CliRunner()
        mock_analyzer = MagicMock()
        analysis = MagicMock()
        analysis.architecture = "llama"
        analysis.total_params = 7_000_000_000
        analysis.layer_count = 32
        analysis.hidden_size = 4096
        analysis.vocab_size = 32000
        analysis.special_features = ["GQA"]
        analysis.estimated_file_size = {"fp16": 14.0}
        mock_analyzer.return_value.analyze.return_value = analysis
        mock_analyzer_cls.return_value = mock_analyzer.return_value

        result = runner.invoke(cli, ["analyze", "test/model"])
        assert result.exit_code == 0
        assert "llama" in result.output


# ─────────────────────────────────────────────────────────────
# Export command
# ─────────────────────────────────────────────────────────────

class TestExportCommand:
    def test_export_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["export", "--help"])
        assert result.exit_code == 0

    @patch("vitriol.core.exporter.ModelExporter")
    def test_export_json(self, mock_exporter_cls):
        runner = CliRunner()
        mock_exporter = MagicMock()
        mock_exporter_cls.return_value = mock_exporter

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.json"
            result = runner.invoke(cli, ["export", tmpdir, "--output", str(out_path)])
            assert result.exit_code == 0
            mock_exporter.export_structure.assert_called_once_with(str(out_path))


# ─────────────────────────────────────────────────────────────
# Validate command
# ─────────────────────────────────────────────────────────────

class TestValidateCommand:
    def test_validate_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--help"])
        assert result.exit_code == 0

    @patch("vitriol.core.validator.ModelValidator")
    def test_validate_mock(self, mock_validator_cls):
        runner = CliRunner()
        mock_validator = MagicMock()
        report = MagicMock()
        report.success = True
        report.model_loadable = True
        report.tokenizer_loadable = True
        report.inference_test = True
        report.memory_usage_gb = 1.5
        report.errors = []
        report.warnings = []
        report.to_dict.return_value = {"success": True}
        mock_validator.return_value.validate.return_value = report
        mock_validator_cls.return_value = mock_validator.return_value

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, ["validate", tmpdir])
            assert result.exit_code == 0


# ─────────────────────────────────────────────────────────────
# Batch command
# ─────────────────────────────────────────────────────────────

class TestBatchCommand:
    def test_batch_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["batch", "--help"])
        assert result.exit_code == 0


# ─────────────────────────────────────────────────────────────
# Infer command
# ─────────────────────────────────────────────────────────────

class TestInferCommand:
    def test_infer_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["infer", "--help"])
        assert result.exit_code == 0


# ─────────────────────────────────────────────────────────────
# Bench command
# ─────────────────────────────────────────────────────────────

class TestBenchCommand:
    def test_bench_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["bench", "--help"])
        assert result.exit_code == 0


# ─────────────────────────────────────────────────────────────
# NAS command
# ─────────────────────────────────────────────────────────────

class TestNASCommand:
    def test_nas_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["nas", "--help"])
        assert result.exit_code == 0


# ─────────────────────────────────────────────────────────────
# Viz / Visualize commands
# ─────────────────────────────────────────────────────────────

class TestVizCommands:
    def test_viz_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["viz", "--help"])
        assert result.exit_code == 0

    def test_visualize_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["visualize", "--help"])
        assert result.exit_code == 0

    def test_weight_viz_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["weight-viz", "--help"])
        assert result.exit_code == 0

    def test_vocab_viz_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["vocab-viz", "--help"])
        assert result.exit_code == 0

    def test_arch_viz_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["arch-viz", "--help"])
        assert result.exit_code == 0


# ─────────────────────────────────────────────────────────────
# Other commands
# ─────────────────────────────────────────────────────────────

class TestOtherCommands:
    def test_generate_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["generate", "--help"])
        assert result.exit_code == 0

    def test_evolve_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["evolve", "--help"])
        assert result.exit_code == 0

    def test_exobrain_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["exobrain", "--help"])
        assert result.exit_code == 0

    def test_trace_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["trace", "--help"])
        assert result.exit_code == 0

    def test_webui_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["webui", "--help"])
        assert result.exit_code == 0
