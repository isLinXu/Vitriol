"""Unit tests for ``vitriol check`` CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from vitriol.cli.main import cli
from vitriol.core.check_runner import CheckReport, CheckStepResult


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def successful_report() -> CheckReport:
    return CheckReport(
        model_id="Tiny/Llama",
        output_dir="/tmp/report",
        success=True,
        vitriol_version="0.3.0",
        generated_at="2026-06-18T00:00:00Z",
        steps=[
            CheckStepResult(name="analyze", success=True, duration_seconds=0.1),
            CheckStepResult(name="arch_viz", success=True, duration_seconds=0.2),
        ],
    )


class TestCheckCLI:
    def test_check_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["check", "--help"])
        assert result.exit_code == 0
        assert "Structure-First" in result.output

    def test_check_listed_in_main_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "check" in result.output

    @patch("vitriol.cli.commands.check.StructureCheckRunner")
    def test_check_success(self, mock_runner_cls, runner: CliRunner, successful_report: CheckReport) -> None:
        mock_runner_cls.return_value.run.return_value = successful_report
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["check", "Tiny/Llama", "-o", "report"])
        assert result.exit_code == 0
        assert "Check passed" in result.output
        mock_runner_cls.assert_called_once()

    @patch("vitriol.cli.commands.check.StructureCheckRunner")
    def test_check_failure_exits_nonzero(self, mock_runner_cls, runner: CliRunner) -> None:
        mock_runner_cls.return_value.run.return_value = CheckReport(
            model_id="Bad/Model",
            output_dir="report",
            success=False,
            vitriol_version="0.3.0",
            generated_at="2026-06-18T00:00:00Z",
            steps=[
                CheckStepResult(
                    name="analyze",
                    success=False,
                    duration_seconds=0.01,
                    error="boom",
                )
            ],
        )
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["check", "Bad/Model", "-o", "report"])
        assert result.exit_code == 1
        assert "Check failed" in result.output

    @patch("vitriol.cli.commands.check.StructureCheckRunner")
    def test_check_fast_flag(self, mock_runner_cls, runner: CliRunner, successful_report: CheckReport) -> None:
        mock_runner_cls.return_value.run.return_value = successful_report
        with runner.isolated_filesystem():
            runner.invoke(cli, ["check", "Tiny/Llama", "-o", "report", "--fast"])
        options = mock_runner_cls.call_args.args[0]
        assert options.run_inference is False
        assert options.compute_weight_hash is False
