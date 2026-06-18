"""Tests for vitriol cis CLI."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from vitriol.cli.main import cli
from vitriol.metrics.compression_intelligence import CompressionIntelligenceScorer


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestCISCLI:
    def test_cis_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["cis", "--help"])
        assert result.exit_code == 0
        assert "Compression Intelligence" in result.output

    def test_cis_rank(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["cis", "rank"])
        assert result.exit_code == 0
        assert "random" in result.output
        assert "ultra" in result.output

    def test_cis_rank_json(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["cis", "rank", "--json"])
        assert result.exit_code == 0
        rows = json.loads(result.output)
        assert isinstance(rows, list)
        assert rows[0]["rank"] == 1

    def test_cis_table(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["cis", "table"])
        assert result.exit_code == 0
        assert "PSI Score" in result.output

    def test_cis_report(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["cis", "report", "-o", "cis.md"])
            assert result.exit_code == 0
            with open("cis.md", encoding="utf-8") as f:
                content = f.read()
            assert "Vitriol CIS Strategy Report" in content

    @patch("vitriol.core.strategy_benchmark.StrategyCompareRunner")
    def test_cis_compare_success(self, mock_runner_cls, runner: CliRunner) -> None:
        from vitriol.core.strategy_benchmark import StrategyCompareReport, StrategyCompareRow

        mock_runner_cls.return_value.run.return_value = StrategyCompareReport(
            model_id="Tiny/Llama",
            output_dir="compare",
            success=True,
            vitriol_version="0.3.1",
            generated_at="2026-06-18T00:00:00Z",
            rows=[
                StrategyCompareRow(
                    strategy="compact",
                    success=True,
                    empirical_psi=0.42,
                    theoretical_psi=0.51,
                    validate_success=True,
                    model_loadable=True,
                    total_size_bytes=1024,
                    duration_seconds=1.2,
                )
            ],
        )
        with runner.isolated_filesystem():
            result = runner.invoke(
                cli,
                ["cis", "compare", "Tiny/Llama", "-o", "compare", "--strategies", "compact"],
            )
        assert result.exit_code == 0
        assert "Compare passed" in result.output


class TestScoreAllStrategies:
    def test_score_all_strategies_sorted(self) -> None:
        scorer = CompressionIntelligenceScorer()
        ranked = scorer.score_all_strategies()
        assert ranked
        psi_values = [psi for _, psi in ranked]
        assert psi_values == sorted(psi_values, reverse=True)
