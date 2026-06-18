"""Golden-path integration test for multi-strategy CIS compare."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vitriol.core.strategy_benchmark import StrategyCompareOptions, StrategyCompareRunner

pytestmark = pytest.mark.integration


class TestGoldenPathStrategyCompare:
    def test_compare_offline_local_model(self, tiny_llama_model_dir: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "compare"
        options = StrategyCompareOptions(
            model_id=str(tiny_llama_model_dir),
            output_dir=str(out_dir),
            strategies=("compact", "random"),
            trust_remote_code=False,
            allow_network=False,
            local_files_only=True,
            run_inference=False,
            cis_tensor_limit=20,
        )
        report = StrategyCompareRunner(options).run()

        assert report.success is True
        assert len(report.rows) == 2
        assert (out_dir / "compare-report.json").exists()
        assert (out_dir / "compare-report.md").exists()
        assert (out_dir / "compact" / "config.json").exists()
        assert (out_dir / "random" / "config.json").exists()

        payload = json.loads((out_dir / "compare-report.json").read_text(encoding="utf-8"))
        assert payload["success"] is True
        assert payload["schema_version"] == 1
        for row in payload["strategies"]:
            assert row["validate_success"] is True
            assert row["empirical_psi"] is not None

        sizes = {row["strategy"]: row["total_size_bytes"] for row in payload["strategies"]}
        assert sizes["compact"] > 0
        assert sizes["random"] > 0
