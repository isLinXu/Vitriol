"""Golden-path integration tests for the Structure-First workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vitriol.core.check_runner import CheckOptions, StructureCheckRunner

pytestmark = pytest.mark.integration


class TestGoldenPathCheck:
    def test_check_offline_local_model(self, tiny_llama_model_dir: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "report"
        options = CheckOptions(
            model_id=str(tiny_llama_model_dir),
            output_dir=str(out_dir),
            strategy="compact",
            trust_remote_code=False,
            allow_network=False,
            local_files_only=True,
            run_inference=False,
            compute_weight_hash=False,
        )
        report = StructureCheckRunner(options).run()

        assert report.success is True
        assert (out_dir / "index.html").exists()
        assert (out_dir / "check-report.json").exists()
        assert (out_dir / "analysis.json").exists()
        assert (out_dir / "architecture.html").exists()
        assert (out_dir / "weights" / "config.json").exists()
        assert (out_dir / "weights" / "vitriol-manifest.json").exists()
        assert (out_dir / "validation.json").exists()
        assert (out_dir / "fingerprint.json").exists()

        payload = json.loads((out_dir / "check-report.json").read_text(encoding="utf-8"))
        assert payload["success"] is True
        assert payload["schema_version"] == 1
        step_names = [step["name"] for step in payload["steps"]]
        assert step_names == ["analyze", "arch_viz", "generate", "validate", "fingerprint"]

    def test_check_skip_generate(self, tiny_llama_model_dir: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "report-lite"
        options = CheckOptions(
            model_id=str(tiny_llama_model_dir),
            output_dir=str(out_dir),
            skip_generate=True,
            allow_network=False,
            local_files_only=True,
        )
        report = StructureCheckRunner(options).run()

        assert report.success is True
        assert (out_dir / "architecture.html").exists()
        assert not (out_dir / "weights").exists()
        skipped = [s for s in report.steps if s.summary.get("skipped")]
        assert len(skipped) >= 2
