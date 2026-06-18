"""Unit tests for check report rendering."""

from vitriol.core.check_report import render_check_index_html
from vitriol.core.check_runner import CheckReport, CheckStepResult


def test_render_check_index_html_includes_steps() -> None:
    report = CheckReport(
        model_id="org/model",
        output_dir="/tmp/report",
        success=True,
        vitriol_version="0.3.0",
        generated_at="2026-06-18T00:00:00Z",
        steps=[
            CheckStepResult(
                name="analyze",
                success=True,
                duration_seconds=0.5,
                artifacts={"analysis.json": "analysis.json"},
            )
        ],
        fingerprint={
            "architecture_hash": "abc",
            "behavioral_dna_hash": "def",
            "weight_distribution_hash": "ghi",
            "vitriol_signature": "arx_123",
        },
    )
    html = render_check_index_html(report)
    assert "Vitriol Structure Check" in html
    assert "org/model" in html
    assert "analyze" in html
    assert "arx_123" in html
