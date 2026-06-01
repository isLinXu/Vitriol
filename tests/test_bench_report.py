import json

from vitriol.cli.commands.bench_report import generate_report


def test_generate_report_json_omits_layers_by_default() -> None:
    report = generate_report(
        {
            "model_id": "demo/model",
            "score": 0.9,
            "layers": [{"idx": 0, "mse": 0.01}],
        },
        output_format="json",
        metadata={"preset": "balanced"},
    )

    payload = json.loads(report)
    assert payload["model_id"] == "demo/model"
    assert payload["score"] == 0.9
    assert "layers" not in payload
    assert payload["metadata"]["preset"] == "balanced"


def test_generate_report_html_contains_report_shell() -> None:
    report = generate_report({"model_id": "demo/model"}, output_format="html")

    assert "<!DOCTYPE html>" in report
    assert "Bench Report" in report
    assert "demo/model" in report
