import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

import vitriol.tools.model_demo as model_demo
from vitriol.tools.model_demo import (
    DEFAULT_MODEL_ID,
    DemoOptions,
    build_demo_plan,
    build_batch_launch_script,
    check_viz_metadata,
    default_markdown_report_path,
    default_launch_group_dir,
    filter_demo_targets,
    infer_target_family,
    list_supported_families,
    load_demo_targets,
    parse_args,
    precheck_viz_targets,
    precheck_demo_targets,
    render_markdown_report,
    render_targets_template,
    render_plan_lines,
)


@pytest.fixture(autouse=True)
def fix_repo_root(monkeypatch):
    """Ensure repo_root_from_module points to the actual project root for test isolation."""
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(model_demo, "repo_root_from_module", lambda: repo_root)


def test_model_demo_plan_contains_prepare_weight_viz_and_arch_viz() -> None:
    options = DemoOptions(
        repo_root=Path("/repo"),
        output_dir=Path("/repo/output/glm_5_1_demo"),
        model_id="zai-org/GLM-5.1",
        no_open=True,
        arch_port=8877,
    )

    plan = build_demo_plan(options)

    assert [step.name for step in plan] == [
        "prepare-demo-dir",
        "weight-viz",
        "arch-viz",
    ]

    assert plan[1].command == [
        "python3",
        "-m",
        "vitriol.cli.main",
        "weight-viz",
        "-m",
        "/repo/output/glm_5_1_demo",
        "--port",
        "8781",
        "--no-open",
    ]
    assert plan[2].command == [
        "python3",
        "-m",
        "vitriol.cli.main",
        "--no-trust-remote-code",
        "viz",
        "/repo/output/glm_5_1_demo",
        "--port",
        "8877",
        "--no-open",
    ]


def test_model_demo_default_model_is_only_example_not_hardcoded_behavior() -> None:
    assert DEFAULT_MODEL_ID == "zai-org/GLM-5.1"


def test_model_demo_plan_uses_local_model_path_without_prepare_step() -> None:
    options = DemoOptions(
        repo_root=Path("/repo"),
        output_dir=Path("/models/local-demo"),
        model_id="",
        source_path=Path("/models/local-demo"),
        static_arch_viz=True,
        no_open=True,
    )

    plan = build_demo_plan(options)

    assert [step.name for step in plan] == [
        "arch-viz-block",
        "arch-viz-detail",
        "arch-viz-html",
        "weight-viz",
        "arch-viz",
    ]
    assert plan[0].command == [
        "python3",
        "-m",
        "vitriol.cli.main",
        "--no-trust-remote-code",
        "arch-viz",
        "/models/local-demo",
        "--block",
        "--output",
        "/models/local-demo/architecture.png",
    ]


def test_parse_args_prefers_model_path_for_existing_local_model() -> None:
    options = parse_args([
        "--repo-root",
        "/repo",
        "--model-path",
        "/models/local-demo",
        "--static-arch-viz",
    ])

    assert options.model_id == ""
    assert options.source_path == Path("/models/local-demo")
    assert options.output_dir == Path("/models/local-demo")
    assert options.static_arch_viz is True
    assert options.trust_remote_code is False


def test_parse_args_enables_remote_code_only_when_requested() -> None:
    options = parse_args([
        "--repo-root",
        "/repo",
        "--trust-remote-code",
    ])

    assert options.trust_remote_code is True


def test_list_supported_families_exposes_registered_adapter_modules() -> None:
    families = list_supported_families()

    assert "glm" in families
    assert "qwen" in families
    assert "minimax" in families


def test_render_plan_lines_supports_dry_run_output() -> None:
    options = DemoOptions(
        repo_root=Path("/repo"),
        output_dir=Path("/repo/output/demo"),
        model_id="zai-org/GLM-5.1",
        static_arch_viz=True,
        no_open=True,
    )

    lines = render_plan_lines(build_demo_plan(options))

    assert any("arch-viz-block" in line for line in lines)
    assert any("weight-viz" in line for line in lines)
    assert any("--no-open" in line for line in lines)


def test_load_demo_targets_supports_model_ids_and_local_paths(tmp_path: Path) -> None:
    models_file = tmp_path / "models.txt"
    models_file.write_text(
        "# demo targets\n"
        "zai-org/GLM-5.1\n"
        "path:/models/local-demo\n"
        "\n"
        "~/models/phi-demo\n",
        encoding="utf-8",
    )

    targets = load_demo_targets(models_file)

    assert targets == [
        {"model_id": "zai-org/GLM-5.1", "model_path": None},
        {"model_id": "", "model_path": Path("/models/local-demo")},
        {"model_id": "", "model_path": Path("~/models/phi-demo").expanduser()},
    ]


def test_build_batch_launch_script_assigns_unique_ports_and_commands() -> None:
    script = build_batch_launch_script(
        targets=[
            {"model_id": "zai-org/GLM-5.1", "model_path": None},
            {"model_id": "", "model_path": Path("/models/local-demo")},
        ],
        arch_port_base=9000,
        weight_port_base=9100,
        no_open=True,
        static_arch_viz=True,
    )

    assert "#!/usr/bin/env bash" in script
    assert "--model-id zai-org/GLM-5.1" in script
    assert "--model-path /models/local-demo" in script
    assert "--arch-port 9000" in script
    assert "--weight-port 9100" in script
    assert "--arch-port 9002" in script
    assert "--weight-port 9102" in script
    assert "--static-arch-viz" in script
    assert "nohup ./scripts/run_model_demo.sh" in script


def test_precheck_demo_targets_reports_ok_and_fail(monkeypatch) -> None:
    targets = [
        {"model_id": "zai-org/GLM-5.1", "model_path": None},
        {"model_id": "bad/model", "model_path": None},
    ]

    def fake_probe(model_id: str, _model_path, **_kwargs):
        if model_id == "bad/model":
            raise RuntimeError("boom")
        return {"model_type": "glm"}

    monkeypatch.setattr(model_demo, "probe_demo_target", fake_probe)

    reports = precheck_demo_targets(targets)

    assert reports[0]["status"] == "ok"
    assert reports[0]["model_type"] == "glm"
    assert reports[1]["status"] == "fail"
    assert "boom" in reports[1]["error"]


def test_render_targets_template_includes_supported_examples() -> None:
    content = render_targets_template(["glm", "qwen", "llama"])

    assert "# Remote model IDs" in content
    assert "zai-org/GLM-5.1" in content
    assert "Qwen/Qwen2.5-7B" in content
    assert "meta-llama/Llama-3.1-8B" in content
    assert "tencent/Hy3-preview" in content


def test_check_viz_metadata_accepts_top_level_or_text_config_fields() -> None:
    assert check_viz_metadata({
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "vocab_size": 32000,
    })["ok"] is True

    assert check_viz_metadata({
        "text_config": {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
        }
    })["ok"] is True


def test_precheck_viz_targets_reports_missing_metadata(monkeypatch) -> None:
    targets = [
        {"model_id": "good/model", "model_path": None},
        {"model_id": "bad/model", "model_path": None},
    ]

    def fake_probe(model_id: str, _model_path, **_kwargs):
        if model_id == "good/model":
            return {
                "model_type": "glm",
                "config_dict": {"hidden_size": 4096, "num_hidden_layers": 32, "vocab_size": 32000},
            }
        return {
            "model_type": "broken",
            "config_dict": {"hidden_size": 4096},
        }

    monkeypatch.setattr(model_demo, "probe_demo_target", fake_probe)

    reports = precheck_viz_targets(targets)

    assert reports[0]["status"] == "ok"
    assert reports[1]["status"] == "fail"
    assert "num_hidden_layers" in reports[1]["error"]


def test_render_markdown_report_contains_sections_and_statuses() -> None:
    markdown = render_markdown_report(
        title="Demo Precheck Report",
        config_reports=[
            {"source": "zai-org/GLM-5.1", "status": "ok", "model_type": "glm"},
            {"source": "bad/model", "status": "fail", "error": "boom"},
        ],
        viz_reports=[
            {"source": "zai-org/GLM-5.1", "status": "ok", "model_type": "glm"},
        ],
        final_targets=["zai-org/GLM-5.1"],
    )

    assert "# Demo Precheck Report" in markdown
    assert "## Config Precheck" in markdown
    assert "## Visualization Precheck" in markdown
    assert "## Final Targets" in markdown
    assert "| zai-org/GLM-5.1 | ok | glm |" in markdown
    assert "| bad/model | fail | - | boom |" in markdown


def test_infer_target_family_handles_model_ids_and_local_paths() -> None:
    assert infer_target_family("zai-org/GLM-5.1", None) == "glm"
    assert infer_target_family("Qwen/Qwen2.5-7B-Instruct", None) == "qwen"
    assert infer_target_family("tencent/Hy3-preview", None) == "hy3"
    assert infer_target_family("", Path("/models/minimax_m2_7_ultra")) == "minimax"


def test_filter_demo_targets_supports_only_and_exclude() -> None:
    targets = [
        {"model_id": "zai-org/GLM-5.1", "model_path": None},
        {"model_id": "Qwen/Qwen2.5-7B-Instruct", "model_path": None},
        {"model_id": "", "model_path": Path("/models/minimax_m2_7_ultra")},
    ]

    only_filtered = filter_demo_targets(targets, only_families={"glm", "qwen"}, exclude_families=set())
    exclude_filtered = filter_demo_targets(targets, only_families=set(), exclude_families={"minimax"})

    assert [t["model_id"] for t in only_filtered] == [
        "zai-org/GLM-5.1",
        "Qwen/Qwen2.5-7B-Instruct",
    ]
    assert [str(t["model_path"] or t["model_id"]) for t in exclude_filtered] == [
        "zai-org/GLM-5.1",
        "Qwen/Qwen2.5-7B-Instruct",
    ]


def test_default_markdown_report_path_points_to_output_directory() -> None:
    assert default_markdown_report_path(Path("/repo")) == Path("/repo/output/demo_precheck_report.md")


def test_default_launch_group_dir_points_to_group_directory() -> None:
    assert default_launch_group_dir(Path("/repo"), "demo-a") == Path("/repo/output/demo_groups/demo-a")


def test_build_batch_launch_script_uses_custom_log_dir() -> None:
    script = build_batch_launch_script(
        targets=[{"model_id": "zai-org/GLM-5.1", "model_path": None}],
        arch_port_base=9000,
        weight_port_base=9100,
        no_open=True,
        static_arch_viz=False,
        log_dir="output/demo_groups/demo-a/logs",
    )

    assert 'mkdir -p "output/demo_groups/demo-a/logs"' in script
    assert '> "output/demo_groups/demo-a/logs/demo_1.log" 2>&1 &' in script
