import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import vitriol.tools.minimax_pipeline as minimax_pipeline
from vitriol.tools.minimax_pipeline import (
    PipelineOptions,
    apply_validation_runtime_patches,
    build_pipeline_plan,
    patch_config_for_validation,
    parse_args,
)


def test_default_pipeline_plan_contains_generate_validate_and_static_viz() -> None:
    options = PipelineOptions(
        repo_root=Path("/repo"),
        output_dir=Path("/repo/output/minimax_m2_7_ultra"),
    )

    plan = build_pipeline_plan(options)

    assert [step.name for step in plan] == [
        "generate",
        "validate-load",
        "arch-viz-block",
        "arch-viz-detail",
        "arch-viz-html",
    ]

    generate_step = plan[0]
    assert generate_step.command == [
        "python3",
        "-m",
        "vitriol.cli.main",
        "--no-trust-remote-code",
        "generate",
        "MiniMaxAI/MiniMax-M2.7",
        "--output-dir",
        "/repo/output/minimax_m2_7_ultra",
        "--strategy",
        "ultra",
        "--max-shard-size",
        "5GB",
        "--no-shrink",
    ]


def test_parse_args_defaults_to_no_remote_code() -> None:
    options = parse_args(["--repo-root", "/repo"])

    assert options.trust_remote_code is False


def test_parse_args_enables_remote_code_only_when_requested() -> None:
    options = parse_args(["--repo-root", "/repo", "--trust-remote-code"])

    assert options.trust_remote_code is True


def test_pipeline_plan_adds_interactive_viz_when_requested() -> None:
    options = PipelineOptions(
        repo_root=Path("/repo"),
        output_dir=Path("/repo/output/minimax_m2_7_ultra"),
        serve_viz=True,
        no_open=True,
        port=9900,
    )

    plan = build_pipeline_plan(options)

    assert plan[-1].name == "viz-serve"
    assert plan[-1].command == [
        "python3",
        "-m",
        "vitriol.cli.main",
        "--no-trust-remote-code",
        "viz",
        "/repo/output/minimax_m2_7_ultra",
        "--port",
        "9900",
        "--no-open",
    ]


def test_apply_validation_runtime_patches_uses_vitriol_global_patches(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(minimax_pipeline, "apply_all_patches", lambda: calls.append("patched"))

    apply_validation_runtime_patches()

    assert calls == ["patched"]


def test_patch_config_for_validation_applies_family_and_adapter_patches(monkeypatch) -> None:
    config = SimpleNamespace(marker=[])

    def fake_apply(cfg, model_id: str) -> None:
        cfg.marker.append(("family", model_id))

    class FakeAdapter:
        def patch_config(self, cfg):
            cfg.marker.append(("adapter", "patched"))
            cfg.extra = "ok"
            return cfg

    monkeypatch.setattr(minimax_pipeline.PatchRegistry, "apply", fake_apply)
    monkeypatch.setattr(minimax_pipeline.AdapterRegistry, "get_adapter", lambda model_id, cfg: FakeAdapter())

    patched = patch_config_for_validation(config, "/repo/output/minimax_m2_7_ultra")

    assert patched is config
    assert patched.marker == [
        ("family", "/repo/output/minimax_m2_7_ultra"),
        ("adapter", "patched"),
    ]
    assert patched.extra == "ok"


def test_patch_config_for_validation_normalizes_minimax_rope_type() -> None:
    config = SimpleNamespace(
        model_type="minimax_m2",
        rope_theta=5_000_000,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "factor": 1.0,
            "scale": 1.0,
        },
        is_encoder_decoder=True,
    )

    patched = patch_config_for_validation(config, "/repo/output/minimax_m2_7_ultra")

    assert patched.rope_parameters["rope_type"] == "linear"
    assert patched.rope_parameters["rope_theta"] == 5_000_000
    assert patched.rope_parameters["factor"] == 1.0
    assert "scale" not in patched.rope_parameters
    assert patched.is_encoder_decoder is False
