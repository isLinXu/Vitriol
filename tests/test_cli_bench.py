import json
from pathlib import Path

from click.testing import CliRunner

from vitriol.cli.main import cli


def test_bench_command_is_listed() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "bench" in result.output


def test_bench_kv_smoke_invokes_runner(monkeypatch) -> None:
    runner = CliRunner()

    captured = {}

    def fake_run_smoke(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "preset": kwargs["preset"], "preset_params": kwargs["preset_params"]}

    monkeypatch.setattr("vitriol.cli.commands.bench.run_smoke", fake_run_smoke)

    result = runner.invoke(
        cli,
        [
            "bench",
            "kv-smoke",
            "demo/model",
            "--preset",
            "ultra-long",
            "--preset-param",
            "quantized_kv_start=256",
            "--preset-param",
            "enable_sparse_v=true",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert captured["preset"] == "ultra-long"
    assert captured["preset_params"]["quantized_kv_start"] == 256
    assert captured["preset_params"]["enable_sparse_v"] is True


def test_bench_kv_suite_invokes_runner(monkeypatch) -> None:
    runner = CliRunner()

    def fake_run_short_suite(cfg):
        return {
            "model_id": cfg.model_id,
            "preset": cfg.preset,
            "prompt_tokens": cfg.prompt_tokens,
            "preset_params": cfg.preset_params,
        }

    monkeypatch.setattr("vitriol.cli.commands.bench.run_short_suite", fake_run_short_suite)

    result = runner.invoke(
        cli,
        [
            "bench",
            "kv-suite",
            "demo/model",
            "--prompt-tokens",
            "1024",
            "--prompt-tokens",
            "4096",
            "--preset",
            "aggressive",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["model_id"] == "demo/model"
    assert payload["preset"] == "aggressive"
    assert payload["prompt_tokens"] == [1024, 4096]


def test_bench_kv_smoke_accepts_fast_balanced_preset(monkeypatch) -> None:
    runner = CliRunner()

    def fake_run_smoke(**kwargs):
        return {"ok": True, "preset": {"name": kwargs["preset"]}}

    monkeypatch.setattr("vitriol.cli.commands.bench.run_smoke", fake_run_smoke)

    result = runner.invoke(
        cli,
        ["bench", "kv-smoke", "demo/model", "--preset", "fast-balanced"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["preset"]["name"] == "fast-balanced"


def test_bench_kv_analyze_summary(monkeypatch) -> None:
    runner = CliRunner()

    monkeypatch.setattr(
        "vitriol.cli.commands.bench.analyze_kv_quantization",
        lambda **kwargs: {
            "model_id": kwargs["model_id"],
            "prompt_tokens": kwargs["prompt_tokens"],
            "base": {
                "preset": {"name": kwargs["preset"]},
                "summary": {
                    "quantized_layers": 6,
                    "avg_key_mse": 0.001234,
                    "avg_value_mse": 0.002345,
                    "avg_logits_mse": 0.003456,
                    "avg_output_mse": 0.004567,
                    "avg_residual_gain_k": 0.10,
                    "avg_residual_gain_v": 0.20,
                },
            },
        },
    )

    result = runner.invoke(
        cli,
        ["bench", "kv-analyze", "demo/model", "--preset", "balanced", "--format", "summary"],
    )

    assert result.exit_code == 0
    assert "model: demo/model" in result.output
    assert "base_preset: balanced" in result.output
    assert "base_quantized_layers: 6" in result.output
    assert "base_avg_key_mse: 0.001234" in result.output


def test_bench_kv_analyze_compare_markdown(monkeypatch) -> None:
    runner = CliRunner()

    monkeypatch.setattr(
        "vitriol.cli.commands.bench.analyze_kv_quantization",
        lambda **kwargs: {
            "model_id": kwargs["model_id"],
            "prompt_tokens": kwargs["prompt_tokens"],
            "base": {
                "preset": {"name": kwargs["preset"]},
                "summary": {"quantized_layers": 6, "avg_key_mse": 0.001234},
            },
            "compare": {
                "preset": {"name": kwargs["compare_preset"]},
                "summary": {"quantized_layers": 6, "avg_key_mse": 0.000987},
            },
        },
    )

    result = runner.invoke(
        cli,
        [
            "bench",
            "kv-analyze",
            "demo/model",
            "--preset",
            "balanced",
            "--compare-preset",
            "fast-balanced",
            "--format",
            "markdown",
        ],
    )

    assert result.exit_code == 0
    assert "## KV Analyze" in result.output
    assert "`base_preset`: balanced" in result.output
    assert "`compare_preset`: fast-balanced" in result.output
    assert "`base_avg_key_mse`: 0.001234" in result.output
    assert "`compare_avg_key_mse`: 0.000987" in result.output


def test_bench_kv_analyze_summary_show_layers(monkeypatch) -> None:
    runner = CliRunner()

    monkeypatch.setattr(
        "vitriol.cli.commands.bench.analyze_kv_quantization",
        lambda **kwargs: {
            "model_id": kwargs["model_id"],
            "prompt_tokens": kwargs["prompt_tokens"],
            "base": {
                "preset": {"name": kwargs["preset"]},
                "summary": {"quantized_layers": 1, "avg_key_mse": 0.001234},
                "layers": [
                    {
                        "layer_idx": 3,
                        "layer_type": "full_attention",
                        "turbo_quantize_k": True,
                        "turbo_quantize_v": False,
                        "key_mse": 0.001234,
                        "logits_mse": 0.123456,
                        "output_mse": 0.000111,
                        "residual_gain_k": 0.222222,
                    }
                ],
            },
        },
    )

    result = runner.invoke(
        cli,
        ["bench", "kv-analyze", "demo/model", "--preset", "balanced", "--format", "summary", "--show-layers"],
    )

    assert result.exit_code == 0
    assert "quantized layers (sorted by layer):" in result.output
    assert "full_attention" in result.output
    assert "0.001234" in result.output
    assert "0.222222" in result.output


def test_bench_kv_analyze_compare_markdown_show_layers(monkeypatch) -> None:
    runner = CliRunner()

    monkeypatch.setattr(
        "vitriol.cli.commands.bench.analyze_kv_quantization",
        lambda **kwargs: {
            "model_id": kwargs["model_id"],
            "prompt_tokens": kwargs["prompt_tokens"],
            "base": {
                "preset": {"name": kwargs["preset"]},
                "summary": {"quantized_layers": 1, "avg_key_mse": 0.001234},
                "layers": [
                    {
                        "layer_idx": 3,
                        "layer_type": "full_attention",
                        "turbo_quantize_k": True,
                        "turbo_quantize_v": False,
                        "key_mse": 0.001234,
                        "logits_mse": 0.123456,
                        "output_mse": 0.000111,
                        "residual_gain_k": 0.222222,
                    }
                ],
            },
            "compare": {
                "preset": {"name": kwargs["compare_preset"]},
                "summary": {"quantized_layers": 1, "avg_key_mse": 0.000987},
                "layers": [
                    {
                        "layer_idx": 3,
                        "layer_type": "full_attention",
                        "turbo_quantize_k": True,
                        "turbo_quantize_v": False,
                        "key_mse": 0.000987,
                        "logits_mse": 0.100000,
                        "output_mse": 0.000101,
                        "residual_gain_k": 0.333333,
                    }
                ],
            },
        },
    )

    result = runner.invoke(
        cli,
        [
            "bench",
            "kv-analyze",
            "demo/model",
            "--preset",
            "balanced",
            "--compare-preset",
            "fast-balanced",
            "--format",
            "markdown",
            "--show-layers",
        ],
    )

    assert result.exit_code == 0
    assert "### Quantized Layers" in result.output
    assert "| layer | type | base key mse | cmp key mse | delta key mse |" in result.output
    assert "| 3 | full_attention | 0.001234 | 0.000987 | -0.000247 |" in result.output


def test_bench_kv_analyze_summary_show_layers_sorted(monkeypatch) -> None:
    runner = CliRunner()

    monkeypatch.setattr(
        "vitriol.cli.commands.bench.analyze_kv_quantization",
        lambda **kwargs: {
            "model_id": kwargs["model_id"],
            "prompt_tokens": kwargs["prompt_tokens"],
            "base": {
                "preset": {"name": kwargs["preset"]},
                "summary": {"quantized_layers": 2, "avg_key_mse": 0.001234},
                "layers": [
                    {
                        "layer_idx": 3,
                        "layer_type": "full_attention",
                        "turbo_quantize_k": True,
                        "turbo_quantize_v": False,
                        "key_mse": 0.010000,
                        "logits_mse": 0.100000,
                        "output_mse": 0.000111,
                        "residual_gain_k": 0.100000,
                    },
                    {
                        "layer_idx": 7,
                        "layer_type": "full_attention",
                        "turbo_quantize_k": True,
                        "turbo_quantize_v": False,
                        "key_mse": 0.020000,
                        "logits_mse": 0.200000,
                        "output_mse": 0.000222,
                        "residual_gain_k": 0.200000,
                    },
                ],
            },
            "compare": {
                "preset": {"name": kwargs["compare_preset"]},
                "summary": {"quantized_layers": 2, "avg_key_mse": 0.000987},
                "layers": [
                    {
                        "layer_idx": 3,
                        "layer_type": "full_attention",
                        "turbo_quantize_k": True,
                        "turbo_quantize_v": False,
                        "key_mse": 0.030000,
                        "logits_mse": 0.300000,
                        "output_mse": 0.000111,
                        "residual_gain_k": 0.000000,
                    },
                    {
                        "layer_idx": 7,
                        "layer_type": "full_attention",
                        "turbo_quantize_k": True,
                        "turbo_quantize_v": False,
                        "key_mse": 0.090000,
                        "logits_mse": 0.900000,
                        "output_mse": 0.000222,
                        "residual_gain_k": 0.000000,
                    },
                ],
            },
        },
    )

    result = runner.invoke(
        cli,
        [
            "bench",
            "kv-analyze",
            "demo/model",
            "--preset",
            "balanced",
            "--compare-preset",
            "fast-balanced",
            "--format",
            "summary",
            "--show-layers",
            "--sort-by",
            "logits_mse_delta",
        ],
    )

    assert result.exit_code == 0
    assert "quantized layers (sorted by logits_mse_delta):" in result.output
    assert result.output.index("7      full_attention") < result.output.index("3      full_attention")


def test_bench_kv_smoke_summary_format(monkeypatch) -> None:
    runner = CliRunner()

    def fake_run_smoke(**kwargs):
        return {
            "model_id": kwargs["model_id"],
            "preset": {"name": kwargs["preset"]},
            "ok": True,
            "exact": True,
            "speedup": 1.25,
            "prefix_match": (8, 8, 100.0),
            "chosen_v_quantize_only_first_n": 2,
            "tuned_memory": {
                "estimated_kv_megabytes": 29.36,
                "peak_device_megabytes": 6390.98,
            },
            "tuned_turboquant": {
                "calls": 6,
                "avg_residual_l2": 0.210,
                "avg_correction_l2": 0.105,
                "correction_to_residual_l2_ratio": 0.500,
            },
            "policy_insights": {
                "quantized_kv_start": 256,
                "counts": {
                    "full_attention": 4,
                    "sliding_window": 0,
                    "mla": 0,
                    "linear_attention": 0,
                    "turbo_k": 2,
                    "turbo_v": 2,
                    "sparse_v": 1,
                    "compute_skip": 0,
                },
                "layers": [
                    {
                        "layer_idx": 0,
                        "layer_type": "full_attention",
                        "turbo_quantize_k": True,
                        "turbo_quantize_v": True,
                        "enable_sparse_v": True,
                        "enable_compute_skip": False,
                    },
                    {
                        "layer_idx": 1,
                        "layer_type": "sliding_window",
                        "turbo_quantize_k": False,
                        "turbo_quantize_v": False,
                        "enable_sparse_v": False,
                        "enable_compute_skip": False,
                    },
                ],
            },
        }

    monkeypatch.setattr("vitriol.cli.commands.bench.run_smoke", fake_run_smoke)

    result = runner.invoke(
        cli,
        ["bench", "kv-smoke", "demo/model", "--preset", "balanced", "--format", "summary"],
    )

    assert result.exit_code == 0
    assert "model: demo/model" in result.output
    assert "speedup: 1.250x" in result.output
    assert "chosen_v_quant_layers: 2" in result.output
    assert "estimated_kv_mb: 29.360" in result.output
    assert "peak_device_mb: 6390.980" in result.output
    assert "peak_minus_estimated_mb: 6361.620" in result.output
    assert "turboquant_calls: 6" in result.output
    assert "correction_to_residual_l2_ratio: 0.500" in result.output
    assert "quantized_kv_start: 256" in result.output
    assert "policy_counts:" in result.output


def test_bench_kv_smoke_compare_summary(monkeypatch) -> None:
    runner = CliRunner()

    def fake_compare_smoke(**kwargs):
        return {
            "model_id": kwargs["model_id"],
            "prompt_tokens": kwargs["prompt_tokens"],
            "base": {
                "preset": {"name": kwargs["preset"]},
                "ok": True,
                "tuned_exact": True,
                "tuned_speedup": 1.05,
                "tuned_memory": {
                    "estimated_kv_megabytes": 48.19,
                    "peak_device_megabytes": 6350.98,
                },
                "tuned_turboquant": {
                    "calls": 4,
                    "avg_residual_l2": 0.200,
                    "avg_correction_l2": 0.100,
                    "correction_to_residual_l2_ratio": 0.500,
                },
                "policy_insights": {"layers": []},
            },
            "compare": {
                "preset": {"name": kwargs["compare_preset"]},
                "ok": True,
                "tuned_exact": False,
                "tuned_speedup": 1.22,
                "tuned_memory": {
                    "estimated_kv_megabytes": 29.36,
                    "peak_device_megabytes": 6390.98,
                },
                "tuned_turboquant": {
                    "calls": 8,
                    "avg_residual_l2": 0.220,
                    "avg_correction_l2": 0.110,
                    "correction_to_residual_l2_ratio": 0.500,
                },
                "policy_insights": {"layers": []},
            },
            "delta_speedup": 0.17,
            "policy_diff": {
                "changed_layers": [
                    {
                        "layer_idx": 0,
                        "changes": {
                            "enable_sparse_v": {"base": False, "compare": True},
                        },
                    }
                ]
            },
        }

    monkeypatch.setattr("vitriol.cli.commands.bench.compare_smoke", fake_compare_smoke)

    result = runner.invoke(
        cli,
        [
            "bench",
            "kv-smoke",
            "demo/model",
            "--compare-preset",
            "aggressive",
            "--format",
            "summary",
        ],
    )

    assert result.exit_code == 0
    assert "base_preset: balanced" in result.output
    assert "compare_preset: aggressive" in result.output
    assert "delta_speedup: 0.170x" in result.output
    assert "base_estimated_kv_mb: 48.190" in result.output
    assert "compare_estimated_kv_mb: 29.360" in result.output
    assert "delta_estimated_kv_mb: -18.830" in result.output
    assert "base_peak_minus_estimated_mb: 6302.790" in result.output
    assert "compare_peak_minus_estimated_mb: 6361.620" in result.output
    assert "base_turboquant_calls: 4" in result.output
    assert "compare_turboquant_calls: 8" in result.output
    assert "enable_sparse_v" in result.output


def test_bench_kv_suite_summary_format(monkeypatch) -> None:
    runner = CliRunner()

    def fake_run_short_suite(cfg):
        return {
            "model_id": cfg.model_id,
            "preset": {"name": cfg.preset},
            "all_cases_exact_match": False,
            "chosen_v_quantize_only_first_n": 3,
            "policy_insights": {
                "quantized_kv_start": 512,
                "counts": {
                    "full_attention": 8,
                    "sliding_window": 2,
                    "mla": 0,
                    "linear_attention": 0,
                    "turbo_k": 3,
                    "turbo_v": 3,
                    "sparse_v": 2,
                    "compute_skip": 1,
                },
                "layers": [
                    {
                        "layer_idx": 0,
                        "layer_type": "full_attention",
                        "turbo_quantize_k": True,
                        "turbo_quantize_v": True,
                        "enable_sparse_v": True,
                        "enable_compute_skip": True,
                    },
                    {
                        "layer_idx": 1,
                        "layer_type": "sliding_window",
                        "turbo_quantize_k": False,
                        "turbo_quantize_v": False,
                        "enable_sparse_v": False,
                        "enable_compute_skip": False,
                    },
                ],
            },
            "results": [
                {
                    "name": "pt1024:qa",
                    "speedup": 1.18,
                    "exact": True,
                    "prefix_match": (16, 16, 100.0),
                    "base_toks_per_s": 42.5,
                    "tuned_toks_per_s": 50.3,
                },
                {
                    "name": "pt4096:chat",
                    "speedup": 1.31,
                    "exact": False,
                    "prefix_match": (14, 16, 87.5),
                    "base_toks_per_s": 20.0,
                    "tuned_toks_per_s": 26.2,
                },
            ],
        }

    monkeypatch.setattr("vitriol.cli.commands.bench.run_short_suite", fake_run_short_suite)

    result = runner.invoke(
        cli,
        ["bench", "kv-suite", "demo/model", "--format", "summary"],
    )

    assert result.exit_code == 0
    assert "case" in result.output
    assert "pt1024:qa" in result.output
    assert "16/16 (100.0%)" in result.output
    assert "pt4096:chat" in result.output
    assert "14/16 (87.5%)" in result.output
    assert "quantized_kv_start: 512" in result.output
    assert "turbo_k=3" in result.output


def test_bench_kv_suite_show_layers(monkeypatch) -> None:
    runner = CliRunner()

    def fake_run_short_suite(cfg):
        return {
            "model_id": cfg.model_id,
            "preset": {"name": cfg.preset},
            "all_cases_exact_match": True,
            "chosen_v_quantize_only_first_n": 1,
            "policy_insights": {
                "quantized_kv_start": 128,
                "counts": {
                    "full_attention": 2,
                    "sliding_window": 0,
                    "mla": 0,
                    "linear_attention": 0,
                    "turbo_k": 2,
                    "turbo_v": 1,
                    "sparse_v": 1,
                    "compute_skip": 0,
                },
                "layers": [
                    {
                        "layer_idx": 0,
                        "layer_type": "full_attention",
                        "turbo_quantize_k": True,
                        "turbo_quantize_v": True,
                        "enable_sparse_v": True,
                        "enable_compute_skip": False,
                    },
                    {
                        "layer_idx": 1,
                        "layer_type": "full_attention",
                        "turbo_quantize_k": True,
                        "turbo_quantize_v": False,
                        "enable_sparse_v": False,
                        "enable_compute_skip": False,
                    },
                ],
            },
            "results": [],
        }

    monkeypatch.setattr("vitriol.cli.commands.bench.run_short_suite", fake_run_short_suite)

    result = runner.invoke(
        cli,
        ["bench", "kv-suite", "demo/model", "--format", "summary", "--show-layers"],
    )

    assert result.exit_code == 0
    assert "layer  type" in result.output
    assert "full_attention" in result.output
    assert "Y" in result.output


def test_bench_kv_plan_single(monkeypatch) -> None:
    runner = CliRunner()

    def fake_build_policy_plan(**kwargs):
        return {
            "model_id": kwargs["model_id"],
            "preset": {"name": kwargs["preset"]},
            "chosen_v_quantize_only_first_n": 2,
            "policy_insights": {
                "quantized_kv_start": 256,
                "counts": {
                    "full_attention": 4,
                    "sliding_window": 1,
                    "mla": 0,
                    "linear_attention": 0,
                    "turbo_k": 4,
                    "turbo_v": 2,
                    "sparse_v": 1,
                    "compute_skip": 0,
                },
                "layers": [],
            },
        }

    monkeypatch.setattr("vitriol.cli.commands.bench.build_policy_plan", fake_build_policy_plan)

    result = runner.invoke(cli, ["bench", "kv-plan", "demo/model", "--format", "summary"])
    assert result.exit_code == 0
    assert "model: demo/model" in result.output
    assert "preset: balanced" in result.output
    assert "quantized_kv_start: 256" in result.output


def test_bench_kv_smoke_compare_markdown(monkeypatch) -> None:
    runner = CliRunner()

    def fake_compare_smoke(**kwargs):
        return {
            "model_id": kwargs["model_id"],
            "prompt_tokens": kwargs["prompt_tokens"],
            "base": {
                "preset": {"name": kwargs["preset"]},
                "ok": True,
                "tuned_exact": True,
                "tuned_speedup": 1.02,
                "tuned_memory": {
                    "estimated_kv_megabytes": 48.19,
                    "peak_device_megabytes": 6350.98,
                },
                "tuned_turboquant": {
                    "calls": 4,
                    "avg_residual_l2": 0.200,
                    "avg_correction_l2": 0.100,
                    "correction_to_residual_l2_ratio": 0.500,
                },
                "policy_insights": {"layers": []},
            },
            "compare": {
                "preset": {"name": kwargs["compare_preset"]},
                "ok": True,
                "tuned_exact": True,
                "tuned_speedup": 1.18,
                "tuned_memory": {
                    "estimated_kv_megabytes": 29.36,
                    "peak_device_megabytes": 6390.98,
                },
                "tuned_turboquant": {
                    "calls": 8,
                    "avg_residual_l2": 0.220,
                    "avg_correction_l2": 0.110,
                    "correction_to_residual_l2_ratio": 0.500,
                },
                "policy_insights": {"layers": []},
            },
            "delta_speedup": 0.16,
            "policy_diff": {
                "changed_layers": [
                    {
                        "layer_idx": 1,
                        "changes": {
                            "turbo_quantize_v": {"base": False, "compare": True},
                        },
                    }
                ]
            },
        }

    monkeypatch.setattr("vitriol.cli.commands.bench.compare_smoke", fake_compare_smoke)

    result = runner.invoke(
        cli,
        [
            "bench",
            "kv-smoke",
            "demo/model",
            "--compare-preset",
            "ultra-long",
            "--format",
            "markdown",
        ],
    )

    assert result.exit_code == 0
    assert "## Experiment Metadata" in result.output
    assert "## KV Smoke Compare" in result.output
    assert "`delta_speedup`: 0.160x" in result.output
    assert "`base_estimated_kv_mb`: 48.190" in result.output
    assert "`compare_peak_device_mb`: 6390.980" in result.output
    assert "`base_turboquant_calls`: 4" in result.output
    assert "`compare_turboquant_calls`: 8" in result.output
    assert "### Policy Changes" in result.output
    assert "turbo_quantize_v" in result.output


def test_bench_kv_plan_diff(monkeypatch) -> None:
    runner = CliRunner()

    def fake_build_policy_plan(**kwargs):
        preset = kwargs["preset"]
        if preset == "balanced":
            return {
                "model_id": kwargs["model_id"],
                "preset": {"name": "balanced"},
                "chosen_v_quantize_only_first_n": 1,
                "policy_insights": {
                    "quantized_kv_start": 2048,
                    "counts": {"full_attention": 2, "sliding_window": 0, "mla": 0, "linear_attention": 0, "turbo_k": 2, "turbo_v": 1, "sparse_v": 0, "compute_skip": 0},
                    "layers": [
                        {"layer_idx": 0, "layer_type": "full_attention", "turbo_quantize_k": True, "turbo_quantize_v": True, "enable_sparse_v": False, "enable_compute_skip": False},
                        {"layer_idx": 1, "layer_type": "full_attention", "turbo_quantize_k": True, "turbo_quantize_v": False, "enable_sparse_v": False, "enable_compute_skip": False},
                    ],
                },
            }
        return {
            "model_id": kwargs["model_id"],
            "preset": {"name": "ultra-long"},
            "chosen_v_quantize_only_first_n": 2,
            "policy_insights": {
                "quantized_kv_start": 512,
                "counts": {"full_attention": 2, "sliding_window": 0, "mla": 0, "linear_attention": 0, "turbo_k": 2, "turbo_v": 2, "sparse_v": 2, "compute_skip": 1},
                "layers": [
                    {"layer_idx": 0, "layer_type": "full_attention", "turbo_quantize_k": True, "turbo_quantize_v": True, "enable_sparse_v": True, "enable_compute_skip": True},
                    {"layer_idx": 1, "layer_type": "full_attention", "turbo_quantize_k": True, "turbo_quantize_v": True, "enable_sparse_v": True, "enable_compute_skip": False},
                ],
            },
        }

    monkeypatch.setattr("vitriol.cli.commands.bench.build_policy_plan", fake_build_policy_plan)

    result = runner.invoke(
        cli,
        ["bench", "kv-plan", "demo/model", "--compare-preset", "ultra-long", "--format", "summary"],
    )
    assert result.exit_code == 0
    assert "base_preset: balanced" in result.output
    assert "compare_preset: ultra-long" in result.output
    assert "changed_layers:" in result.output
    assert "compare" in result.output


def test_bench_kv_plan_json_output_file(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    def fake_build_policy_plan(**kwargs):
        return {
            "model_id": kwargs["model_id"],
            "preset": {"name": kwargs["preset"]},
            "chosen_v_quantize_only_first_n": 2,
            "policy_insights": {"quantized_kv_start": 256, "counts": {}, "layers": []},
        }

    monkeypatch.setattr("vitriol.cli.commands.bench.build_policy_plan", fake_build_policy_plan)

    output_path = tmp_path / "plan.json"
    result = runner.invoke(
        cli,
        ["bench", "kv-plan", "demo/model", "--format", "json", "--output", str(output_path)],
    )

    assert result.exit_code == 0
    assert str(output_path) in result.output
    payload = json.loads(output_path.read_text())
    assert payload["model_id"] == "demo/model"
    assert payload["preset"]["name"] == "balanced"


def test_bench_kv_suite_markdown(monkeypatch) -> None:
    runner = CliRunner()

    def fake_run_short_suite(cfg):
        return {
            "model_id": cfg.model_id,
            "preset": {"name": cfg.preset},
            "all_cases_exact_match": False,
            "chosen_v_quantize_only_first_n": 2,
            "policy_insights": {
                "quantized_kv_start": 512,
                "counts": {
                    "full_attention": 4,
                    "sliding_window": 0,
                    "mla": 0,
                    "linear_attention": 0,
                    "turbo_k": 4,
                    "turbo_v": 2,
                    "sparse_v": 1,
                    "compute_skip": 0,
                },
                "layers": [],
            },
            "results": [
                {
                    "name": "pt1024:qa",
                    "speedup": 1.10,
                    "exact": True,
                    "prefix_match": (8, 8, 100.0),
                    "base_toks_per_s": 20.0,
                    "tuned_toks_per_s": 22.0,
                }
            ],
        }

    monkeypatch.setattr("vitriol.cli.commands.bench.run_short_suite", fake_run_short_suite)

    result = runner.invoke(
        cli,
        [
            "bench",
            "kv-suite",
            "demo/model",
            "--format",
            "markdown",
            "--preset-param",
            "quantized_kv_start=512",
        ],
    )
    assert result.exit_code == 0
    assert "## Experiment Metadata" in result.output
    assert "`command`: bench kv-suite" in result.output
    assert "`preset_params`: quantized_kv_start=512" in result.output
    assert "## KV Suite" in result.output
    assert "| case | speedup | exact |" in result.output
    assert "pt1024:qa" in result.output


def test_bench_kv_suite_compare_summary(monkeypatch) -> None:
    runner = CliRunner()

    def fake_compare_short_suite(cfg, compare_preset, compare_preset_params=None):
        return {
            "model_id": cfg.model_id,
            "base": {
                "preset": {"name": cfg.preset},
                "policy_insights": {
                    "layers": [
                        {
                            "layer_idx": 0,
                            "layer_type": "full_attention",
                            "turbo_quantize_k": True,
                            "turbo_quantize_v": True,
                            "enable_sparse_v": False,
                            "enable_compute_skip": False,
                        }
                    ]
                },
            },
            "compare": {
                "preset": {"name": compare_preset},
                "policy_insights": {
                    "layers": [
                        {
                            "layer_idx": 0,
                            "layer_type": "full_attention",
                            "turbo_quantize_k": True,
                            "turbo_quantize_v": True,
                            "enable_sparse_v": True,
                            "enable_compute_skip": False,
                        }
                    ]
                },
            },
            "case_diffs": [
                {
                    "name": "pt1024:qa",
                    "base_speedup": 1.10,
                    "compare_speedup": 1.32,
                    "delta_speedup": 0.22,
                    "base_exact": True,
                    "compare_exact": False,
                }
            ],
            "policy_diff": {
                "changed_layers": [
                    {
                        "layer_idx": 0,
                        "changes": {
                            "enable_sparse_v": {"base": False, "compare": True},
                        },
                    }
                ]
            },
        }

    monkeypatch.setattr("vitriol.cli.commands.bench.compare_short_suite", fake_compare_short_suite)

    result = runner.invoke(
        cli,
        [
            "bench",
            "kv-suite",
            "demo/model",
            "--compare-preset",
            "ultra-long",
            "--format",
            "summary",
        ],
    )

    assert result.exit_code == 0
    assert "base_preset: balanced" in result.output
    assert "compare_preset: ultra-long" in result.output
    assert "pt1024:qa" in result.output
    assert "0.220x" in result.output
    assert "enable_sparse_v" in result.output


def test_bench_kv_suite_compare_markdown(monkeypatch) -> None:
    runner = CliRunner()

    def fake_compare_short_suite(cfg, compare_preset, compare_preset_params=None):
        return {
            "model_id": cfg.model_id,
            "base": {
                "preset": {"name": cfg.preset},
                "policy_insights": {"layers": []},
            },
            "compare": {
                "preset": {"name": compare_preset},
                "policy_insights": {"layers": []},
            },
            "case_diffs": [
                {
                    "name": "pt1024:qa",
                    "base_speedup": 1.05,
                    "compare_speedup": 1.28,
                    "delta_speedup": 0.23,
                    "base_exact": True,
                    "compare_exact": True,
                }
            ],
            "policy_diff": {
                "changed_layers": [
                    {
                        "layer_idx": 1,
                        "changes": {
                            "turbo_quantize_v": {"base": False, "compare": True},
                        },
                    }
                ]
            },
        }

    monkeypatch.setattr("vitriol.cli.commands.bench.compare_short_suite", fake_compare_short_suite)

    result = runner.invoke(
        cli,
        [
            "bench",
            "kv-suite",
            "demo/model",
            "--compare-preset",
            "aggressive",
            "--format",
            "markdown",
        ],
    )

    assert result.exit_code == 0
    assert "## Experiment Metadata" in result.output
    assert "## KV Suite Compare" in result.output
    assert "| case | base speedup | compare speedup | delta |" in result.output
    assert "### Policy Changes" in result.output
    assert "turbo_quantize_v" in result.output


def test_bench_kv_long_compare_summary(monkeypatch) -> None:
    runner = CliRunner()

    def fake_compare_long_context_preset(**kwargs):
        return {
            "model_id": kwargs["model_id"],
            "prompt_tokens": kwargs["prompt_tokens"],
            "base": {
                "preset": {"name": kwargs["preset"]},
                "tuned_exact": True,
                "tuned_speedup": 1.08,
                "tuned_memory": {
                    "estimated_kv_megabytes": 48.19,
                    "peak_device_megabytes": 6350.98,
                },
                "tuned_turboquant": {
                    "calls": 10,
                    "avg_residual_l2": 0.210,
                    "avg_correction_l2": 0.105,
                    "correction_to_residual_l2_ratio": 0.500,
                },
                "policy_insights": {"layers": []},
            },
            "compare": {
                "preset": {"name": kwargs["compare_preset"]},
                "tuned_exact": False,
                "tuned_speedup": 1.34,
                "tuned_memory": {
                    "estimated_kv_megabytes": 29.36,
                    "peak_device_megabytes": 6390.98,
                },
                "tuned_turboquant": {
                    "calls": 12,
                    "avg_residual_l2": 0.230,
                    "avg_correction_l2": 0.115,
                    "correction_to_residual_l2_ratio": 0.500,
                },
                "policy_insights": {"layers": []},
            },
            "delta_speedup": 0.26,
            "policy_diff": {
                "changed_layers": [
                    {
                        "layer_idx": 2,
                        "changes": {
                            "enable_compute_skip": {"base": False, "compare": True},
                        },
                    }
                ]
            },
        }

    monkeypatch.setattr("vitriol.cli.commands.bench.compare_long_context_preset", fake_compare_long_context_preset)

    result = runner.invoke(
        cli,
        [
            "bench",
            "kv-long",
            "demo/model",
            "--compare-preset",
            "ultra-long",
            "--format",
            "summary",
        ],
    )

    assert result.exit_code == 0
    assert "base_preset: balanced" in result.output
    assert "compare_preset: ultra-long" in result.output
    assert "delta_speedup: 0.260x" in result.output
    assert "base_peak_device_mb: 6350.980" in result.output
    assert "delta_peak_device_mb: 40.000" in result.output
    assert "compare_avg_correction_l2: 0.115" in result.output
    assert "enable_compute_skip" in result.output


def test_bench_kv_long_compare_markdown(monkeypatch) -> None:
    runner = CliRunner()

    def fake_compare_long_context_preset(**kwargs):
        return {
            "model_id": kwargs["model_id"],
            "prompt_tokens": kwargs["prompt_tokens"],
            "base": {
                "preset": {"name": kwargs["preset"]},
                "tuned_exact": True,
                "tuned_speedup": 1.10,
                "tuned_memory": {
                    "estimated_kv_megabytes": 48.19,
                    "peak_device_megabytes": 6350.98,
                },
                "tuned_turboquant": {
                    "calls": 10,
                    "avg_residual_l2": 0.210,
                    "avg_correction_l2": 0.105,
                    "correction_to_residual_l2_ratio": 0.500,
                },
                "policy_insights": {"layers": []},
            },
            "compare": {
                "preset": {"name": kwargs["compare_preset"]},
                "tuned_exact": True,
                "tuned_speedup": 1.29,
                "tuned_memory": {
                    "estimated_kv_megabytes": 29.36,
                    "peak_device_megabytes": 6390.98,
                },
                "tuned_turboquant": {
                    "calls": 12,
                    "avg_residual_l2": 0.230,
                    "avg_correction_l2": 0.115,
                    "correction_to_residual_l2_ratio": 0.500,
                },
                "policy_insights": {"layers": []},
            },
            "delta_speedup": 0.19,
            "policy_diff": {
                "changed_layers": [
                    {
                        "layer_idx": 1,
                        "changes": {
                            "turbo_quantize_v": {"base": False, "compare": True},
                        },
                    }
                ]
            },
        }

    monkeypatch.setattr("vitriol.cli.commands.bench.compare_long_context_preset", fake_compare_long_context_preset)

    result = runner.invoke(
        cli,
        [
            "bench",
            "kv-long",
            "demo/model",
            "--compare-preset",
            "aggressive",
            "--format",
            "markdown",
        ],
    )

    assert result.exit_code == 0
    assert "## Experiment Metadata" in result.output
    assert "## KV Long Compare" in result.output
    assert "`delta_speedup`: 0.190x" in result.output
    assert "`delta_estimated_kv_mb`: -18.830" in result.output
    assert "`compare_avg_residual_l2`: 0.230" in result.output
    assert "### Policy Changes" in result.output
    assert "turbo_quantize_v" in result.output


def test_bench_kv_report_summary(monkeypatch) -> None:
    runner = CliRunner()

    monkeypatch.setattr(
        "vitriol.cli.commands.bench.compare_smoke",
        lambda **kwargs: {
            "delta_speedup": 0.12,
            "base": {"exact": True, "tuned_memory": {"estimated_kv_megabytes": 48.19, "peak_device_megabytes": 6350.98}},
            "compare": {"exact": False, "tuned_memory": {"estimated_kv_megabytes": 29.36, "peak_device_megabytes": 6390.98}},
            "policy_diff": {"changed_layers": []},
        },
    )
    monkeypatch.setattr(
        "vitriol.cli.commands.bench.compare_long_context_preset",
        lambda **kwargs: {
            "delta_speedup": 0.24,
            "base": {"tuned_exact": True, "tuned_memory": {"estimated_kv_megabytes": 48.19, "peak_device_megabytes": 6350.98}},
            "compare": {"tuned_exact": True, "tuned_memory": {"estimated_kv_megabytes": 29.36, "peak_device_megabytes": 6390.98}},
            "policy_diff": {"changed_layers": []},
        },
    )
    monkeypatch.setattr(
        "vitriol.cli.commands.bench.compare_short_suite",
        lambda cfg, compare_preset, compare_preset_params=None: {
            "case_diffs": [
                {
                    "name": "pt512:qa",
                    "base_speedup": 1.02,
                    "compare_speedup": 1.20,
                    "delta_speedup": 0.18,
                    "base_exact": True,
                    "compare_exact": True,
                }
            ],
            "policy_diff": {
                "changed_layers": [
                    {
                        "layer_idx": 0,
                        "changes": {"enable_sparse_v": {"base": False, "compare": True}},
                    }
                ]
            },
        },
    )

    result = runner.invoke(cli, ["bench", "kv-report", "demo/model", "--format", "summary", "--show-layers"])
    assert result.exit_code == 0
    assert "model: demo/model" in result.output
    assert "smoke:" in result.output
    assert "base_estimated_kv_mb: 48.190" in result.output
    assert "long:" in result.output
    assert "delta_peak_device_mb: 40.000" in result.output
    assert "suite:" in result.output
    assert "pt512:qa" in result.output
    assert "suite policy changes:" in result.output


def test_bench_kv_report_markdown(monkeypatch) -> None:
    runner = CliRunner()

    monkeypatch.setattr(
        "vitriol.cli.commands.bench.compare_smoke",
        lambda **kwargs: {
            "delta_speedup": 0.10,
            "base": {"exact": True, "tuned_memory": {"estimated_kv_megabytes": 48.19, "peak_device_megabytes": 6350.98}},
            "compare": {"exact": True, "tuned_memory": {"estimated_kv_megabytes": 29.36, "peak_device_megabytes": 6390.98}},
            "policy_diff": {"changed_layers": []},
        },
    )
    monkeypatch.setattr(
        "vitriol.cli.commands.bench.compare_long_context_preset",
        lambda **kwargs: {
            "delta_speedup": 0.21,
            "base": {"tuned_exact": True, "tuned_memory": {"estimated_kv_megabytes": 48.19, "peak_device_megabytes": 6350.98}},
            "compare": {"tuned_exact": False, "tuned_memory": {"estimated_kv_megabytes": 29.36, "peak_device_megabytes": 6390.98}},
            "policy_diff": {"changed_layers": []},
        },
    )
    monkeypatch.setattr(
        "vitriol.cli.commands.bench.compare_short_suite",
        lambda cfg, compare_preset, compare_preset_params=None: {
            "case_diffs": [
                {
                    "name": "pt2048:chat",
                    "base_speedup": 1.05,
                    "compare_speedup": 1.27,
                    "delta_speedup": 0.22,
                    "base_exact": True,
                    "compare_exact": False,
                }
            ],
            "policy_diff": {
                "changed_layers": [
                    {
                        "layer_idx": 1,
                        "changes": {"turbo_quantize_v": {"base": False, "compare": True}},
                    }
                ]
            },
        },
    )

    result = runner.invoke(cli, ["bench", "kv-report", "demo/model", "--format", "markdown", "--show-layers"])
    assert result.exit_code == 0
    assert "## Experiment Metadata" in result.output
    assert "## KV Report" in result.output
    assert "### Smoke" in result.output
    assert "`base_estimated_kv_mb`: 48.190" in result.output
    assert "### Long" in result.output
    assert "`compare_peak_device_mb`: 6390.980" in result.output
    assert "### Suite" in result.output
    assert "| case | base speedup | compare speedup | delta |" in result.output
    assert "### Suite Policy Changes" in result.output


def test_bench_kv_report_output_dir(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setattr(
        "vitriol.cli.commands.bench.compare_smoke",
        lambda **kwargs: {
            "delta_speedup": 0.10,
            "base": {"exact": True},
            "compare": {"exact": True},
            "policy_diff": {"changed_layers": []},
        },
    )
    monkeypatch.setattr(
        "vitriol.cli.commands.bench.compare_long_context_preset",
        lambda **kwargs: {
            "delta_speedup": 0.21,
            "base": {"tuned_exact": True},
            "compare": {"tuned_exact": False},
            "policy_diff": {"changed_layers": []},
        },
    )
    monkeypatch.setattr(
        "vitriol.cli.commands.bench.compare_short_suite",
        lambda cfg, compare_preset, compare_preset_params=None: {
            "case_diffs": [
                {
                    "name": "pt2048:chat",
                    "base_speedup": 1.05,
                    "compare_speedup": 1.27,
                    "delta_speedup": 0.22,
                    "base_exact": True,
                    "compare_exact": False,
                }
            ],
            "policy_diff": {"changed_layers": []},
        },
    )

    output_dir = tmp_path / "report-assets"
    result = runner.invoke(cli, ["bench", "kv-report", "demo/model", "--output-dir", str(output_dir)])
    assert result.exit_code == 0
    json_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    assert json_path.exists()
    assert markdown_path.exists()
    assert str(json_path) in result.output
    assert str(markdown_path) in result.output
    payload = json.loads(json_path.read_text())
    assert payload["model_id"] == "demo/model"
    text = markdown_path.read_text()
    assert "## KV Report" in text
    assert "### Suite" in text


def test_bench_kv_report_rejects_output_and_output_dir(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "bench",
            "kv-report",
            "demo/model",
            "--output",
            str(tmp_path / "report.md"),
            "--output-dir",
            str(tmp_path / "report-assets"),
        ],
    )
    assert result.exit_code != 0
    assert "--output and --output-dir cannot be used together" in result.output


def test_bench_kv_plan_markdown_output_file(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    def fake_build_policy_plan(**kwargs):
        return {
            "model_id": kwargs["model_id"],
            "preset": {"name": kwargs["preset"]},
            "chosen_v_quantize_only_first_n": 1,
            "policy_insights": {
                "quantized_kv_start": 128,
                "counts": {
                    "full_attention": 2,
                    "sliding_window": 0,
                    "mla": 0,
                    "linear_attention": 0,
                    "turbo_k": 2,
                    "turbo_v": 1,
                    "sparse_v": 0,
                    "compute_skip": 0,
                },
                "layers": [],
            },
        }

    monkeypatch.setattr("vitriol.cli.commands.bench.build_policy_plan", fake_build_policy_plan)

    output_path = tmp_path / "plan.md"
    result = runner.invoke(
        cli,
        ["bench", "kv-plan", "demo/model", "--format", "markdown", "--output", str(output_path)],
    )

    assert result.exit_code == 0
    text = output_path.read_text()
    assert "## Experiment Metadata" in text
    assert "`command`: bench kv-plan" in text
    assert "`format`: markdown" in text
    assert "## KV Plan" in text
    assert "`quantized_kv_start`" in text
