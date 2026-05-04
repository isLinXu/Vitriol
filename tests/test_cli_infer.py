import json

from click.testing import CliRunner

from vitriol.cli.main import cli


def test_infer_command_is_listed() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "infer" in result.output


def test_infer_invokes_runner(monkeypatch) -> None:
    runner = CliRunner()
    captured = {}

    def fake_run_generate_preset(**kwargs):
        captured.update(kwargs)
        return {
            "model_id": kwargs["model_id"],
            "preset": {"name": kwargs["preset"]},
            "generated_text": "hello",
            "decode_toks_per_s": 12.34,
            "prompt_tokens": 3,
            "decode_tokens": 1,
            "chosen_v_quantize_only_first_n": 2,
        }

    monkeypatch.setattr("vitriol.cli.commands.infer.run_generate_preset", fake_run_generate_preset)

    result = runner.invoke(
        cli,
        [
            "infer",
            "demo/model",
            "--prompt",
            "hi there",
            "--preset",
            "fast-balanced",
            "--preset-param",
            "quantized_kv_start=0",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["generated_text"] == "hello"
    assert captured["prompt"] == "hi there"
    assert captured["preset"] == "fast-balanced"
    assert captured["preset_params"]["quantized_kv_start"] == 0


def test_infer_prompt_file_and_summary(monkeypatch, tmp_path) -> None:
    runner = CliRunner()
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("from file", encoding="utf-8")

    monkeypatch.setattr(
        "vitriol.cli.commands.infer.run_generate_preset",
        lambda **kwargs: {
            "model_id": kwargs["model_id"],
            "preset": {"name": kwargs["preset"]},
            "generated_text": "OK",
            "decode_toks_per_s": 9.87,
            "prompt_tokens": 5,
            "decode_tokens": 1,
            "chosen_v_quantize_only_first_n": 2,
        },
    )

    result = runner.invoke(
        cli,
        [
            "infer",
            "demo/model",
            "--prompt-file",
            str(prompt_file),
            "--format",
            "summary",
        ],
    )

    assert result.exit_code == 0
    assert "model: demo/model" in result.output
    assert "preset: balanced" in result.output
    assert "generated_text:" in result.output
    assert "OK" in result.output


def test_infer_smoke_invokes_smoke_runner(monkeypatch) -> None:
    runner = CliRunner()
    captured = {}

    def fake_run_smoke(**kwargs):
        captured.update(kwargs)
        return {
            "model_id": kwargs["model_id"],
            "preset": {"name": kwargs["preset"]},
            "ok": True,
            "tuned_exact": True,
            "tuned_speedup": 1.1,
            "chosen_v_quantize_only_first_n": 1,
        }

    def fail_generate(**_kwargs):
        raise AssertionError("run_generate_preset should not be called for --smoke")

    monkeypatch.setattr("vitriol.cli.commands.infer.run_smoke", fake_run_smoke)
    monkeypatch.setattr("vitriol.cli.commands.infer.run_generate_preset", fail_generate)

    result = runner.invoke(
        cli,
        [
            "infer",
            "demo/model",
            "--smoke",
            "--preset",
            "aggressive",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert captured["model_id"] == "demo/model"
    assert captured["preset"] == "aggressive"


def test_infer_smoke_forwards_trust_remote_code(monkeypatch) -> None:
    runner = CliRunner()
    captured = {}

    def fake_run_smoke(**kwargs):
        captured.update(kwargs)
        return {"model_id": kwargs["model_id"], "ok": True, "preset": {"name": kwargs["preset"]}}

    monkeypatch.setattr("vitriol.cli.commands.infer.run_smoke", fake_run_smoke)

    result = runner.invoke(
        cli,
        [
            "--no-trust-remote-code",
            "infer",
            "demo/model",
            "--smoke",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    assert captured["trust_remote_code"] is False


def test_infer_chat_renders_system_prompt_and_assistant_prefix(monkeypatch) -> None:
    runner = CliRunner()
    captured = {}

    monkeypatch.setattr(
        "vitriol.cli.commands.infer._build_chat_prompt",
        lambda **kwargs: f"system::{kwargs['system_prompt']}\nuser::{kwargs['user_prompt']}\nassistant::{kwargs['assistant_prefix']}",
        raising=False,
    )

    def fake_run_generate_preset(**kwargs):
        captured.update(kwargs)
        return {
            "model_id": kwargs["model_id"],
            "preset": {"name": kwargs["preset"]},
            "generated_text": "answer",
            "decode_toks_per_s": 1.0,
            "prompt_tokens": 1,
            "decode_tokens": 1,
            "chosen_v_quantize_only_first_n": 1,
        }

    monkeypatch.setattr("vitriol.cli.commands.infer.run_generate_preset", fake_run_generate_preset)

    result = runner.invoke(
        cli,
        [
            "infer",
            "demo/model",
            "--prompt",
            "介绍 TurboQuant",
            "--chat",
            "--system-prompt",
            "你是中文助手",
            "--assistant-prefix",
            "TurboQuant",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    assert "system::你是中文助手" in captured["prompt"]
    assert "user::介绍 TurboQuant" in captured["prompt"]
    assert captured["prompt"].endswith("assistant::TurboQuant")


def test_infer_strip_think_removes_think_block_from_text_output(monkeypatch) -> None:
    runner = CliRunner()

    monkeypatch.setattr(
        "vitriol.cli.commands.infer.run_generate_preset",
        lambda **kwargs: {
            "model_id": kwargs["model_id"],
            "preset": {"name": kwargs["preset"]},
            "generated_text": "<think>internal</think>最终答案",
            "decode_toks_per_s": 1.0,
            "prompt_tokens": 1,
            "decode_tokens": 1,
            "chosen_v_quantize_only_first_n": 1,
        },
    )

    result = runner.invoke(
        cli,
        ["infer", "demo/model", "--prompt", "hi", "--strip-think"],
    )

    assert result.exit_code == 0
    assert "internal" not in result.output
    assert "最终答案" in result.output


def test_infer_qwen_chat_applies_default_chat_settings(monkeypatch) -> None:
    runner = CliRunner()
    captured = {}

    def fake_build_chat_prompt(**kwargs):
        captured.update(kwargs)
        return "rendered-chat-prompt"

    monkeypatch.setattr("vitriol.cli.commands.infer._build_chat_prompt", fake_build_chat_prompt)
    monkeypatch.setattr(
        "vitriol.cli.commands.infer.run_generate_preset",
        lambda **kwargs: {
            "model_id": kwargs["model_id"],
            "preset": {"name": kwargs["preset"]},
            "generated_text": "<think>hidden</think>答案",
            "decode_toks_per_s": 1.0,
            "prompt_tokens": 1,
            "decode_tokens": 1,
            "chosen_v_quantize_only_first_n": 1,
        },
    )

    result = runner.invoke(
        cli,
        ["infer", "demo/model", "--prompt", "介绍 TurboQuant", "--qwen-chat"],
    )

    assert result.exit_code == 0
    assert (
        captured["system_prompt"]
        == "You are a concise Chinese assistant. Output exactly one Chinese sentence as the conclusion. Do not output any thinking process, tags, Markdown, or English."
    )
    assert captured["assistant_prefix"] == "TurboQuant"
    assert "hidden" not in result.output
    assert "答案" in result.output


def test_infer_qwen_chat_allows_explicit_prompt_overrides(monkeypatch) -> None:
    runner = CliRunner()
    captured = {}

    def fake_build_chat_prompt(**kwargs):
        captured.update(kwargs)
        return "rendered-chat-prompt"

    monkeypatch.setattr("vitriol.cli.commands.infer._build_chat_prompt", fake_build_chat_prompt)
    monkeypatch.setattr(
        "vitriol.cli.commands.infer.run_generate_preset",
        lambda **kwargs: {
            "model_id": kwargs["model_id"],
            "preset": {"name": kwargs["preset"]},
            "generated_text": "答案",
            "decode_toks_per_s": 1.0,
            "prompt_tokens": 1,
            "decode_tokens": 1,
            "chosen_v_quantize_only_first_n": 1,
        },
    )

    result = runner.invoke(
        cli,
        [
            "infer",
            "demo/model",
            "--prompt",
            "介绍 TurboQuant",
            "--qwen-chat",
            "--system-prompt",
            "自定义 system",
            "--assistant-prefix",
            "自定义前缀",
        ],
    )

    assert result.exit_code == 0
    assert captured["system_prompt"] == "自定义 system"
    assert captured["assistant_prefix"] == "自定义前缀"


def test_infer_qwen_chat_keep_think_overrides_default_strip(monkeypatch) -> None:
    runner = CliRunner()

    monkeypatch.setattr(
        "vitriol.cli.commands.infer.run_generate_preset",
        lambda **kwargs: {
            "model_id": kwargs["model_id"],
            "preset": {"name": kwargs["preset"]},
            "generated_text": "<think>hidden</think>答案",
            "decode_toks_per_s": 1.0,
            "prompt_tokens": 1,
            "decode_tokens": 1,
            "chosen_v_quantize_only_first_n": 1,
        },
    )
    monkeypatch.setattr("vitriol.cli.commands.infer._build_chat_prompt", lambda **kwargs: "rendered")

    result = runner.invoke(
        cli,
        ["infer", "demo/model", "--prompt", "介绍 TurboQuant", "--qwen-chat", "--keep-think"],
    )

    assert result.exit_code == 0
    assert "hidden" in result.output


def test_infer_qwen_chat_rejects_smoke_mode() -> None:
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["infer", "demo/model", "--smoke", "--qwen-chat", "--format", "json"],
    )

    assert result.exit_code != 0
    assert "--qwen-chat only works with prompt generation" in result.output


def test_infer_qwen_chat_preset_enables_qwen_shortcut_and_quant_start(monkeypatch) -> None:
    runner = CliRunner()
    captured = {}

    monkeypatch.setattr("vitriol.cli.commands.infer._build_chat_prompt", lambda **kwargs: "rendered")

    def fake_run_generate_preset(**kwargs):
        captured.update(kwargs)
        return {
            "model_id": kwargs["model_id"],
            "preset": {"name": kwargs["preset"]},
            "generated_text": "答案",
            "decode_toks_per_s": 1.0,
            "prompt_tokens": 1,
            "decode_tokens": 1,
            "chosen_v_quantize_only_first_n": 1,
        }

    monkeypatch.setattr("vitriol.cli.commands.infer.run_generate_preset", fake_run_generate_preset)

    result = runner.invoke(
        cli,
        ["infer", "demo/model", "--prompt", "介绍 TurboQuant", "--preset", "qwen-chat", "--format", "json"],
    )

    assert result.exit_code == 0
    assert captured["preset"] == "aggressive"
    assert captured["preset_params"]["quantized_kv_start"] == 0


def test_infer_qwen_chat_preset_keeps_user_preset_param_override(monkeypatch) -> None:
    runner = CliRunner()
    captured = {}

    monkeypatch.setattr("vitriol.cli.commands.infer._build_chat_prompt", lambda **kwargs: "rendered")

    def fake_run_generate_preset(**kwargs):
        captured.update(kwargs)
        return {
            "model_id": kwargs["model_id"],
            "preset": {"name": kwargs["preset"]},
            "generated_text": "答案",
            "decode_toks_per_s": 1.0,
            "prompt_tokens": 1,
            "decode_tokens": 1,
            "chosen_v_quantize_only_first_n": 1,
        }

    monkeypatch.setattr("vitriol.cli.commands.infer.run_generate_preset", fake_run_generate_preset)

    result = runner.invoke(
        cli,
        [
            "infer",
            "demo/model",
            "--prompt",
            "介绍 TurboQuant",
            "--preset",
            "qwen-chat",
            "--preset-param",
            "quantized_kv_start=32",
        ],
    )

    assert result.exit_code == 0
    assert captured["preset_params"]["quantized_kv_start"] == 32


def test_infer_qwen_chat_preset_applies_default_chat_prompting(monkeypatch) -> None:
    runner = CliRunner()
    captured = {}

    def fake_build_chat_prompt(**kwargs):
        captured.update(kwargs)
        return "rendered"

    monkeypatch.setattr("vitriol.cli.commands.infer._build_chat_prompt", fake_build_chat_prompt)
    monkeypatch.setattr(
        "vitriol.cli.commands.infer.run_generate_preset",
        lambda **kwargs: {
            "model_id": kwargs["model_id"],
            "preset": {"name": kwargs["preset"]},
            "generated_text": "<think>hidden</think>答案",
            "decode_toks_per_s": 1.0,
            "prompt_tokens": 1,
            "decode_tokens": 1,
            "chosen_v_quantize_only_first_n": 1,
        },
    )

    result = runner.invoke(
        cli,
        ["infer", "demo/model", "--prompt", "介绍 TurboQuant", "--preset", "qwen-chat"],
    )

    assert result.exit_code == 0
    assert (
        captured["system_prompt"]
        == "You are a concise Chinese assistant. Output exactly one Chinese sentence as the conclusion. Do not output any thinking process, tags, Markdown, or English."
    )
    assert captured["assistant_prefix"] == "TurboQuant"
    assert "hidden" not in result.output


def test_infer_summary_show_stats(monkeypatch) -> None:
    runner = CliRunner()

    monkeypatch.setattr(
        "vitriol.cli.commands.infer.run_generate_preset",
        lambda **kwargs: {
            "model_id": kwargs["model_id"],
            "device": "mps",
            "dtype": "torch.float16",
            "preset": {"name": kwargs["preset"]},
            "generated_text": "OK",
            "decode_toks_per_s": 9.87,
            "prefill_s": 0.123,
            "decode_s": 0.456,
            "prompt_tokens": 5,
            "decode_tokens": 1,
            "chosen_v_quantize_only_first_n": 2,
            "policy_insights": {
                "quantized_kv_start": 0,
                "counts": {"full_attention": 6, "linear_attention": 18, "turbo_k": 6, "turbo_v": 2},
            },
            "tuned_memory": {"estimated_kv_megabytes": 12.5, "peak_device_megabytes": 256.0},
            "tuned_turboquant": {"calls": 6, "correction_to_residual_l2_ratio": 0.5},
        },
    )

    result = runner.invoke(
        cli,
        [
            "infer",
            "demo/model",
            "--prompt",
            "hello",
            "--format",
            "summary",
            "--show-stats",
        ],
    )

    assert result.exit_code == 0
    assert "stats:" in result.output
    assert "device: mps" in result.output
    assert "quantized_kv_start: 0" in result.output
    assert "estimated_kv_mb: 12.50" in result.output
    assert "peak_device_mb: 256.00" in result.output
    assert "peak_minus_estimated_mb: 243.50" in result.output
    assert "turboquant_calls: 6" in result.output


def test_infer_text_show_stats(monkeypatch) -> None:
    runner = CliRunner()

    monkeypatch.setattr(
        "vitriol.cli.commands.infer.run_generate_preset",
        lambda **kwargs: {
            "model_id": kwargs["model_id"],
            "device": "mps",
            "dtype": "torch.float16",
            "preset": {"name": kwargs["preset"]},
            "generated_text": "hello world",
            "decode_toks_per_s": 9.87,
            "prefill_s": 0.123,
            "decode_s": 0.456,
            "prompt_tokens": 5,
            "decode_tokens": 1,
            "chosen_v_quantize_only_first_n": 2,
            "policy_insights": {"quantized_kv_start": 0, "counts": {}},
            "tuned_memory": {"estimated_kv_megabytes": 12.5},
            "tuned_turboquant": {"calls": 6, "correction_to_residual_l2_ratio": 0.5},
        },
    )

    result = runner.invoke(
        cli,
        [
            "infer",
            "demo/model",
            "--prompt",
            "hello",
            "--show-stats",
        ],
    )

    assert result.exit_code == 0
    assert "hello world" in result.output


def test_infer_hy3_summary_smoke_shape(monkeypatch) -> None:
    runner = CliRunner()

    monkeypatch.setattr(
        "vitriol.cli.commands.infer.run_generate_preset",
        lambda **kwargs: {
            "model_id": kwargs["model_id"],
            "device": "mps",
            "dtype": "torch.float16",
            "preset": {"name": kwargs["preset"]},
            "generated_text": "!!!!!!!!!!!!",
            "decode_toks_per_s": 3.4265,
            "prefill_s": 3.4212,
            "decode_s": 3.5022,
            "prompt_tokens": 10,
            "decode_tokens": 12,
            "chosen_v_quantize_only_first_n": 0,
            "policy_insights": {
                "quantized_kv_start": 0,
                "counts": {"full_attention": 2, "linear_attention": 0, "turbo_k": 0, "turbo_v": 0},
            },
            "tuned_memory": {"estimated_kv_megabytes": 0.04, "peak_device_megabytes": 1050.83},
            "tuned_turboquant": {"calls": 0, "correction_to_residual_l2_ratio": 0.0},
        },
    )

    result = runner.invoke(
        cli,
        [
            "--trust-remote-code",
            "infer",
            "output/hy3_preview_ultra_final",
            "--prompt",
            "你好，请只用一句中文介绍这个模型。",
            "--preset",
            "safe",
            "--max-new-tokens",
            "12",
            "--format",
            "summary",
            "--show-stats",
        ],
    )

    assert result.exit_code == 0
    assert "model: output/hy3_preview_ultra_final" in result.output
    assert "preset: safe" in result.output
    assert "generated_text:" in result.output
    assert "!!!!!!!!!!!!" in result.output
    assert "policy_counts: full=2, sliding=0, compressed=0, hash=0, linear=0, turbo_k=0, turbo_v=0" in result.output
    assert "stats:" in result.output
    assert "turboquant_correction_ratio: 0.000000" in result.output


def test_infer_accepts_model_specific_presets(monkeypatch) -> None:
    runner = CliRunner()
    captured = []

    def fake_run_generate_preset(**kwargs):
        captured.append(kwargs)
        return {
            "model_id": kwargs["model_id"],
            "preset": {"name": kwargs["preset"]},
            "generated_text": "ok",
            "decode_toks_per_s": 1.0,
            "prompt_tokens": 1,
            "decode_tokens": 1,
            "chosen_v_quantize_only_first_n": 1,
        }

    monkeypatch.setattr("vitriol.cli.commands.infer.run_generate_preset", fake_run_generate_preset)

    for preset in ("deepseek-v4", "hy3"):
        result = runner.invoke(
            cli,
            ["infer", "demo/model", "--prompt", "hi", "--preset", preset, "--format", "json"],
        )
        assert result.exit_code == 0

    assert [call["preset"] for call in captured] == ["deepseek-v4", "hy3"]
