import json
import re
from pathlib import Path
from typing import Any, Optional

import click
from click.core import ParameterSource

from ...utils.experimental import experimental
from ...utils.hf_loading import load_tokenizer as hf_load_tokenizer

_PRESET_CHOICES = ["safe", "balanced", "fast-balanced", "aggressive", "ultra-long", "deepseek-v4", "hy3", "qwen-chat"]
_QWEN_CHAT_SYSTEM_PROMPT = (
    "You are a concise Chinese assistant. Output exactly one Chinese sentence as the conclusion. "
    "Do not output any thinking process, tags, Markdown, or English."
)
_QWEN_CHAT_ASSISTANT_PREFIX = "TurboQuant"


def run_generate_preset(*args, **kwargs) -> Any:
    from ...bench.runner import run_generate_preset as _run_generate_preset

    return _run_generate_preset(*args, **kwargs)


def run_smoke(*args, **kwargs) -> Any:
    from ...bench.runner import run_smoke as _run_smoke

    return _run_smoke(*args, **kwargs)


def _parse_scalar(value: str):
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_preset_params(values: tuple[str, ...]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for item in values:
        if "=" not in item:
            raise click.BadParameter(f"Invalid preset param '{item}'. Expected key=value.")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise click.BadParameter(f"Invalid preset param '{item}'. Empty key.")
        parsed[key] = _parse_scalar(raw.strip())
    return parsed


def _resolve_prompt(prompt: Optional[str], prompt_file: Optional[str]) -> str:
    if bool(prompt) == bool(prompt_file):
        raise click.BadParameter("Provide exactly one of --prompt or --prompt-file")
    if prompt_file:
        return Path(prompt_file).read_text(encoding="utf-8")
    return str(prompt)


def _build_chat_prompt(
    *,
    model_id: str,
    user_prompt: str,
    system_prompt: Optional[str],
    assistant_prefix: Optional[str],
    trust_remote_code: bool,
) -> str:
    ctx = click.get_current_context(silent=True)
    allow_network = bool((ctx.obj or {}).get("allow_network", True)) if ctx else True
    local_files_only = bool((ctx.obj or {}).get("local_files_only", False)) if ctx else False
    tokenizer = hf_load_tokenizer(
        model_id,
        security={
            "trust_remote_code": trust_remote_code,
            "allow_network": allow_network,
            "local_files_only": local_files_only,
        },
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if assistant_prefix:
        rendered = f"{rendered}{assistant_prefix}"
    return rendered


def _strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def _apply_infer_preset_overrides(
    *,
    preset: str,
    preset_params: dict[str, object],
    qwen_chat: bool,
) -> tuple[str, dict[str, object], bool]:
    effective_preset = preset
    effective_params = dict(preset_params)
    effective_qwen_chat = qwen_chat
    if preset == "qwen-chat":
        effective_preset = "aggressive"
        effective_qwen_chat = True
        effective_params.setdefault("quantized_kv_start", 0)
    return effective_preset, effective_params, effective_qwen_chat


def _summary_text(result: dict[str, object]) -> str:
    if not result.get("ok", True):
        return f"Error running inference for {result.get('model_id', '-')}:\n{result.get('error', 'Unknown error')}"

    preset = (result.get("preset") or {}) if isinstance(result.get("preset"), dict) else {}
    lines = [
        f"model: {result.get('model_id', '-')}",
        f"preset: {preset.get('name', '-')}",
        f"prompt_tokens: {result.get('prompt_tokens', '-')}",
        f"decode_tokens: {result.get('decode_tokens', '-')}",
        f"decode_toks_per_s: {float(result.get('decode_toks_per_s', 0.0) or 0.0):.4f}",
        f"chosen_v_quant_layers: {result.get('chosen_v_quantize_only_first_n', '-')}",
        "",
        "generated_text:",
        str(result.get("generated_text", "")),
    ]
    return "\n".join(lines)


def _stats_text(result: dict[str, object]) -> str:
    if not result.get("ok", True):
        return ""
    policy_insights = result.get("policy_insights") or {}
    counts = (policy_insights.get("counts") or {}) if isinstance(policy_insights, dict) else {}
    tuned_memory = result.get("tuned_memory") or {}
    tuned_turboquant = result.get("tuned_turboquant") or {}
    estimated_kv_mb = tuned_memory.get("estimated_kv_megabytes")
    peak_device_mb = tuned_memory.get("peak_device_megabytes")
    lines = [
        "stats:",
        f"device: {result.get('device', '-')}",
        f"dtype: {result.get('dtype', '-')}",
        f"prefill_s: {float(result.get('prefill_s', 0.0) or 0.0):.4f}",
        f"decode_s: {float(result.get('decode_s', 0.0) or 0.0):.4f}",
        f"decode_toks_per_s: {float(result.get('decode_toks_per_s', 0.0) or 0.0):.4f}",
        f"quantized_kv_start: {policy_insights.get('quantized_kv_start', '-')}",
        f"chosen_v_quant_layers: {result.get('chosen_v_quantize_only_first_n', '-')}",
        (
            f"policy_counts: full={counts.get('full_attention', 0)}, sliding={counts.get('sliding_window', 0)}, "
            f"compressed={counts.get('compressed_attention', 0)}, hash={counts.get('hash_attention', 0)}, "
            f"linear={counts.get('linear_attention', 0)}, turbo_k={counts.get('turbo_k', 0)}, turbo_v={counts.get('turbo_v', 0)}"
        ),
        f"estimated_kv_mb: {float(estimated_kv_mb or 0.0):.2f}",
        f"peak_device_mb: {float(peak_device_mb or 0.0):.2f}" if peak_device_mb is not None else "peak_device_mb: -",
        f"turboquant_calls: {int(tuned_turboquant.get('calls', 0) or 0)}",
        f"turboquant_correction_ratio: {float(tuned_turboquant.get('correction_to_residual_l2_ratio', 0.0) or 0.0):.6f}",
    ]
    if estimated_kv_mb is not None and peak_device_mb is not None:
        lines.insert(-2, f"peak_minus_estimated_mb: {float(peak_device_mb) - float(estimated_kv_mb):.2f}")

    # KV hook 统计（只读展示；不影响推理流程）
    try:
        from ...patches.cache_hooks import get_cache_hook_stats

        hook_stats = get_cache_hook_stats()
        lines.append("kv_hook_stats:")
        for k in sorted(hook_stats.keys()):
            lines.append(f"  {k}: {hook_stats[k]}")
    except Exception:
        lines.append("kv_hook_stats: -")
    return "\n".join(lines)


def _smoke_summary_text(result: dict[str, object]) -> str:
    if not result.get("ok", True):
        return f"Error running smoke test for {result.get('model_id', '-')}:\n{result.get('error', 'Unknown error')}"

    prefix_match = result.get("prefix_match", result.get("tuned_prefix_match"))
    lines = [
        f"model: {result.get('model_id', '-')}",
        f"preset: {(result.get('preset') or {}).get('name', '-') if isinstance(result.get('preset'), dict) else result.get('preset', '-')}",
        f"ok: {result.get('ok', '-')}",
        f"exact: {result.get('exact', result.get('tuned_exact', '-'))}",
        f"speedup: {float(result.get('speedup', result.get('tuned_speedup', 0.0)) or 0.0):.3f}x",
        f"chosen_v_quant_layers: {result.get('chosen_v_quantize_only_first_n', '-')}",
    ]
    if isinstance(prefix_match, (list, tuple)) and len(prefix_match) == 3:
        lines.append(f"prefix_match: {prefix_match[0]}/{prefix_match[1]} ({float(prefix_match[2]):.1f}%)")
    return "\n".join(lines)


@experimental("TurboQuant single-prompt inference CLI", detail="Research path; not for production serving.")
@click.command(name="infer")
@click.argument("model_id")
@click.option("--prompt", type=str, help="Prompt text to run")
@click.option("--prompt-file", type=click.Path(exists=True, dir_okay=False, path_type=str), help="Read prompt from a UTF-8 text file")
@click.option("--preset", type=click.Choice(_PRESET_CHOICES), default="balanced", show_default=True)
@click.option("--smoke", is_flag=True, help="Run the existing smoke benchmark path instead of single-prompt generation")
@click.option("--qwen-chat", is_flag=True, help="Shortcut for Qwen-friendly chat rendering and think stripping")
@click.option("--chat", is_flag=True, help="Render the input as a chat prompt via the tokenizer chat template")
@click.option("--system-prompt", type=str, help="Optional system message used with --chat")
@click.option("--assistant-prefix", type=str, help="Optional assistant prefix appended after the rendered chat prompt")
@click.option("--strip-think/--keep-think", default=False, help="Remove <think>...</think> blocks from displayed text")
@click.option("--preset-param", "preset_params", multiple=True, help="Override preset params with key=value")
@click.option("--max-new-tokens", type=int, default=64, show_default=True)
@click.option("--calib-new-tokens", type=int, default=8, show_default=True)
@click.option("--search-max-n", type=int, default=2, show_default=True)
@click.option("--show-stats", is_flag=True, help="Show TurboQuant/runtime stats with the generation output")
@click.option("--format", "fmt", type=click.Choice(["text", "summary", "json"]), default="text", show_default=True)
@click.pass_context
def infer(
    ctx: click.Context,
    model_id: str,
    prompt: Optional[str],
    prompt_file: Optional[str],
    preset: str,
    smoke: bool,
    qwen_chat: bool,
    chat: bool,
    system_prompt: Optional[str],
    assistant_prefix: Optional[str],
    strip_think: bool,
    preset_params: tuple[str, ...],
    max_new_tokens: int,
    calib_new_tokens: int,
    search_max_n: int,
    show_stats: bool,
    fmt: str,
) -> None:
    """Run single-prompt inference with Vitriol TurboQuant presets."""
    trust_remote_code = bool(ctx.obj.get("trust_remote_code", False)) if getattr(ctx, "obj", None) else False
    parsed_preset_params = _parse_preset_params(preset_params)
    effective_preset, parsed_preset_params, effective_qwen_chat = _apply_infer_preset_overrides(
        preset=preset,
        preset_params=parsed_preset_params,
        qwen_chat=qwen_chat,
    )

    if effective_qwen_chat and smoke:
        raise click.UsageError("--qwen-chat only works with prompt generation, not smoke mode")

    strip_think_source = ctx.get_parameter_source("strip_think")
    effective_chat = bool(chat or effective_qwen_chat)
    effective_system_prompt = system_prompt or (_QWEN_CHAT_SYSTEM_PROMPT if effective_qwen_chat else None)
    effective_assistant_prefix = assistant_prefix or (_QWEN_CHAT_ASSISTANT_PREFIX if effective_qwen_chat else None)
    effective_strip_think = strip_think
    if effective_qwen_chat and strip_think_source == ParameterSource.DEFAULT:
        effective_strip_think = True
    if smoke:
        result = run_smoke(
            model_id=model_id,
            preset=effective_preset,
            max_new_tokens=int(max_new_tokens),
            calib_new_tokens=int(calib_new_tokens),
            search_max_n=int(search_max_n),
            preset_params=parsed_preset_params,
            trust_remote_code=trust_remote_code,
        )
    else:
        resolved_prompt = _resolve_prompt(prompt, prompt_file)
        if effective_chat:
            resolved_prompt = _build_chat_prompt(
                model_id=model_id,
                user_prompt=resolved_prompt,
                system_prompt=effective_system_prompt,
                assistant_prefix=effective_assistant_prefix,
                trust_remote_code=trust_remote_code,
            )
        result = run_generate_preset(
            model_id=model_id,
            prompt=resolved_prompt,
            preset=effective_preset,
            max_new_tokens=int(max_new_tokens),
            calib_new_tokens=int(calib_new_tokens),
            search_max_n=int(search_max_n),
            preset_params=parsed_preset_params,
            trust_remote_code=trust_remote_code,
        )
    if fmt == "json":
        if isinstance(result, dict):
            result.setdefault("kv", {})
        click.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if fmt == "summary":
        text = _smoke_summary_text(result) if smoke else _summary_text(result)
        if effective_strip_think and not smoke and result.get("ok", True):
            text = text.replace(str(result.get("generated_text", "")), _strip_think_blocks(str(result.get("generated_text", ""))))
        if show_stats:
            text = f"{text}\n\n{_stats_text(result)}"
        click.echo(text)
        return

    if not result.get("ok", True):
        click.echo(f"Error running inference for {result.get('model_id', '-')}:\n{result.get('error', 'Unknown error')}")
        return

    text = _smoke_summary_text(result) if smoke else str(result.get("generated_text", ""))
    if effective_strip_think and not smoke:
        text = _strip_think_blocks(text)
    if show_stats:
        text = f"{text}\n\n{_stats_text(result)}"
    click.echo(text)
