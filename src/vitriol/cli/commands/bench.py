"""
Bench CLI commands for Vitriol KV cache benchmarking system.

Provides Click-based CLI commands for:
  - Running benchmark suites (smoke, short, long-context)
  - Comparing different KV policy presets
  - Analyzing KV quantization quality per-layer
  - Building and diffing KV policy plans
  - TurboQuantum synthetic benchmarks

Output formats supported: JSON, Markdown, plain text.
"""

from typing import Any, Dict, Optional

import click

from ...bench import (
    RunConfig,
    analyze_kv_quantization,
    build_policy_plan,
    compare_long_context_preset,
    compare_short_suite,
    compare_smoke,
    diff_policy_plans,
    run_long_context_preset,
    run_short_suite,
    run_smoke,
)

try:
    from ...bench import (
        compare_turboquantum_modes,
        run_turboquantum_on_model_kv,
        run_turboquantum_synthetic,
    )
    _HAS_TURBOQUANTUM = True
except ImportError:
    _HAS_TURBOQUANTUM = False
    run_turboquantum_synthetic = None
    compare_turboquantum_modes = None
    run_turboquantum_on_model_kv = None

# Valid preset choices for --preset CLI option
from .bench_format import (
    _KV_ANALYZE_SORT_CHOICES,
    _emit_dual_report_files,
    _emit_formatted_result,
    _emit_result,
    _emit_text,
    _fmt_float,
    _markdown_for_report,
    _markdown_for_turboquantum_synthetic,
    _markdown_metadata,
    _turboquantum_mode_comparison_table,
    _turboquantum_summary_lines,
)

_PRESET_CHOICES = ["safe", "balanced", "fast-balanced", "aggressive", "ultra-long", "deepseek-v4", "hy3"]
# Valid TurboQuantum mode choices for --mode CLI option
_TURBOQUANTUM_MODES = ["conservative", "balanced", "aggressive", "ultra-long"]


def _parse_preset_params(values: tuple[str, ...]) -> Dict[str, Any]:
    """Parse CLI --param key=value pairs into a typed dict.

    Supports bool (true/false), int, float, and string values.
    Raises click.BadParameter on malformed input.
    """
    params: Dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise click.BadParameter(f"Invalid preset param '{item}', expected key=value")
        key, raw = item.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if not key:
            raise click.BadParameter(f"Invalid preset param '{item}', empty key")
        lowered = raw.lower()
        if lowered in {"true", "false"}:
            value: Any = lowered == "true"
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        params[key] = value
    return params


@click.group(name="bench")
def bench_group() -> None:
    """Benchmark KV cache inference presets."""


@bench_group.command(name="kv-plan")
@click.argument("model_id")
@click.option("--preset", type=click.Choice(_PRESET_CHOICES), default="balanced", show_default=True)
@click.option("--compare-preset", type=click.Choice(_PRESET_CHOICES))
@click.option("--preset-param", "preset_params", multiple=True, help="Override base preset params with key=value")
@click.option("--compare-preset-param", "compare_preset_params", multiple=True, help="Override compare preset params with key=value")
@click.option("--format", "fmt", type=click.Choice(["json", "summary", "markdown"]), default="json", show_default=True)
@click.option("--show-layers", is_flag=True, help="Show per-layer policy decisions in summary mode")
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=str))
@click.pass_context
def kv_plan(
    ctx: click.Context,
    model_id: str,
    preset: str,
    compare_preset: Optional[str],
    preset_params: tuple[str, ...],
    compare_preset_params: tuple[str, ...],
    fmt: str,
    show_layers: bool,
    output: Optional[str],
) -> None:
    """Build or diff KV policy plans for a model without running inference."""
    trust_remote_code = bool(ctx.obj.get("trust_remote_code", False))
    parsed_preset_params = _parse_preset_params(preset_params)
    parsed_compare_preset_params = _parse_preset_params(compare_preset_params)
    base = build_policy_plan(
        model_id=model_id,
        preset=preset,
        preset_params=parsed_preset_params,
        trust_remote_code=trust_remote_code,
    )
    if compare_preset:
        compare = build_policy_plan(
            model_id=model_id,
            preset=compare_preset,
            preset_params=parsed_compare_preset_params,
            trust_remote_code=trust_remote_code,
        )
        result = diff_policy_plans(base, compare)
        _emit_formatted_result(
            result,
            fmt,
            output=output,
            kind="plan-diff",
            show_layers=show_layers,
            markdown_meta=_markdown_metadata(
                "kv-plan",
                show_layers,
                output,
                extras={
                    "preset": preset,
                    "preset_params": parsed_preset_params,
                    "compare_preset": compare_preset,
                    "compare_preset_params": parsed_compare_preset_params,
                    "trust_remote_code": trust_remote_code,
                },
            ),
        )
        return
    _emit_formatted_result(
        base,
        fmt,
        output=output,
        kind="plan",
        show_layers=show_layers,
        markdown_meta=_markdown_metadata(
            "kv-plan",
            show_layers,
            output,
            extras={
                "preset": preset,
                "preset_params": parsed_preset_params,
                "trust_remote_code": trust_remote_code,
            },
        ),
    )


@bench_group.command(name="kv-analyze")
@click.argument("model_id")
@click.option("--preset", type=click.Choice(_PRESET_CHOICES), default="balanced", show_default=True)
@click.option("--compare-preset", type=click.Choice(_PRESET_CHOICES))
@click.option("--prompt-tokens", type=int, default=1024, show_default=True)
@click.option("--show-layers", is_flag=True, help="Include per-layer quantized error rows in summary/markdown output")
@click.option("--sort-by", type=click.Choice(_KV_ANALYZE_SORT_CHOICES), default="layer", show_default=True, help="Sort quantized-layer rows in summary/markdown output")
@click.option("--preset-param", "preset_params", multiple=True, help="Override base preset params with key=value")
@click.option("--compare-preset-param", "compare_preset_params", multiple=True, help="Override compare preset params with key=value")
@click.option("--format", "fmt", type=click.Choice(["json", "summary", "markdown"]), default="json", show_default=True)
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=str))
def kv_analyze(
    model_id: str,
    preset: str,
    compare_preset: Optional[str],
    prompt_tokens: int,
    show_layers: bool,
    sort_by: str,
    preset_params: tuple[str, ...],
    compare_preset_params: tuple[str, ...],
    fmt: str,
    output: Optional[str],
) -> None:
    """Analyze per-layer KV quantization quality (MSE, cosine, residual gain)."""
    parsed_preset_params = _parse_preset_params(preset_params)
    parsed_compare_preset_params = _parse_preset_params(compare_preset_params)
    result = analyze_kv_quantization(
        model_id=model_id,
        prompt_tokens=prompt_tokens,
        preset=preset,
        compare_preset=compare_preset,
        preset_params=parsed_preset_params,
        compare_preset_params=parsed_compare_preset_params,
    )
    _emit_formatted_result(
        result,
        fmt,
        output,
        kind="kv-analyze",
        show_layers=show_layers,
        sort_by=sort_by,
        markdown_meta=_markdown_metadata(
            "kv-analyze",
            show_layers,
            output,
            extras={
                "preset": preset,
                "preset_params": parsed_preset_params,
                "compare_preset": compare_preset or "-",
                "compare_preset_params": parsed_compare_preset_params,
                "prompt_tokens": prompt_tokens,
                "sort_by": sort_by,
            },
        ),
    )


@bench_group.command(name="kv-smoke")
@click.argument("model_id")
@click.option("--preset", type=click.Choice(_PRESET_CHOICES), default="balanced", show_default=True)
@click.option("--compare-preset", type=click.Choice(_PRESET_CHOICES))
@click.option("--prompt-tokens", type=int, default=64, show_default=True)
@click.option("--max-new-tokens", type=int, default=8, show_default=True)
@click.option("--calib-new-tokens", type=int, default=8, show_default=True)
@click.option("--search-max-n", type=int, default=2, show_default=True)
@click.option("--preset-param", "preset_params", multiple=True, help="Override preset params with key=value")
@click.option("--compare-preset-param", "compare_preset_params", multiple=True, help="Override compare preset params with key=value")
@click.option("--format", "fmt", type=click.Choice(["json", "summary", "markdown"]), default="json", show_default=True)
@click.option("--show-layers", is_flag=True, help="Show per-layer policy decisions in summary mode")
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=str))
def kv_smoke(
    model_id: str,
    preset: str,
    compare_preset: Optional[str],
    prompt_tokens: int,
    max_new_tokens: int,
    calib_new_tokens: int,
    search_max_n: int,
    preset_params: tuple[str, ...],
    compare_preset_params: tuple[str, ...],
    fmt: str,
    show_layers: bool,
    output: Optional[str],
) -> None:
    """Run a quick smoke test (short prompt) to verify preset correctness."""
    parsed_preset_params = _parse_preset_params(preset_params)
    if compare_preset:
        parsed_compare_preset_params = _parse_preset_params(compare_preset_params)
        result = compare_smoke(
            model_id=model_id,
            preset=preset,
            compare_preset=compare_preset,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            calib_new_tokens=calib_new_tokens,
            search_max_n=search_max_n,
            preset_params=parsed_preset_params,
            compare_preset_params=parsed_compare_preset_params,
        )
        _emit_formatted_result(
            result,
            fmt,
            output,
            kind="smoke-compare",
            show_layers=show_layers,
            markdown_meta=_markdown_metadata(
                "kv-smoke",
                show_layers,
                output,
                extras={
                    "preset": preset,
                    "preset_params": parsed_preset_params,
                    "compare_preset": compare_preset,
                    "compare_preset_params": parsed_compare_preset_params,
                    "prompt_tokens": prompt_tokens,
                    "max_new_tokens": max_new_tokens,
                    "calib_new_tokens": calib_new_tokens,
                    "search_max_n": search_max_n,
                },
            ),
        )
        return
    result = run_smoke(
        model_id=model_id,
        preset=preset,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        calib_new_tokens=calib_new_tokens,
        search_max_n=search_max_n,
        preset_params=parsed_preset_params,
    )
    _emit_formatted_result(
        result,
        fmt,
        output,
        kind="smoke",
        show_layers=show_layers,
        markdown_meta=_markdown_metadata(
            "kv-smoke",
            show_layers,
            output,
            extras={
                "preset": preset,
                "preset_params": parsed_preset_params,
                "prompt_tokens": prompt_tokens,
                "max_new_tokens": max_new_tokens,
                "calib_new_tokens": calib_new_tokens,
                "search_max_n": search_max_n,
            },
        ),
    )


@bench_group.command(name="kv-long")
@click.argument("model_id")
@click.option("--preset", type=click.Choice(_PRESET_CHOICES), default="balanced", show_default=True)
@click.option("--compare-preset", type=click.Choice(_PRESET_CHOICES))
@click.option("--prompt-tokens", type=int, default=32768, show_default=True)
@click.option("--max-new-tokens", type=int, default=32, show_default=True)
@click.option("--calib-new-tokens", type=int, default=8, show_default=True)
@click.option("--search-max-n", type=int, default=8, show_default=True)
@click.option("--preset-param", "preset_params", multiple=True, help="Override preset params with key=value")
@click.option("--compare-preset-param", "compare_preset_params", multiple=True, help="Override compare preset params with key=value")
@click.option("--format", "fmt", type=click.Choice(["json", "summary", "markdown"]), default="json", show_default=True)
@click.option("--show-layers", is_flag=True, help="Show per-layer policy decisions in summary mode")
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=str))
def kv_long(
    model_id: str,
    preset: str,
    compare_preset: Optional[str],
    prompt_tokens: int,
    max_new_tokens: int,
    calib_new_tokens: int,
    search_max_n: int,
    preset_params: tuple[str, ...],
    compare_preset_params: tuple[str, ...],
    fmt: str,
    show_layers: bool,
    output: Optional[str],
) -> None:
    """Run long-context benchmark (32K+ tokens) to test KV cache efficiency."""
    parsed_preset_params = _parse_preset_params(preset_params)
    if compare_preset:
        parsed_compare_preset_params = _parse_preset_params(compare_preset_params)
        result = compare_long_context_preset(
            model_id=model_id,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            preset=preset,
            compare_preset=compare_preset,
            calib_new_tokens=calib_new_tokens,
            search_max_n=search_max_n,
            preset_params=parsed_preset_params,
            compare_preset_params=parsed_compare_preset_params,
        )
        _emit_formatted_result(
            result,
            fmt,
            output,
            kind="long-compare",
            show_layers=show_layers,
            markdown_meta=_markdown_metadata(
                "kv-long",
                show_layers,
                output,
                extras={
                    "preset": preset,
                    "preset_params": parsed_preset_params,
                    "compare_preset": compare_preset,
                    "compare_preset_params": parsed_compare_preset_params,
                    "prompt_tokens": prompt_tokens,
                    "max_new_tokens": max_new_tokens,
                    "calib_new_tokens": calib_new_tokens,
                    "search_max_n": search_max_n,
                },
            ),
        )
        return
    result = run_long_context_preset(
        model_id=model_id,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        preset=preset,
        calib_new_tokens=calib_new_tokens,
        search_max_n=search_max_n,
        preset_params=parsed_preset_params,
    )
    _emit_formatted_result(
        result,
        fmt,
        output,
        kind="long",
        show_layers=show_layers,
        markdown_meta=_markdown_metadata(
            "kv-long",
            show_layers,
            output,
            extras={
                "preset": preset,
                "preset_params": parsed_preset_params,
                "prompt_tokens": prompt_tokens,
                "max_new_tokens": max_new_tokens,
                "calib_new_tokens": calib_new_tokens,
                "search_max_n": search_max_n,
            },
        ),
    )


@bench_group.command(name="kv-suite")
@click.argument("model_id")
@click.option("--preset", type=click.Choice(_PRESET_CHOICES), default="balanced", show_default=True)
@click.option("--compare-preset", type=click.Choice(_PRESET_CHOICES))
@click.option("--prompt-tokens", multiple=True, type=int, default=(512, 2048), show_default=True)
@click.option("--max-new-tokens", type=int, default=32, show_default=True)
@click.option("--calib-new-tokens", type=int, default=8, show_default=True)
@click.option("--search-max-n", type=int, default=8, show_default=True)
@click.option("--preset-param", "preset_params", multiple=True, help="Override preset params with key=value")
@click.option("--compare-preset-param", "compare_preset_params", multiple=True, help="Override compare preset params with key=value")
@click.option("--format", "fmt", type=click.Choice(["json", "summary", "markdown"]), default="json", show_default=True)
@click.option("--show-layers", is_flag=True, help="Show per-layer policy decisions in summary mode")
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=str))
def kv_suite(
    model_id: str,
    preset: str,
    compare_preset: Optional[str],
    prompt_tokens: tuple[int, ...],
    max_new_tokens: int,
    calib_new_tokens: int,
    search_max_n: int,
    preset_params: tuple[str, ...],
    compare_preset_params: tuple[str, ...],
    fmt: str,
    show_layers: bool,
    output: Optional[str],
) -> None:
    """Run a suite of benchmarks across multiple prompt lengths."""
    parsed_preset_params = _parse_preset_params(preset_params)
    cfg = RunConfig(
        model_id=model_id,
        prompt_tokens=list(prompt_tokens),
        max_new_tokens=max_new_tokens,
        calib_new_tokens=calib_new_tokens,
        preset=preset,
        search_max_n=search_max_n,
        preset_params=parsed_preset_params,
    )
    if compare_preset:
        parsed_compare_preset_params = _parse_preset_params(compare_preset_params)
        result = compare_short_suite(cfg, compare_preset=compare_preset, compare_preset_params=parsed_compare_preset_params)
        _emit_formatted_result(
            result,
            fmt,
            output,
            kind="suite-compare",
            show_layers=show_layers,
            markdown_meta=_markdown_metadata(
                "kv-suite",
                show_layers,
                output,
                extras={
                    "preset": preset,
                    "preset_params": parsed_preset_params,
                    "compare_preset": compare_preset,
                    "compare_preset_params": parsed_compare_preset_params,
                    "prompt_tokens": list(prompt_tokens),
                    "max_new_tokens": max_new_tokens,
                    "calib_new_tokens": calib_new_tokens,
                    "search_max_n": search_max_n,
                },
            ),
        )
        return
    result = run_short_suite(cfg)
    _emit_formatted_result(
        result,
        fmt,
        output,
        kind="suite",
        show_layers=show_layers,
        markdown_meta=_markdown_metadata(
            "kv-suite",
            show_layers,
            output,
            extras={
                "preset": preset,
                "preset_params": parsed_preset_params,
                "prompt_tokens": list(prompt_tokens),
                "max_new_tokens": max_new_tokens,
                "calib_new_tokens": calib_new_tokens,
                "search_max_n": search_max_n,
            },
        ),
    )


@bench_group.command(name="kv-report")
@click.argument("model_id")
@click.option("--preset", type=click.Choice(_PRESET_CHOICES), default="balanced", show_default=True)
@click.option("--compare-preset", type=click.Choice(_PRESET_CHOICES), default="ultra-long", show_default=True)
@click.option("--smoke-prompt-tokens", type=int, default=64, show_default=True)
@click.option("--long-prompt-tokens", type=int, default=32768, show_default=True)
@click.option("--suite-prompt-tokens", multiple=True, type=int, default=(512, 2048), show_default=True)
@click.option("--max-new-tokens", type=int, default=32, show_default=True)
@click.option("--calib-new-tokens", type=int, default=8, show_default=True)
@click.option("--search-max-n", type=int, default=8, show_default=True)
@click.option("--preset-param", "preset_params", multiple=True, help="Override base preset params with key=value")
@click.option("--compare-preset-param", "compare_preset_params", multiple=True, help="Override compare preset params with key=value")
@click.option("--format", "fmt", type=click.Choice(["json", "summary", "markdown"]), default="json", show_default=True)
@click.option("--show-layers", is_flag=True, help="Show policy diff tables for each section in summary/markdown mode")
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=str))
@click.option("--output-dir", type=click.Path(file_okay=False, dir_okay=True, path_type=str))
def kv_report(
    model_id: str,
    preset: str,
    compare_preset: str,
    smoke_prompt_tokens: int,
    long_prompt_tokens: int,
    suite_prompt_tokens: tuple[int, ...],
    max_new_tokens: int,
    calib_new_tokens: int,
    search_max_n: int,
    preset_params: tuple[str, ...],
    compare_preset_params: tuple[str, ...],
    fmt: str,
    show_layers: bool,
    output: Optional[str],
    output_dir: Optional[str],
) -> None:
    """Run full benchmark report: smoke + long + suite in one command."""
    if output and output_dir:
        raise click.BadParameter("--output and --output-dir cannot be used together")
    parsed_preset_params = _parse_preset_params(preset_params)
    parsed_compare_preset_params = _parse_preset_params(compare_preset_params)
    suite_cfg = RunConfig(
        model_id=model_id,
        prompt_tokens=list(suite_prompt_tokens),
        max_new_tokens=max_new_tokens,
        calib_new_tokens=calib_new_tokens,
        preset=preset,
        search_max_n=search_max_n,
        preset_params=parsed_preset_params,
    )
    result = {
        "model_id": model_id,
        "base_preset": preset,
        "compare_preset": compare_preset,
        "smoke": compare_smoke(
            model_id=model_id,
            preset=preset,
            compare_preset=compare_preset,
            prompt_tokens=smoke_prompt_tokens,
            max_new_tokens=min(max_new_tokens, 8),
            calib_new_tokens=calib_new_tokens,
            search_max_n=max(2, search_max_n),
            preset_params=parsed_preset_params,
            compare_preset_params=parsed_compare_preset_params,
        ),
        "long": compare_long_context_preset(
            model_id=model_id,
            prompt_tokens=long_prompt_tokens,
            max_new_tokens=max_new_tokens,
            preset=preset,
            compare_preset=compare_preset,
            calib_new_tokens=calib_new_tokens,
            search_max_n=search_max_n,
            preset_params=parsed_preset_params,
            compare_preset_params=parsed_compare_preset_params,
        ),
        "suite": compare_short_suite(
            suite_cfg,
            compare_preset=compare_preset,
            compare_preset_params=parsed_compare_preset_params,
        ),
    }
    markdown_meta = _markdown_metadata(
        "kv-report",
        show_layers,
        output or output_dir,
        extras={
            "preset": preset,
            "preset_params": parsed_preset_params,
            "compare_preset": compare_preset,
            "compare_preset_params": parsed_compare_preset_params,
            "smoke_prompt_tokens": smoke_prompt_tokens,
            "long_prompt_tokens": long_prompt_tokens,
            "suite_prompt_tokens": list(suite_prompt_tokens),
            "max_new_tokens": max_new_tokens,
            "calib_new_tokens": calib_new_tokens,
            "search_max_n": search_max_n,
        },
    )
    if output_dir:
        markdown_text = _markdown_for_report(result, show_layers=show_layers, metadata=markdown_meta)
        _emit_dual_report_files(result, markdown_text, output_dir)
        return
    _emit_formatted_result(
        result,
        fmt,
        output,
        kind="report",
        show_layers=show_layers,
        markdown_meta=markdown_meta,
    )


# ============================================================================
# TurboQuantum CLI Commands
# ============================================================================

@bench_group.command(name="turboquantum")
@click.option("--mode", type=click.Choice(_TURBOQUANTUM_MODES), default="balanced", show_default=True)
@click.option("--heads", type=int, default=8, show_default=True)
@click.option("--seq-len", type=int, default=256, show_default=True)
@click.option("--head-dim", type=int, default=128, show_default=True)
@click.option("--target-bits", type=float, default=3.0, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--compare-modes", is_flag=True, help="Compare all 4 modes side-by-side")
@click.option("--format", "fmt", type=click.Choice(["json", "summary", "markdown"]), default="json", show_default=True)
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=str))
def turboquantum_cmd(
    mode: str,
    heads: int,
    seq_len: int,
    head_dim: int,
    target_bits: float,
    seed: int,
    compare_modes: bool,
    fmt: str,
    output: Optional[str],
) -> None:
    """Run TurboQuantum synthetic benchmark (no model needed)."""
    if not _HAS_TURBOQUANTUM:
        raise click.ClickException("TurboQuantum module not available. Install torch.")

    if compare_modes:
        result = compare_turboquantum_modes(
            num_heads=heads, seq_len=seq_len, head_dim=head_dim, seed=seed,
        )
        if fmt == "json":
            _emit_result(result, output)
            return
        if fmt == "markdown":
            lines = ["## TurboQuantum Mode Comparison", "", f"Shape: heads={heads}, seq_len={seq_len}, head_dim={head_dim}", ""]
            table = _turboquantum_mode_comparison_table(result.get("comparison_table", []))
            lines.extend([table, ""])
            if result.get("best_quality_mode"):
                lines.append(f"- **Best quality mode**: `{result['best_quality_mode']}`")
            if result.get("best_compression_mode"):
                lines.append(f"- **Best compression mode**: `{result['best_compression_mode']}`")
            _emit_text("\n".join(lines), output)
            return
        click.echo(f"TurboQuantum Mode Comparison (h={heads}, s={seq_len}, d={head_dim})")
        click.echo("")
        click.echo(_turboquantum_mode_comparison_table(result.get("comparison_table", [])))
        if result.get("best_quality_mode"):
            click.echo(f"\nBest quality mode: {result['best_quality_mode']}")
        return

    result = run_turboquantum_synthetic(
        num_heads=heads, seq_len=seq_len, head_dim=head_dim,
        mode=mode, target_avg_bits=target_bits, seed=seed,
    )

    if fmt == "json":
        _emit_result(result, output)
        return
    if fmt == "markdown":
        _emit_text(_markdown_for_turboquantum_synthetic(result), output)
        return
    # summary format
    click.echo(f"TurboQuantum Synthetic Benchmark (mode={mode})")
    click.echo("")
    for line in _turboquantum_summary_lines(result):
        click.echo(f"  {line}")


@bench_group.command(name="turboquantum-model")
@click.argument("model_id")
@click.option("--mode", type=click.Choice(_TURBOQUANTUM_MODES), default="balanced", show_default=True)
@click.option("--prompt-tokens", type=int, default=128, show_default=True)
@click.option("--format", "fmt", type=click.Choice(["json", "summary", "markdown"]), default="json", show_default=True)
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=str))
def turboquantum_model_cmd(
    model_id: str,
    mode: str,
    prompt_tokens: int,
    fmt: str,
    output: Optional[str],
) -> None:
    """Run TurboQuantum on a real model's KV cache."""
    if not _HAS_TURBOQUANTUM:
        raise click.ClickException("TurboQuantum module not available. Install torch.")
    result = run_turboquantum_on_model_kv(
        model_id=model_id, prompt_tokens=prompt_tokens, mode=mode,
    )
    if fmt == "json":
        _emit_result(result, output)
        return
    if fmt == "markdown":
        lines = [
            "## TurboQuantum Model KV Cache Benchmark",
            "",
            f"- `model`: {model_id}",
            f"- `mode`: {mode}",
            f"- `prompt_tokens`: {prompt_tokens}",
            f"- `total_layers`: {result.get('total_layers', 0)}",
            f"- `original_mb`: {_fmt_float(result.get('total_original_mb', 0), 2)}",
            f"- `compressed_mb`: {_fmt_float(result.get('total_compressed_mb', 0), 2)}",
            f"- `savings_pct`: {_fmt_float(result.get('overall_savings_pct', 0), 1)}%",
            "",
            "### Quality Averages",
            f"- K MSE: {_fmt_float((result.get('averages') or {}).get('k_mse', 0), 6)}",
            f"- V MSE: {_fmt_float((result.get('averages') or {}).get('v_mse', 0), 6)}",
        ]
        layer_details = result.get("layer_details", [])
        if layer_details:
            lines.extend([
                "",
                "### Per-Layer Details",
                "| layer | bpv | k_mse | v_mse | k_cos | v_cos | time_ms | orig_kb | comp_kb |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for layer in layer_details:
                lines.append(
                    f"| {layer['layer_idx']} | {_fmt_float(layer['bpv'])} | "
                    f"{_fmt_float(layer['k_mse'], 6)} | {_fmt_float(layer['v_mse'], 6)} | "
                    f"{_fmt_float(layer.get('k_cosine', 0), 4)} | {_fmt_float(layer.get('v_cosine', 0), 4)} | "
                    f"{layer['time_ms']} | {_fmt_float(layer['orig_kb'], 1)} | {_fmt_float(layer['comp_kb'], 1)} |"
                )
        _emit_text("\n".join(lines), output)
        return
    # summary
    click.echo(f"TurboQuantum on {model_id} (mode={mode})")
    click.echo(f"  layers: {result.get('total_layers', 0)}")
    click.echo(f"  original: {_fmt_float(result.get('total_original_mb', 0), 2)} MB")
    click.echo(f"  compressed: {_fmt_float(result.get('total_compressed_mb', 0), 2)} MB")
    click.echo(f"  savings: {_fmt_float(result.get('overall_savings_pct', 0), 1)}%")
    avg = result.get("averages") or {}
    click.echo(f"  k_mse: {_fmt_float(avg.get('k_mse', 0), 6)}")
    click.echo(f"  v_mse: {_fmt_float(avg.get('v_mse', 0), 6)}")
