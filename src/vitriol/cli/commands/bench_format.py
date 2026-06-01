"""Result formatting & report rendering helpers for the bench CLI."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import click


def _emit_result(result: Dict[str, Any], output: Optional[str]) -> None:
    """Emit a JSON result dict to stdout or write to file."""
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if output:
        Path(output).write_text(text)
        click.echo(output)
        return
    click.echo(text)


def _emit_text(text: str, output: Optional[str]) -> None:
    """Emit plain text to stdout or write to file."""
    if output:
        Path(output).write_text(text)
        click.echo(output)
        return
    click.echo(text)


def _emit_dual_report_files(result: Dict[str, Any], markdown_text: str, output_dir: str) -> None:
    """Write both JSON and Markdown reports to output_dir."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / "report.json"
    markdown_path = target_dir / "report.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    markdown_path.write_text(markdown_text)
    click.echo(str(json_path))
    click.echo(str(markdown_path))


def _stringify_metadata_value(value: Any) -> str:
    """Format metadata values for Markdown display."""
    if isinstance(value, dict):
        if not value:
            return "{}"
        parts = [f"{key}={value[key]}" for key in sorted(value)]
        return ", ".join(parts)
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) if value else "-"
    if value in (None, ""):
        return "-"
    return str(value)


def _markdown_metadata_block(metadata: Optional[Dict[str, Any]]) -> list[str]:
    """Render experiment metadata as Markdown list items."""
    if not metadata:
        return []
    lines = ["## Experiment Metadata", ""]
    for key, value in metadata.items():
        lines.append(f"- `{key}`: {_stringify_metadata_value(value)}")
    lines.append("")
    return lines


def _fmt_float(value: Any, digits: int = 3) -> str:
    """Format a numeric value with fixed decimal places."""
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _fmt_prefix_match(value: Any) -> str:
    """Format prefix match tuple (matched, total, pct) as human-readable string."""
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return f"{value[0]}/{value[1]} ({_fmt_float(value[2], 1)}%)"
    return "-"


def _memory_summary_lines(result: Dict[str, Any], prefix: str = "") -> list[str]:
    """Extract memory summary lines from benchmark result."""
    memory = result.get("tuned_memory") or {}
    if not memory:
        return []
    label_prefix = f"{prefix}_" if prefix else ""
    lines: list[str] = []
    estimated = memory.get("estimated_kv_megabytes")
    peak = memory.get("peak_device_megabytes")
    if estimated is not None:
        lines.append(f"{label_prefix}estimated_kv_mb: {_fmt_float(estimated)}")
    if peak is not None:
        lines.append(f"{label_prefix}peak_device_mb: {_fmt_float(peak)}")
    if estimated is not None and peak is not None:
        lines.append(f"{label_prefix}peak_minus_estimated_mb: {_fmt_float(float(peak) - float(estimated))}")
    return lines


def _memory_markdown_lines(result: Dict[str, Any], prefix: str = "") -> list[str]:
    """Convert memory summary lines to Markdown bullet format."""
    return [f"- `{line.split(': ', 1)[0]}`: {line.split(': ', 1)[1]}" for line in _memory_summary_lines(result, prefix=prefix)]


def _memory_compare_summary_lines(base: Dict[str, Any], compare: Dict[str, Any]) -> list[str]:
    """Generate side-by-side memory comparison summary lines."""
    base_memory = base.get("tuned_memory") or {}
    compare_memory = compare.get("tuned_memory") or {}
    if not base_memory and not compare_memory:
        return []
    lines: list[str] = []
    base_estimated = base_memory.get("estimated_kv_megabytes")
    compare_estimated = compare_memory.get("estimated_kv_megabytes")
    base_peak = base_memory.get("peak_device_megabytes")
    compare_peak = compare_memory.get("peak_device_megabytes")
    if base_estimated is not None:
        lines.append(f"base_estimated_kv_mb: {_fmt_float(base_estimated)}")
    if compare_estimated is not None:
        lines.append(f"compare_estimated_kv_mb: {_fmt_float(compare_estimated)}")
    if base_estimated is not None and compare_estimated is not None:
        lines.append(f"delta_estimated_kv_mb: {_fmt_float(compare_estimated - base_estimated)}")
    if base_peak is not None:
        lines.append(f"base_peak_device_mb: {_fmt_float(base_peak)}")
    if compare_peak is not None:
        lines.append(f"compare_peak_device_mb: {_fmt_float(compare_peak)}")
    if base_estimated is not None and base_peak is not None:
        lines.append(f"base_peak_minus_estimated_mb: {_fmt_float(float(base_peak) - float(base_estimated))}")
    if compare_estimated is not None and compare_peak is not None:
        lines.append(f"compare_peak_minus_estimated_mb: {_fmt_float(float(compare_peak) - float(compare_estimated))}")
    if base_peak is not None and compare_peak is not None:
        lines.append(f"delta_peak_device_mb: {_fmt_float(compare_peak - base_peak)}")
    return lines


def _memory_compare_markdown_lines(base: Dict[str, Any], compare: Dict[str, Any]) -> list[str]:
    """Convert memory comparison lines to Markdown bullet format."""
    return [f"- `{line.split(': ', 1)[0]}`: {line.split(': ', 1)[1]}" for line in _memory_compare_summary_lines(base, compare)]


def _turboquant_summary_lines(result: Dict[str, Any], prefix: str = "") -> list[str]:
    """Extract TurboQuantum statistics summary lines from benchmark result."""
    stats = result.get("tuned_turboquant") or {}
    if not stats:
        return []
    label_prefix = f"{prefix}_" if prefix else ""
    lines = [f"{label_prefix}turboquant_calls: {int(stats.get('calls', 0) or 0)}"]
    for key in [
        "avg_residual_l2",
        "avg_correction_l2",
        "correction_to_residual_l2_ratio",
        "avg_residual_abs_mean",
        "avg_correction_abs_mean",
    ]:
        if key in stats:
            lines.append(f"{label_prefix}{key}: {_fmt_float(stats.get(key, 0.0))}")
    return lines


def _turboquant_markdown_lines(result: Dict[str, Any], prefix: str = "") -> list[str]:
    """Convert TurboQuantum summary lines to Markdown bullet format."""
    return [f"- `{line.split(': ', 1)[0]}`: {line.split(': ', 1)[1]}" for line in _turboquant_summary_lines(result, prefix=prefix)]


def _turboquant_compare_summary_lines(base: Dict[str, Any], compare: Dict[str, Any]) -> list[str]:
    """Generate side-by-side TurboQuantum comparison summary lines."""
    return _turboquant_summary_lines(base, prefix="base") + _turboquant_summary_lines(compare, prefix="compare")


def _turboquant_compare_markdown_lines(base: Dict[str, Any], compare: Dict[str, Any]) -> list[str]:
    """Convert TurboQuantum comparison lines to Markdown bullet format."""
    return [f"- `{line.split(': ', 1)[0]}`: {line.split(': ', 1)[1]}" for line in _turboquant_compare_summary_lines(base, compare)]


def _render_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Render a simple text table with auto-sized columns."""
    widths = [len(h) for h in headers]
    rendered_rows: list[list[str]] = []
    for row in rows:
        rendered = [str(cell) for cell in row]
        rendered_rows.append(rendered)
        for idx, cell in enumerate(rendered):
            widths[idx] = max(widths[idx], len(cell))

    def render_row(row: list[str]) -> str:
        return "  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    sep = "  ".join("-" * width for width in widths)
    lines = [render_row(headers), sep]
    lines.extend(render_row(row) for row in rendered_rows)
    return "\n".join(lines)


def _policy_summary_lines(result: Dict[str, Any]) -> list[str]:
    """Extract KV policy summary lines from benchmark result."""
    insights = result.get("policy_insights") or {}
    counts = insights.get("counts") or {}
    if not counts:
        return []
    return [
        f"quantized_kv_start: {insights.get('quantized_kv_start', '-')}",
        "policy_counts: "
        f"full={counts.get('full_attention', 0)}, "
        f"sliding={counts.get('sliding_window', 0)}, "
        f"mla={counts.get('mla', 0)}, "
        f"linear={counts.get('linear_attention', 0)}, "
        f"turbo_k={counts.get('turbo_k', 0)}, "
        f"turbo_v={counts.get('turbo_v', 0)}, "
        f"sparse_v={counts.get('sparse_v', 0)}, "
        f"compute_skip={counts.get('compute_skip', 0)}",
    ]


def _policy_layer_table(result: Dict[str, Any]) -> str:
    """Render a text table of per-layer policy decisions."""
    insights = result.get("policy_insights") or {}
    layers = insights.get("layers") or []
    if not layers:
        return ""
    return _render_table(
        ["layer", "type", "turbo_k", "turbo_v", "sparse_v", "compute_skip"],
        [
            [
                layer.get("layer_idx", "-"),
                layer.get("layer_type", "-"),
                "Y" if layer.get("turbo_quantize_k") else "-",
                "Y" if layer.get("turbo_quantize_v") else "-",
                "Y" if layer.get("enable_sparse_v") else "-",
                "Y" if layer.get("enable_compute_skip") else "-",
            ]
            for layer in layers
        ],
    )


def _markdown_policy_lines(result: Dict[str, Any]) -> list[str]:
    """Convert policy summary to Markdown bullet format."""
    insights = result.get("policy_insights") or {}
    counts = insights.get("counts") or {}
    if not counts:
        return []
    return [
        f"- `quantized_kv_start`: {insights.get('quantized_kv_start', '-')}",
        "- `policy_counts`: "
        f"full={counts.get('full_attention', 0)}, "
        f"sliding={counts.get('sliding_window', 0)}, "
        f"mla={counts.get('mla', 0)}, "
        f"linear={counts.get('linear_attention', 0)}, "
        f"turbo_k={counts.get('turbo_k', 0)}, "
        f"turbo_v={counts.get('turbo_v', 0)}, "
        f"sparse_v={counts.get('sparse_v', 0)}, "
        f"compute_skip={counts.get('compute_skip', 0)}",
    ]


def _markdown_layer_table(result: Dict[str, Any]) -> str:
    """Render per-layer policy decisions as a Markdown table."""
    insights = result.get("policy_insights") or {}
    layers = insights.get("layers") or []
    if not layers:
        return ""
    lines = [
        "| layer | type | turbo_k | turbo_v | sparse_v | compute_skip |",
        "|---:|---|:---:|:---:|:---:|:---:|",
    ]
    for layer in layers:
        lines.append(
            f"| {layer.get('layer_idx', '-')} | {layer.get('layer_type', '-')} | "
            f"{'Y' if layer.get('turbo_quantize_k') else '-'} | "
            f"{'Y' if layer.get('turbo_quantize_v') else '-'} | "
            f"{'Y' if layer.get('enable_sparse_v') else '-'} | "
            f"{'Y' if layer.get('enable_compute_skip') else '-'} |"
        )
    return "\n".join(lines)


def _markdown_suite_table(result: Dict[str, Any]) -> str:
    """Render benchmark suite results as a Markdown table."""
    rows = result.get("results", []) or []
    if not rows:
        return ""
    lines = [
        "| case | speedup | exact | prefix_match | base tok/s | tuned tok/s |",
        "|---|---:|:---:|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('name', '-')} | {_fmt_float(row.get('speedup', 0.0))}x | "
            f"{row.get('exact', '-')} | {_fmt_prefix_match(row.get('prefix_match'))} | "
            f"{_fmt_float(row.get('base_toks_per_s', 0.0), 2)} | {_fmt_float(row.get('tuned_toks_per_s', 0.0), 2)} |"
        )
    return "\n".join(lines)


def _markdown_diff_table(result: Dict[str, Any]) -> str:
    """Render policy plan diff as a Markdown table."""
    changed_layers = result.get("changed_layers", []) or []
    if not changed_layers:
        return ""
    lines = [
        "| layer | field | base | compare |",
        "|---:|---|---|---|",
    ]
    for item in changed_layers:
        for field, values in item.get("changes", {}).items():
            lines.append(f"| {item['layer_idx']} | {field} | {values['base']} | {values['compare']} |")
    return "\n".join(lines)


def _suite_compare_rows(result: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Extract case-level comparison rows from suite compare result."""
    return list(result.get("case_diffs") or [])


# Valid sort keys for KV analyze layer table
_KV_ANALYZE_SORT_CHOICES = ["layer", "key_mse_delta", "logits_mse_delta", "residual_gain_k"]


def _sorted_kv_analyze_rows(
    rows: list[Dict[str, Any]],
    compare_rows: Optional[list[Dict[str, Any]]] = None,
    sort_by: str = "layer",
) -> list[Dict[str, Any]]:
    """Sort KV analyze layer rows by the specified criterion.

    Args:
        rows: Base layer analysis rows.
        compare_rows: Optional comparison rows for delta sorting.
        sort_by: One of 'layer', 'key_mse_delta', 'logits_mse_delta', 'residual_gain_k'.
    """
    if sort_by == "layer" or not rows:
        return sorted(rows, key=lambda row: int(row.get("layer_idx", -1)))
    compare_by_layer = {int(row.get("layer_idx", -1)): row for row in (compare_rows or [])}

    def score(row: Dict[str, Any]) -> float:
        layer_idx = int(row.get("layer_idx", -1))
        compare_row = compare_by_layer.get(layer_idx, {}) or {}
        if sort_by == "key_mse_delta":
            return float(compare_row.get("key_mse", 0.0)) - float(row.get("key_mse", 0.0))
        if sort_by == "logits_mse_delta":
            return float(compare_row.get("logits_mse", 0.0)) - float(row.get("logits_mse", 0.0))
        if sort_by == "residual_gain_k":
            return float(row.get("residual_gain_k", 0.0))
        return float(layer_idx)

    return sorted(rows, key=score, reverse=True)


def _kv_analyze_summary(result: Dict[str, Any]) -> str:
    """Generate plain-text summary of KV quantization analysis."""
    base = result.get("base", {}) or {}
    base_summary = base.get("summary", {}) or {}
    lines = [
        f"model: {result.get('model_id', '-')}",
        f"prompt_tokens: {result.get('prompt_tokens', '-')}",
        f"base_preset: {base.get('preset', {}).get('name', '-')}",
        f"base_quantized_layers: {base_summary.get('quantized_layers', 0)}",
        f"base_avg_key_mse: {_fmt_float(base_summary.get('avg_key_mse', 0.0), 6)}",
        f"base_avg_value_mse: {_fmt_float(base_summary.get('avg_value_mse', 0.0), 6)}",
        f"base_avg_logits_mse: {_fmt_float(base_summary.get('avg_logits_mse', 0.0), 6)}",
        f"base_avg_output_mse: {_fmt_float(base_summary.get('avg_output_mse', 0.0), 6)}",
        f"base_avg_residual_gain_k: {_fmt_float(base_summary.get('avg_residual_gain_k', 0.0), 6)}",
        f"base_avg_residual_gain_v: {_fmt_float(base_summary.get('avg_residual_gain_v', 0.0), 6)}",
    ]
    compare = result.get("compare")
    if compare:
        compare_summary = compare.get("summary", {}) or {}
        lines.extend(
            [
                "",
                f"compare_preset: {compare.get('preset', {}).get('name', '-')}",
                f"compare_quantized_layers: {compare_summary.get('quantized_layers', 0)}",
                f"compare_avg_key_mse: {_fmt_float(compare_summary.get('avg_key_mse', 0.0), 6)}",
                f"compare_avg_value_mse: {_fmt_float(compare_summary.get('avg_value_mse', 0.0), 6)}",
                f"compare_avg_logits_mse: {_fmt_float(compare_summary.get('avg_logits_mse', 0.0), 6)}",
                f"compare_avg_output_mse: {_fmt_float(compare_summary.get('avg_output_mse', 0.0), 6)}",
                f"compare_avg_residual_gain_k: {_fmt_float(compare_summary.get('avg_residual_gain_k', 0.0), 6)}",
                f"compare_avg_residual_gain_v: {_fmt_float(compare_summary.get('avg_residual_gain_v', 0.0), 6)}",
            ]
        )
    base_layers = [row for row in (base.get("layers") or []) if row.get("turbo_quantize_k") or row.get("turbo_quantize_v")]
    compare_layers = [row for row in (compare.get("layers") or []) if row.get("turbo_quantize_k") or row.get("turbo_quantize_v")] if compare else []
    result["_base_quantized_layers"] = base_layers
    result["_compare_quantized_layers"] = compare_layers
    return "\n".join(lines)


def _kv_analyze_layer_table(
    rows: list[Dict[str, Any]],
    compare_rows: Optional[list[Dict[str, Any]]] = None,
    sort_by: str = "layer",
) -> str:
    """Render KV analyze per-layer metrics as a text table.

    Args:
        rows: Base layer analysis rows.
        compare_rows: Optional comparison rows for side-by-side diff.
        sort_by: Sort criterion for rows.
    """
    rows = _sorted_kv_analyze_rows(rows, compare_rows, sort_by=sort_by)
    if compare_rows is None:
        return _render_table(
            ["layer", "type", "K", "V", "key mse", "logits mse", "output mse", "res gain K"],
            [
                [
                    row.get("layer_idx", "-"),
                    row.get("layer_type", "-"),
                    "Y" if row.get("turbo_quantize_k") else "-",
                    "Y" if row.get("turbo_quantize_v") else "-",
                    _fmt_float(row.get("key_mse", 0.0), 6),
                    _fmt_float(row.get("logits_mse", 0.0), 6),
                    _fmt_float(row.get("output_mse", 0.0), 6),
                    _fmt_float(row.get("residual_gain_k", 0.0), 6),
                ]
                for row in rows
            ],
        )
    compare_by_layer = {int(row.get("layer_idx", -1)): row for row in (compare_rows or [])}
    return _render_table(
        ["layer", "type", "base key mse", "cmp key mse", "delta", "base logits mse", "cmp logits mse", "delta", "base gain K", "cmp gain K"],
        [
            [
                row.get("layer_idx", "-"),
                row.get("layer_type", "-"),
                _fmt_float(row.get("key_mse", 0.0), 6),
                _fmt_float((compare_by_layer.get(int(row.get("layer_idx", -1)), {}) or {}).get("key_mse", 0.0), 6),
                _fmt_float(
                    float((compare_by_layer.get(int(row.get("layer_idx", -1)), {}) or {}).get("key_mse", 0.0))
                    - float(row.get("key_mse", 0.0)),
                    6,
                ),
                _fmt_float(row.get("logits_mse", 0.0), 6),
                _fmt_float((compare_by_layer.get(int(row.get("layer_idx", -1)), {}) or {}).get("logits_mse", 0.0), 6),
                _fmt_float(
                    float((compare_by_layer.get(int(row.get("layer_idx", -1)), {}) or {}).get("logits_mse", 0.0))
                    - float(row.get("logits_mse", 0.0)),
                    6,
                ),
                _fmt_float(row.get("residual_gain_k", 0.0), 6),
                _fmt_float((compare_by_layer.get(int(row.get("layer_idx", -1)), {}) or {}).get("residual_gain_k", 0.0), 6),
            ]
            for row in rows
        ],
    )


def _markdown_kv_analyze_layer_table(
    rows: list[Dict[str, Any]],
    compare_rows: Optional[list[Dict[str, Any]]] = None,
    sort_by: str = "layer",
) -> str:
    """Render KV analyze per-layer metrics as a Markdown table."""
    rows = _sorted_kv_analyze_rows(rows, compare_rows, sort_by=sort_by)
    if compare_rows is None:
        lines = [
            "| layer | type | K | V | key mse | logits mse | output mse | res gain K |",
            "|---:|---|:---:|:---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| {row.get('layer_idx', '-')} | {row.get('layer_type', '-')} | "
                f"{'Y' if row.get('turbo_quantize_k') else '-'} | "
                f"{'Y' if row.get('turbo_quantize_v') else '-'} | "
                f"{_fmt_float(row.get('key_mse', 0.0), 6)} | "
                f"{_fmt_float(row.get('logits_mse', 0.0), 6)} | "
                f"{_fmt_float(row.get('output_mse', 0.0), 6)} | "
                f"{_fmt_float(row.get('residual_gain_k', 0.0), 6)} |"
            )
        return "\n".join(lines)
    compare_by_layer = {int(row.get("layer_idx", -1)): row for row in (compare_rows or [])}
    lines = [
        "| layer | type | base key mse | cmp key mse | delta key mse | base logits mse | cmp logits mse | delta logits mse | base gain K | cmp gain K |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        compare_row = compare_by_layer.get(int(row.get("layer_idx", -1)), {}) or {}
        lines.append(
            f"| {row.get('layer_idx', '-')} | {row.get('layer_type', '-')} | "
            f"{_fmt_float(row.get('key_mse', 0.0), 6)} | "
            f"{_fmt_float(compare_row.get('key_mse', 0.0), 6)} | "
            f"{_fmt_float(float(compare_row.get('key_mse', 0.0)) - float(row.get('key_mse', 0.0)), 6)} | "
            f"{_fmt_float(row.get('logits_mse', 0.0), 6)} | "
            f"{_fmt_float(compare_row.get('logits_mse', 0.0), 6)} | "
            f"{_fmt_float(float(compare_row.get('logits_mse', 0.0)) - float(row.get('logits_mse', 0.0)), 6)} | "
            f"{_fmt_float(row.get('residual_gain_k', 0.0), 6)} | "
            f"{_fmt_float(compare_row.get('residual_gain_k', 0.0), 6)} |"
        )
    return "\n".join(lines)


def _kv_analyze_summary_with_layers(result: Dict[str, Any], show_layers: bool = False, sort_by: str = "layer") -> str:
    """Generate KV analyze summary with optional per-layer detail table."""
    text = _kv_analyze_summary(result)
    if not show_layers:
        return text
    base = result.get("base", {}) or {}
    compare = result.get("compare", {}) or {}
    base_layers = result.get("_base_quantized_layers")
    if base_layers is None:
        base_layers = [row for row in (base.get("layers") or []) if row.get("turbo_quantize_k") or row.get("turbo_quantize_v")]
    compare_layers = result.get("_compare_quantized_layers")
    if compare and compare_layers is None:
        compare_layers = [row for row in (compare.get("layers") or []) if row.get("turbo_quantize_k") or row.get("turbo_quantize_v")]
    if not base_layers:
        return text
    lines = [text, "", f"quantized layers (sorted by {sort_by}):"]
    lines.append(_kv_analyze_layer_table(base_layers, compare_layers if compare else None, sort_by=sort_by))
    return "\n".join(lines)


def _kv_analyze_markdown(
    result: Dict[str, Any],
    show_layers: bool = False,
    sort_by: str = "layer",
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Render full KV analyze report in Markdown format."""
    base = result.get("base", {}) or {}
    compare = result.get("compare", {}) or {}
    lines = _markdown_metadata_block(metadata)
    lines.extend(
        [
            "## KV Analyze",
            "",
            f"- `model`: {result.get('model_id', '-')}",
            f"- `prompt_tokens`: {result.get('prompt_tokens', '-')}",
            f"- `base_preset`: {base.get('preset', {}).get('name', '-')}",
        ]
    )
    for key, value in (base.get("summary", {}) or {}).items():
        lines.append(f"- `base_{key}`: {_fmt_float(value, 6) if isinstance(value, (int, float)) else value}")
    if compare:
        lines.extend(["", f"- `compare_preset`: {compare.get('preset', {}).get('name', '-')}"])
        for key, value in (compare.get("summary", {}) or {}).items():
            lines.append(f"- `compare_{key}`: {_fmt_float(value, 6) if isinstance(value, (int, float)) else value}")
    if show_layers:
        base_layers = [row for row in (base.get("layers") or []) if row.get("turbo_quantize_k") or row.get("turbo_quantize_v")]
        compare_layers = [row for row in (compare.get("layers") or []) if row.get("turbo_quantize_k") or row.get("turbo_quantize_v")] if compare else []
        if base_layers:
            lines.extend(
                [
                    "",
                    f"### Quantized Layers (sorted by `{sort_by}`)",
                    "",
                    _markdown_kv_analyze_layer_table(base_layers, compare_layers if compare else None, sort_by=sort_by),
                ]
            )
    return "\n".join(lines)


def _markdown_suite_compare_table(result: Dict[str, Any]) -> str:
    """Render suite comparison results as a Markdown table."""
    rows = _suite_compare_rows(result)
    if not rows:
        return ""
    lines = [
        "| case | base speedup | compare speedup | delta | base exact | compare exact |",
        "|---|---:|---:|---:|:---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('name', '-')} | "
            f"{_fmt_float(row.get('base_speedup', 0.0))}x | "
            f"{_fmt_float(row.get('compare_speedup', 0.0))}x | "
            f"{_fmt_float(row.get('delta_speedup', 0.0), 3)}x | "
            f"{row.get('base_exact', '-')} | {row.get('compare_exact', '-')} |"
        )
    return "\n".join(lines)


def _markdown_for_plan(result: Dict[str, Any], show_layers: bool = False, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Render KV policy plan as Markdown."""
    lines = _markdown_metadata_block(metadata)
    lines.extend([
        "## KV Plan",
        "",
        f"- `model`: {result.get('model_id', '-')}",
        f"- `preset`: {result.get('preset', {}).get('name', result.get('preset', '-'))}",
        f"- `chosen_v_quant_layers`: {result.get('chosen_v_quantize_only_first_n', '-')}",
    ])
    lines.extend(_markdown_policy_lines(result))
    if show_layers:
        table = _markdown_layer_table(result)
        if table:
            lines.extend(["", table])
    return "\n".join(lines)


def _markdown_for_plan_diff(result: Dict[str, Any], show_layers: bool = False, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Render KV policy plan diff report in Markdown."""
    base = result.get("base", {})
    compare = result.get("compare", {})
    lines = _markdown_metadata_block(metadata)
    lines.extend([
        "## KV Plan Diff",
        "",
        f"- `model`: {result.get('model_id', '-')}",
        f"- `base_preset`: {base.get('preset', {}).get('name', '-')}",
        f"- `compare_preset`: {compare.get('preset', {}).get('name', '-')}",
        f"- `changed_layers`: {len(result.get('changed_layers', []) or [])}",
        "",
        "### Base",
        _markdown_for_plan(base, show_layers=False),
        "",
        "### Compare",
        _markdown_for_plan(compare, show_layers=False),
    ])
    diff_table = _markdown_diff_table(result)
    if diff_table:
        lines.extend(["", "### Changes", "", diff_table])
    if show_layers:
        base_table = _markdown_layer_table(base)
        compare_table = _markdown_layer_table(compare)
        if base_table:
            lines.extend(["", "### Base Layers", "", base_table])
        if compare_table:
            lines.extend(["", "### Compare Layers", "", compare_table])
    return "\n".join(lines)


def _markdown_for_smoke(result: Dict[str, Any], show_layers: bool = False, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Render single smoke test result in Markdown."""
    exact = result.get("exact", result.get("tuned_exact", "-"))
    speedup = result.get("speedup", result.get("tuned_speedup", 0.0))
    prefix_match = result.get("prefix_match", result.get("tuned_prefix_match"))
    lines = _markdown_metadata_block(metadata)
    lines.extend([
        "## KV Smoke",
        "",
        f"- `model`: {result.get('model_id', '-')}",
        f"- `preset`: {result.get('preset', {}).get('name', result.get('preset', '-'))}",
        f"- `ok`: {result.get('ok', '-')}",
        f"- `exact`: {exact}",
        f"- `speedup`: {_fmt_float(speedup)}x",
        f"- `prefix_match`: {_fmt_prefix_match(prefix_match)}",
        f"- `chosen_v_quant_layers`: {result.get('chosen_v_quantize_only_first_n', '-')}",
    ])
    lines.extend(_memory_markdown_lines(result))
    lines.extend(_turboquant_markdown_lines(result))
    lines.extend(_markdown_policy_lines(result))
    if show_layers:
        table = _markdown_layer_table(result)
        if table:
            lines.extend(["", table])
    return "\n".join(lines)


def _markdown_for_smoke_compare(result: Dict[str, Any], show_layers: bool = False, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Render smoke test comparison in Markdown."""
    base = result.get("base", {})
    compare = result.get("compare", {})
    policy_diff = result.get("policy_diff") or {}
    lines = _markdown_metadata_block(metadata)
    lines.extend(
        [
            "## KV Smoke Compare",
            "",
            f"- `model`: {result.get('model_id', '-')}",
            f"- `prompt_tokens`: {result.get('prompt_tokens', '-')}",
            f"- `base_preset`: {base.get('preset', {}).get('name', '-')}",
            f"- `compare_preset`: {compare.get('preset', {}).get('name', '-')}",
            f"- `base_ok`: {base.get('ok', '-')}",
            f"- `compare_ok`: {compare.get('ok', '-')}",
            f"- `base_exact`: {base.get('exact', base.get('tuned_exact', '-'))}",
            f"- `compare_exact`: {compare.get('exact', compare.get('tuned_exact', '-'))}",
            f"- `base_speedup`: {_fmt_float(base.get('speedup', base.get('tuned_speedup', 0.0)))}x",
            f"- `compare_speedup`: {_fmt_float(compare.get('speedup', compare.get('tuned_speedup', 0.0)))}x",
            f"- `delta_speedup`: {_fmt_float(result.get('delta_speedup', 0.0))}x",
            f"- `changed_layers`: {len(policy_diff.get('changed_layers', []) or [])}",
        ]
    )
    lines.extend(_memory_compare_markdown_lines(base, compare))
    lines.extend(_turboquant_compare_markdown_lines(base, compare))
    diff_table = _markdown_diff_table(policy_diff)
    if diff_table:
        lines.extend(["", "### Policy Changes", "", diff_table])
    if show_layers:
        base_table = _markdown_layer_table(base)
        compare_table = _markdown_layer_table(compare)
        if base_table:
            lines.extend(["", "### Base Layers", "", base_table])
        if compare_table:
            lines.extend(["", "### Compare Layers", "", compare_table])
    return "\n".join(lines)


def _markdown_for_long(result: Dict[str, Any], show_layers: bool = False, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Render long-context benchmark result in Markdown."""
    lines = _markdown_metadata_block(metadata)
    lines.extend([
        "## KV Long",
        "",
        f"- `model`: {result.get('model_id', '-')}",
        f"- `preset`: {result.get('preset', {}).get('name', result.get('preset', '-'))}",
        f"- `prompt_tokens`: {result.get('prompt_tokens', '-')}",
        f"- `tuned_exact`: {result.get('tuned_exact', '-')}",
        f"- `tuned_speedup`: {_fmt_float(result.get('tuned_speedup', 0.0))}x",
        f"- `prefix_match`: {_fmt_prefix_match(result.get('tuned_prefix_match'))}",
        f"- `chosen_v_quant_layers`: {result.get('chosen_v_quantize_only_first_n', '-')}",
    ])
    lines.extend(_memory_markdown_lines(result))
    lines.extend(_turboquant_markdown_lines(result))
    lines.extend(_markdown_policy_lines(result))
    if show_layers:
        table = _markdown_layer_table(result)
        if table:
            lines.extend(["", table])
    return "\n".join(lines)


def _markdown_for_long_compare(result: Dict[str, Any], show_layers: bool = False, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Render long-context benchmark comparison in Markdown."""
    base = result.get("base", {})
    compare = result.get("compare", {})
    policy_diff = result.get("policy_diff") or {}
    lines = _markdown_metadata_block(metadata)
    lines.extend(
        [
            "## KV Long Compare",
            "",
            f"- `model`: {result.get('model_id', '-')}",
            f"- `prompt_tokens`: {result.get('prompt_tokens', '-')}",
            f"- `base_preset`: {base.get('preset', {}).get('name', '-')}",
            f"- `compare_preset`: {compare.get('preset', {}).get('name', '-')}",
            f"- `base_exact`: {base.get('tuned_exact', '-')}",
            f"- `compare_exact`: {compare.get('tuned_exact', '-')}",
            f"- `base_speedup`: {_fmt_float(base.get('tuned_speedup', 0.0))}x",
            f"- `compare_speedup`: {_fmt_float(compare.get('tuned_speedup', 0.0))}x",
            f"- `delta_speedup`: {_fmt_float(result.get('delta_speedup', 0.0))}x",
            f"- `changed_layers`: {len(policy_diff.get('changed_layers', []) or [])}",
        ]
    )
    lines.extend(_memory_compare_markdown_lines(base, compare))
    lines.extend(_turboquant_compare_markdown_lines(base, compare))
    diff_table = _markdown_diff_table(policy_diff)
    if diff_table:
        lines.extend(["", "### Policy Changes", "", diff_table])
    if show_layers:
        base_table = _markdown_layer_table(base)
        compare_table = _markdown_layer_table(compare)
        if base_table:
            lines.extend(["", "### Base Layers", "", base_table])
        if compare_table:
            lines.extend(["", "### Compare Layers", "", compare_table])
    return "\n".join(lines)


def _markdown_for_suite(result: Dict[str, Any], show_layers: bool = False, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Render benchmark suite result in Markdown."""
    lines = _markdown_metadata_block(metadata)
    lines.extend([
        "## KV Suite",
        "",
        f"- `model`: {result.get('model_id', '-')}",
        f"- `preset`: {result.get('preset', {}).get('name', result.get('preset', '-'))}",
        f"- `all_cases_exact_match`: {result.get('all_cases_exact_match', '-')}",
        f"- `chosen_v_quant_layers`: {result.get('chosen_v_quantize_only_first_n', '-')}",
    ])
    lines.extend(_markdown_policy_lines(result))
    suite_table = _markdown_suite_table(result)
    if suite_table:
        lines.extend(["", suite_table])
    if show_layers:
        table = _markdown_layer_table(result)
        if table:
            lines.extend(["", table])
    return "\n".join(lines)


def _markdown_for_suite_compare(result: Dict[str, Any], show_layers: bool = False, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Render benchmark suite comparison in Markdown."""
    base = result.get("base", {})
    compare = result.get("compare", {})
    policy_diff = result.get("policy_diff") or {}
    lines = _markdown_metadata_block(metadata)
    lines.extend(
        [
            "## KV Suite Compare",
            "",
            f"- `model`: {result.get('model_id', '-')}",
            f"- `base_preset`: {base.get('preset', {}).get('name', '-')}",
            f"- `compare_preset`: {compare.get('preset', {}).get('name', '-')}",
            f"- `changed_layers`: {len(policy_diff.get('changed_layers', []) or [])}",
        ]
    )
    table = _markdown_suite_compare_table(result)
    if table:
        lines.extend(["", table])
    diff_table = _markdown_diff_table(policy_diff)
    if diff_table:
        lines.extend(["", "### Policy Changes", "", diff_table])
    if show_layers:
        base_table = _markdown_layer_table(base)
        compare_table = _markdown_layer_table(compare)
        if base_table:
            lines.extend(["", "### Base Layers", "", base_table])
        if compare_table:
            lines.extend(["", "### Compare Layers", "", compare_table])
    return "\n".join(lines)


def _report_sections(result: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Extract smoke, long, and suite sections from a combined report result."""
    return result.get("smoke", {}) or {}, result.get("long", {}) or {}, result.get("suite", {}) or {}


def _markdown_for_report(result: Dict[str, Any], show_layers: bool = False, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Render combined benchmark report (smoke + long + suite) in Markdown."""
    smoke, long_result, suite = _report_sections(result)
    lines = _markdown_metadata_block(metadata)
    lines.extend(
        [
            "## KV Report",
            "",
            f"- `model`: {result.get('model_id', '-')}",
            f"- `base_preset`: {result.get('base_preset', '-')}",
            f"- `compare_preset`: {result.get('compare_preset', '-')}",
        ]
    )
    if smoke:
        lines.extend(
            [
                "",
                "### Smoke",
                "",
                f"- `delta_speedup`: {_fmt_float(smoke.get('delta_speedup', 0.0))}x",
                f"- `base_exact`: {smoke.get('base', {}).get('exact', smoke.get('base', {}).get('tuned_exact', '-'))}",
                f"- `compare_exact`: {smoke.get('compare', {}).get('exact', smoke.get('compare', {}).get('tuned_exact', '-'))}",
            ]
        )
        lines.extend(_memory_compare_markdown_lines(smoke.get("base", {}) or {}, smoke.get("compare", {}) or {}))
        lines.extend(_turboquant_compare_markdown_lines(smoke.get("base", {}) or {}, smoke.get("compare", {}) or {}))
    if long_result:
        lines.extend(
            [
                "",
                "### Long",
                "",
                f"- `delta_speedup`: {_fmt_float(long_result.get('delta_speedup', 0.0))}x",
                f"- `base_exact`: {long_result.get('base', {}).get('tuned_exact', '-')}",
                f"- `compare_exact`: {long_result.get('compare', {}).get('tuned_exact', '-')}",
            ]
        )
        lines.extend(_memory_compare_markdown_lines(long_result.get("base", {}) or {}, long_result.get("compare", {}) or {}))
        lines.extend(_turboquant_compare_markdown_lines(long_result.get("base", {}) or {}, long_result.get("compare", {}) or {}))
    if suite:
        lines.extend(
            [
                "",
                "### Suite",
                "",
                f"- `cases`: {len(suite.get('case_diffs', []) or [])}",
                f"- `changed_layers`: {len((suite.get('policy_diff') or {}).get('changed_layers', []) or [])}",
            ]
        )
        table = _markdown_suite_compare_table(suite)
        if table:
            lines.extend(["", table])
    if show_layers:
        for section_name, section in [("Smoke", smoke), ("Long", long_result), ("Suite", suite)]:
            diff_table = _markdown_diff_table(section.get("policy_diff") or {})
            if diff_table:
                lines.extend(["", f"### {section_name} Policy Changes", "", diff_table])
    return "\n".join(lines)


def _plan_summary(result: Dict[str, Any], show_layers: bool = False) -> str:
    """Render KV policy plan summary in plain text."""
    lines = [
        f"model: {result.get('model_id', '-')}",
        f"preset: {result.get('preset', {}).get('name', result.get('preset', '-'))}",
        f"chosen_v_quant_layers: {result.get('chosen_v_quantize_only_first_n', '-')}",
    ]
    lines.extend(_policy_summary_lines(result))
    if show_layers:
        table = _policy_layer_table(result)
        if table:
            lines.extend(["", table])
    return "\n".join(lines)


def _plan_diff_summary(result: Dict[str, Any], show_layers: bool = False) -> str:
    """Render KV policy plan diff summary in plain text."""
    base = result.get("base", {})
    compare = result.get("compare", {})
    changed_layers = result.get("changed_layers", []) or []
    lines = [
        f"model: {result.get('model_id', '-')}",
        f"base_preset: {base.get('preset', {}).get('name', '-')}",
        f"compare_preset: {compare.get('preset', {}).get('name', '-')}",
        f"changed_layers: {len(changed_layers)}",
        "",
        "base:",
        _plan_summary(base, show_layers=False),
        "",
        "compare:",
        _plan_summary(compare, show_layers=False),
    ]
    if changed_layers:
        diff_rows = []
        for item in changed_layers:
            for field, values in item.get("changes", {}).items():
                diff_rows.append([item["layer_idx"], field, values["base"], values["compare"]])
        lines.extend(["", _render_table(["layer", "field", "base", "compare"], diff_rows)])
    if show_layers:
        base_table = _policy_layer_table(base)
        compare_table = _policy_layer_table(compare)
        if base_table:
            lines.extend(["", "base layers:", base_table])
        if compare_table:
            lines.extend(["", "compare layers:", compare_table])
    return "\n".join(lines)


def _summary_for_smoke(result: Dict[str, Any], show_layers: bool = False) -> str:
    """Render single smoke test result summary in plain text."""
    exact = result.get("exact", result.get("tuned_exact", "-"))
    speedup = result.get("speedup", result.get("tuned_speedup", 0.0))
    prefix_match = result.get("prefix_match", result.get("tuned_prefix_match"))
    lines = [
        f"model: {result.get('model_id', '-')}",
        f"preset: {result.get('preset', {}).get('name', result.get('preset', '-'))}",
        f"ok: {result.get('ok', '-')}",
        f"exact: {exact}",
        f"speedup: {_fmt_float(speedup)}x",
    ]
    lines.append(f"prefix_match: {_fmt_prefix_match(prefix_match)}")
    chosen = result.get("chosen_v_quantize_only_first_n")
    if chosen is not None:
        lines.append(f"chosen_v_quant_layers: {chosen}")
    lines.extend(_memory_summary_lines(result))
    lines.extend(_turboquant_summary_lines(result))
    lines.extend(_policy_summary_lines(result))
    if show_layers:
        table = _policy_layer_table(result)
        if table:
            lines.extend(["", table])
    return "\n".join(lines)


def _summary_for_smoke_compare(result: Dict[str, Any], show_layers: bool = False) -> str:
    """Render smoke test comparison summary in plain text."""
    base = result.get("base", {})
    compare = result.get("compare", {})
    policy_diff = result.get("policy_diff") or {}
    lines = [
        f"model: {result.get('model_id', '-')}",
        f"prompt_tokens: {result.get('prompt_tokens', '-')}",
        f"base_preset: {base.get('preset', {}).get('name', '-')}",
        f"compare_preset: {compare.get('preset', {}).get('name', '-')}",
        f"base_ok: {base.get('ok', '-')}",
        f"compare_ok: {compare.get('ok', '-')}",
        f"base_exact: {base.get('exact', base.get('tuned_exact', '-'))}",
        f"compare_exact: {compare.get('exact', compare.get('tuned_exact', '-'))}",
        f"base_speedup: {_fmt_float(base.get('speedup', base.get('tuned_speedup', 0.0)))}x",
        f"compare_speedup: {_fmt_float(compare.get('speedup', compare.get('tuned_speedup', 0.0)))}x",
        f"delta_speedup: {_fmt_float(result.get('delta_speedup', 0.0))}x",
        f"changed_layers: {len(policy_diff.get('changed_layers', []) or [])}",
    ]
    lines.extend(_memory_compare_summary_lines(base, compare))
    lines.extend(_turboquant_compare_summary_lines(base, compare))
    changed_layers = policy_diff.get("changed_layers", []) or []
    if changed_layers:
        diff_rows = []
        for item in changed_layers:
            for field, values in item.get("changes", {}).items():
                diff_rows.append([item["layer_idx"], field, values["base"], values["compare"]])
        lines.extend(["", _render_table(["layer", "field", "base", "compare"], diff_rows)])
    if show_layers:
        base_table = _policy_layer_table(base)
        compare_table = _policy_layer_table(compare)
        if base_table:
            lines.extend(["", "base layers:", base_table])
        if compare_table:
            lines.extend(["", "compare layers:", compare_table])
    return "\n".join(lines)


def _summary_for_long(result: Dict[str, Any], show_layers: bool = False) -> str:
    """Render long-context benchmark result summary in plain text."""
    lines = [
        f"model: {result.get('model_id', '-')}",
        f"preset: {result.get('preset', {}).get('name', result.get('preset', '-'))}",
        f"prompt_tokens: {result.get('prompt_tokens', '-')}",
        f"tuned_exact: {result.get('tuned_exact', '-')}",
        f"tuned_speedup: {_fmt_float(result.get('tuned_speedup', 0.0))}x",
    ]
    lines.append(f"prefix_match: {_fmt_prefix_match(result.get('tuned_prefix_match'))}")
    chosen = result.get("chosen_v_quantize_only_first_n")
    if chosen is not None:
        lines.append(f"chosen_v_quant_layers: {chosen}")
    lines.extend(_memory_summary_lines(result))
    lines.extend(_turboquant_summary_lines(result))
    lines.extend(_policy_summary_lines(result))
    if show_layers:
        table = _policy_layer_table(result)
        if table:
            lines.extend(["", table])
    return "\n".join(lines)


def _summary_for_long_compare(result: Dict[str, Any], show_layers: bool = False) -> str:
    """Render long-context benchmark comparison summary in plain text."""
    base = result.get("base", {})
    compare = result.get("compare", {})
    policy_diff = result.get("policy_diff") or {}
    lines = [
        f"model: {result.get('model_id', '-')}",
        f"prompt_tokens: {result.get('prompt_tokens', '-')}",
        f"base_preset: {base.get('preset', {}).get('name', '-')}",
        f"compare_preset: {compare.get('preset', {}).get('name', '-')}",
        f"base_exact: {base.get('tuned_exact', '-')}",
        f"compare_exact: {compare.get('tuned_exact', '-')}",
        f"base_speedup: {_fmt_float(base.get('tuned_speedup', 0.0))}x",
        f"compare_speedup: {_fmt_float(compare.get('tuned_speedup', 0.0))}x",
        f"delta_speedup: {_fmt_float(result.get('delta_speedup', 0.0))}x",
        f"changed_layers: {len(policy_diff.get('changed_layers', []) or [])}",
    ]
    lines.extend(_memory_compare_summary_lines(base, compare))
    lines.extend(_turboquant_compare_summary_lines(base, compare))
    changed_layers = policy_diff.get("changed_layers", []) or []
    if changed_layers:
        diff_rows = []
        for item in changed_layers:
            for field, values in item.get("changes", {}).items():
                diff_rows.append([item["layer_idx"], field, values["base"], values["compare"]])
        lines.extend(["", _render_table(["layer", "field", "base", "compare"], diff_rows)])
    if show_layers:
        base_table = _policy_layer_table(base)
        compare_table = _policy_layer_table(compare)
        if base_table:
            lines.extend(["", "base layers:", base_table])
        if compare_table:
            lines.extend(["", "compare layers:", compare_table])
    return "\n".join(lines)


def _summary_for_suite(result: Dict[str, Any], show_layers: bool = False) -> str:
    """Render benchmark suite result summary in plain text."""
    lines = [
        f"model: {result.get('model_id', '-')}",
        f"preset: {result.get('preset', {}).get('name', result.get('preset', '-'))}",
        f"all_cases_exact_match: {result.get('all_cases_exact_match', '-')}",
        f"chosen_v_quant_layers: {result.get('chosen_v_quantize_only_first_n', '-')}",
    ]
    lines.extend(_policy_summary_lines(result))
    rows = result.get("results", []) or []
    if rows:
        lines.extend(
            [
                "",
                _render_table(
                    ["case", "speedup", "exact", "prefix_match", "base tok/s", "tuned tok/s"],
                    [
                        [
                            row.get("name", "-"),
                            f"{_fmt_float(row.get('speedup', 0.0))}x",
                            row.get("exact", "-"),
                            _fmt_prefix_match(row.get("prefix_match")),
                            _fmt_float(row.get("base_toks_per_s", 0.0), 2),
                            _fmt_float(row.get("tuned_toks_per_s", 0.0), 2),
                        ]
                        for row in rows
                    ],
                ),
            ]
        )
    if show_layers:
        table = _policy_layer_table(result)
        if table:
            lines.extend(["", table])
    return "\n".join(lines)


def _summary_for_suite_compare(result: Dict[str, Any], show_layers: bool = False) -> str:
    """Render benchmark suite comparison summary in plain text."""
    base = result.get("base", {})
    compare = result.get("compare", {})
    policy_diff = result.get("policy_diff") or {}
    lines = [
        f"model: {result.get('model_id', '-')}",
        f"base_preset: {base.get('preset', {}).get('name', '-')}",
        f"compare_preset: {compare.get('preset', {}).get('name', '-')}",
        f"changed_layers: {len(policy_diff.get('changed_layers', []) or [])}",
    ]
    rows = _suite_compare_rows(result)
    if rows:
        lines.extend(
            [
                "",
                _render_table(
                    ["case", "base speedup", "compare speedup", "delta", "base exact", "compare exact"],
                    [
                        [
                            row.get("name", "-"),
                            f"{_fmt_float(row.get('base_speedup', 0.0))}x",
                            f"{_fmt_float(row.get('compare_speedup', 0.0))}x",
                            f"{_fmt_float(row.get('delta_speedup', 0.0))}x",
                            row.get("base_exact", "-"),
                            row.get("compare_exact", "-"),
                        ]
                        for row in rows
                    ],
                ),
            ]
        )
    changed_layers = policy_diff.get("changed_layers", []) or []
    if changed_layers:
        diff_rows = []
        for item in changed_layers:
            for field, values in item.get("changes", {}).items():
                diff_rows.append([item["layer_idx"], field, values["base"], values["compare"]])
        lines.extend(["", _render_table(["layer", "field", "base", "compare"], diff_rows)])
    if show_layers:
        base_table = _policy_layer_table(base)
        compare_table = _policy_layer_table(compare)
        if base_table:
            lines.extend(["", "base layers:", base_table])
        if compare_table:
            lines.extend(["", "compare layers:", compare_table])
    return "\n".join(lines)


def _summary_for_report(result: Dict[str, Any], show_layers: bool = False) -> str:
    """Render combined benchmark report summary in plain text."""
    smoke, long_result, suite = _report_sections(result)
    lines = [
        f"model: {result.get('model_id', '-')}",
        f"base_preset: {result.get('base_preset', '-')}",
        f"compare_preset: {result.get('compare_preset', '-')}",
    ]
    if smoke:
        lines.extend(
            [
                "",
                "smoke:",
                f"delta_speedup: {_fmt_float(smoke.get('delta_speedup', 0.0))}x",
                f"base_exact: {smoke.get('base', {}).get('exact', smoke.get('base', {}).get('tuned_exact', '-'))}",
                f"compare_exact: {smoke.get('compare', {}).get('exact', smoke.get('compare', {}).get('tuned_exact', '-'))}",
            ]
        )
        lines.extend(_memory_compare_summary_lines(smoke.get("base", {}) or {}, smoke.get("compare", {}) or {}))
        lines.extend(_turboquant_compare_summary_lines(smoke.get("base", {}) or {}, smoke.get("compare", {}) or {}))
    if long_result:
        lines.extend(
            [
                "",
                "long:",
                f"delta_speedup: {_fmt_float(long_result.get('delta_speedup', 0.0))}x",
                f"base_exact: {long_result.get('base', {}).get('tuned_exact', '-')}",
                f"compare_exact: {long_result.get('compare', {}).get('tuned_exact', '-')}",
            ]
        )
        lines.extend(_memory_compare_summary_lines(long_result.get("base", {}) or {}, long_result.get("compare", {}) or {}))
        lines.extend(_turboquant_compare_summary_lines(long_result.get("base", {}) or {}, long_result.get("compare", {}) or {}))
    if suite:
        lines.extend(
            [
                "",
                "suite:",
                f"cases: {len(suite.get('case_diffs', []) or [])}",
                f"changed_layers: {len((suite.get('policy_diff') or {}).get('changed_layers', []) or [])}",
            ]
        )
        rows = _suite_compare_rows(suite)
        if rows:
            lines.extend(
                [
                    "",
                    _render_table(
                        ["case", "base speedup", "compare speedup", "delta", "base exact", "compare exact"],
                        [
                            [
                                row.get("name", "-"),
                                f"{_fmt_float(row.get('base_speedup', 0.0))}x",
                                f"{_fmt_float(row.get('compare_speedup', 0.0))}x",
                                f"{_fmt_float(row.get('delta_speedup', 0.0))}x",
                                row.get("base_exact", "-"),
                                row.get("compare_exact", "-"),
                            ]
                            for row in rows
                        ],
                    ),
                ]
            )
    if show_layers:
        for section_name, section in [("smoke", smoke), ("long", long_result), ("suite", suite)]:
            changed_layers = (section.get("policy_diff") or {}).get("changed_layers", []) or []
            if changed_layers:
                diff_rows = []
                for item in changed_layers:
                    for field, values in item.get("changes", {}).items():
                        diff_rows.append([item["layer_idx"], field, values["base"], values["compare"]])
                lines.extend(["", f"{section_name} policy changes:", _render_table(["layer", "field", "base", "compare"], diff_rows)])
    return "\n".join(lines)


def _markdown_metadata(kind: str, show_layers: bool, output: Optional[str], extras: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build metadata dict for Markdown report headers."""
    metadata: Dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": f"bench {kind}",
        "format": "markdown",
        "show_layers": show_layers,
        "output": output or "-",
    }
    if extras:
        metadata.update(extras)
    return metadata


def _emit_formatted_result(
    result: Dict[str, Any],
    fmt: str,
    output: Optional[str],
    kind: str,
    show_layers: bool = False,
    sort_by: str = "layer",
    markdown_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Dispatch result to the correct formatter based on --format and command kind.

    Args:
        result: Benchmark result dict.
        fmt: One of 'json', 'summary', 'markdown'.
        output: Optional file path to write to.
        kind: Command kind (smoke, long, plan, etc.).
        show_layers: Whether to include per-layer tables.
        sort_by: Sort criterion for layer tables.
        markdown_meta: Metadata injected into Markdown headers.
    """
    if fmt == "json":
        _emit_result(result, output)
        return

    if fmt == "markdown":
        if kind == "kv-analyze":
            _emit_text(_kv_analyze_markdown(result, show_layers=show_layers, sort_by=sort_by, metadata=markdown_meta), output)
            return
        if kind == "smoke":
            _emit_text(_markdown_for_smoke(result, show_layers=show_layers, metadata=markdown_meta), output)
            return
        if kind == "smoke-compare":
            _emit_text(_markdown_for_smoke_compare(result, show_layers=show_layers, metadata=markdown_meta), output)
            return
        if kind == "long":
            _emit_text(_markdown_for_long(result, show_layers=show_layers, metadata=markdown_meta), output)
            return
        if kind == "long-compare":
            _emit_text(_markdown_for_long_compare(result, show_layers=show_layers, metadata=markdown_meta), output)
            return
        if kind == "plan":
            _emit_text(_markdown_for_plan(result, show_layers=show_layers, metadata=markdown_meta), output)
            return
        if kind == "plan-diff":
            _emit_text(_markdown_for_plan_diff(result, show_layers=show_layers, metadata=markdown_meta), output)
            return
        if kind == "suite-compare":
            _emit_text(_markdown_for_suite_compare(result, show_layers=show_layers, metadata=markdown_meta), output)
            return
        if kind == "report":
            _emit_text(_markdown_for_report(result, show_layers=show_layers, metadata=markdown_meta), output)
            return
        _emit_text(_markdown_for_suite(result, show_layers=show_layers, metadata=markdown_meta), output)
        return

    if output:
        raise click.BadParameter("--output is only supported with --format json")

    if kind == "kv-analyze":
        click.echo(_kv_analyze_summary_with_layers(result, show_layers=show_layers, sort_by=sort_by))
        return
    if kind == "smoke":
        click.echo(_summary_for_smoke(result, show_layers=show_layers))
        return
    if kind == "smoke-compare":
        click.echo(_summary_for_smoke_compare(result, show_layers=show_layers))
        return
    if kind == "long":
        click.echo(_summary_for_long(result, show_layers=show_layers))
        return
    if kind == "long-compare":
        click.echo(_summary_for_long_compare(result, show_layers=show_layers))
        return
    if kind == "plan":
        click.echo(_plan_summary(result, show_layers=show_layers))
        return
    if kind == "plan-diff":
        click.echo(_plan_diff_summary(result, show_layers=show_layers))
        return
    if kind == "suite-compare":
        click.echo(_summary_for_suite_compare(result, show_layers=show_layers))
        return
    if kind == "report":
        click.echo(_summary_for_report(result, show_layers=show_layers))
        return
    click.echo(_summary_for_suite(result, show_layers=show_layers))


def _turboquantum_summary_lines(result: Dict[str, Any]) -> list[str]:
    """Render TurboQuantum synthetic benchmark summary lines."""
    if not result.get("ok"):
        return [f"error: {result.get('error', 'unknown')}"]
    lines = [
        f"device: {result.get('device', '-')}",
        f"shape: b={result['shape']['batch']}, h={result['shape']['heads']}, "
        f"s={result['shape']['seq_len']}, d={result['shape']['head_dim']}",
        f"mode: {result.get('mode')}",
        "",
        "compression:",
        f"  effective_bpv: {_fmt_float(result['compression'].get('effective_bpv', 0))}",
        f"  savings_pct: {_fmt_float(result['compression'].get('savings_percent', 0))}%",
        f"  original_kb: {_fmt_float(result['compression'].get('original_kb', 0), 1)}",
        f"  compressed_kb: {_fmt_float(result['compression'].get('compressed_kb', 0), 1)}",
        "",
        "quality:",
        f"  k_mse: {_fmt_float(result['quality'].get('k_mse', 0), 6)}",
        f"  v_mse: {_fmt_float(result['quality'].get('v_mse', 0), 6)}",
        f"  k_cosine: {_fmt_float(result['quality'].get('k_cosine', 0), 4)}",
        f"  v_cosine: {_fmt_float(result['quality'].get('v_cosine', 0), 4)}",
        "",
        f"timing_ms: {result.get('timing_ms', 0)}",
    ]
    tunnel = result.get("tunneling_stats", {})
    if tunnel:
        lines.extend([
            "tunneling:",
            f"  protected_tokens_pct: {_fmt_float(tunnel.get('protected_tokens_pct', 0), 1)}%",
            f"  tunneling_enabled: {tunnel.get('tunneling_enabled', False)}",
        ])
    return lines


def _turboquantum_mode_comparison_table(comparison: list) -> str:
    """Render TurboQuantum mode comparison as a text table."""
    if not comparison:
        return "(no comparison data)"
    headers = ["mode", "bpv", "k_cos", "v_cos", "savings%", "time_ms"]
    rows = [
        [row["mode"], _fmt_float(row["bpv"]), _fmt_float(row["k_cosine"], 4),
         _fmt_float(row["v_cosine"], 4), _fmt_float(row["savings_pct"], 1), _fmt_float(row["time_ms"])]
        for row in comparison
    ]
    return _render_table(headers, rows)


def _markdown_for_turboquantum_synthetic(result: Dict[str, Any]) -> str:
    """Render TurboQuantum synthetic benchmark in Markdown."""
    lines = ["## TurboQuantum Synthetic Benchmark", ""]
    for line in _turboquantum_summary_lines(result):
        lines.append(f"- `{line.split(': ', 1)[0]}`: {line.split(': ', 1)[1]}" if ": " in line and not line.startswith(" ") else line)
    return "\n".join(lines)
