"""Bench report generator."""

import json
from html import escape
from typing import Any, Dict, Optional


def generate_report(
    result: Dict[str, Any],
    output_format: str = "markdown",
    show_layers: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate bench report in specified format."""
    if output_format == "markdown":
        return _generate_markdown_report(result, show_layers, metadata)
    elif output_format == "html":
        return _generate_html_report(result, show_layers, metadata)
    elif output_format == "json":
        return _generate_json_report(result, show_layers, metadata)
    else:
        raise ValueError(f"Unsupported format: {output_format}")


def _generate_markdown_report(
    result: Dict[str, Any],
    show_layers: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    payload = _normalize_report_payload(result, show_layers=show_layers)
    lines = []
    lines.append("# Bench Report")
    lines.append("")
    lines.append(f"Model: {payload.get('model_id', 'N/A')}")
    if metadata:
        lines.append("")
        lines.append("## Metadata")
        for key in sorted(metadata):
            lines.append(f"- {key}: {metadata[key]}")
    if payload:
        lines.append("")
        lines.append("## Summary")
        for key in sorted(payload):
            if key == "model_id":
                continue
            lines.append(f"- {key}: {payload[key]}")
    return "\n".join(lines)


def _generate_html_report(
    result: Dict[str, Any],
    show_layers: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    markdown = _generate_markdown_report(result, show_layers=show_layers, metadata=metadata)
    return (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head><meta charset=\"utf-8\"><title>Bench Report</title></head>\n"
        "<body>\n"
        f"<pre>{escape(markdown)}</pre>\n"
        "</body>\n"
        "</html>"
    )


def _generate_json_report(
    result: Dict[str, Any],
    show_layers: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    payload = _normalize_report_payload(result, show_layers=show_layers)
    if metadata:
        payload = {**payload, "metadata": metadata}
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)


def _normalize_report_payload(result: Dict[str, Any], *, show_layers: bool) -> Dict[str, Any]:
    """Drop bulky layer payloads unless the caller explicitly requests them."""

    def _normalize(value: Any) -> Any:
        if isinstance(value, dict):
            items = {}
            for key, item in value.items():
                if key == "layers" and not show_layers:
                    continue
                items[key] = _normalize(item)
            return items
        if isinstance(value, list):
            return [_normalize(item) for item in value]
        return value

    return _normalize(result)
