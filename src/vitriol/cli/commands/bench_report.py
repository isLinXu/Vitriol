"""Bench report generator."""

from typing import Dict, Any, Optional, List


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
    lines = []
    lines.append("# Bench Report")
    lines.append("")
    lines.append(f"Model: {result.get('model_id', 'N/A')}")
    # ... more lines
    return "\n".join(lines)
