"""
Evolution Tree Visualizer
=========================

Visualize architecture evolution trees as interactive HTML.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from .tree_builder import EvolutionTree, ArchNode

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HTML Template for Evolution Tree
# ─────────────────────────────────────────────────────────────────────────────

EVOLUTION_TREE_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vitriol Architecture Evolution Tree</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
        }

        .header {
            background: rgba(0, 0, 0, 0.3);
            padding: 20px 40px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .header h1 {
            font-size: 24px;
            font-weight: 600;
            background: linear-gradient(90deg, #00d4ff, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .header p {
            font-size: 14px;
            color: #888;
            margin-top: 5px;
        }

        #tree-container {
            width: 100%;
            height: calc(100vh - 100px);
            overflow: hidden;
        }

        .node {
            cursor: pointer;
        }

        .node circle {
            stroke: #fff;
            stroke-width: 2px;
            transition: all 0.3s ease;
        }

        .node:hover circle {
            stroke-width: 4px;
            filter: drop-shadow(0 0 10px rgba(0, 212, 255, 0.5));
        }

        .node text {
            font-size: 11px;
            fill: #e0e0e0;
            text-anchor: middle;
            pointer-events: none;
        }

        .link {
            fill: none;
            stroke: #4a5568;
            stroke-width: 2px;
            stroke-opacity: 0.6;
        }

        .link:hover {
            stroke: #00d4ff;
            stroke-opacity: 1;
        }

        .tooltip {
            position: absolute;
            background: rgba(0, 0, 0, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            padding: 15px;
            font-size: 12px;
            max-width: 350px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            z-index: 1000;
        }

        .tooltip.visible {
            opacity: 1;
        }

        .tooltip h3 {
            font-size: 14px;
            margin-bottom: 8px;
            color: #00d4ff;
        }

        .tooltip .family {
            font-size: 11px;
            color: #888;
            margin-bottom: 10px;
        }

        .tooltip .params {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 5px;
            margin-bottom: 10px;
        }

        .tooltip .param {
            background: rgba(255, 255, 255, 0.1);
            padding: 4px 8px;
            border-radius: 4px;
        }

        .tooltip .param span {
            color: #888;
            font-size: 10px;
        }

        .tooltip .param strong {
            color: #fff;
            font-size: 12px;
        }

        .tooltip .innovations {
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            padding-top: 10px;
            margin-top: 10px;
        }

        .tooltip .innovation {
            display: inline-block;
            background: linear-gradient(90deg, #7c3aed, #00d4ff);
            color: #fff;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 10px;
            margin: 2px;
        }

        .legend {
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 15px;
            font-size: 11px;
        }

        .legend h4 {
            margin-bottom: 10px;
            color: #888;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .legend-item {
            display: flex;
            align-items: center;
            margin-bottom: 5px;
        }

        .legend-color {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }

        .controls {
            position: fixed;
            top: 80px;
            right: 20px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .control-btn {
            background: rgba(0, 0, 0, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #fff;
            padding: 10px 15px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
        }

        .control-btn:hover {
            background: rgba(0, 212, 255, 0.2);
            border-color: #00d4ff;
        }

        .stats {
            position: fixed;
            top: 80px;
            left: 20px;
            background: rgba(0, 0, 0, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 15px;
            font-size: 12px;
        }

        .stats h4 {
            color: #888;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }

        .stat-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }

        .stat-label {
            color: #888;
        }

        .stat-value {
            color: #00d4ff;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{title}}</h1>
        <p>{{description}}</p>
    </div>

    <div class="stats">
        <h4>Statistics</h4>
        <div class="stat-item">
            <span class="stat-label">Total Models</span>
            <span class="stat-value">{{total_models}}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Families</span>
            <span class="stat-value">{{total_families}}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Innovations</span>
            <span class="stat-value">{{total_innovations}}</span>
        </div>
    </div>

    <div class="controls">
        <button class="control-btn" onclick="zoomIn()">Zoom In</button>
        <button class="control-btn" onclick="zoomOut()">Zoom Out</button>
        <button class="control-btn" onclick="resetZoom()">Reset</button>
        <button class="control-btn" onclick="toggleLabels()">Toggle Labels</button>
    </div>

    <div id="tree-container"></div>

    <div class="tooltip" id="tooltip">
        <h3 id="tooltip-title"></h3>
        <div class="family" id="tooltip-family"></div>
        <div class="params" id="tooltip-params"></div>
        <div class="innovations" id="tooltip-innovations"></div>
    </div>

    <script>
        // Tree data
        const treeData = {{tree_json}};

        // Family colors
        const familyColors = {{family_colors}};

        // Current zoom state
        let currentZoom = 1;
        let showLabels = true;

        // Create SVG
        const container = document.getElementById('tree-container');
        const width = container.clientWidth;
        const height = container.clientHeight;

        const svg = d3.select('#tree-container')
            .append('svg')
            .attr('width', '100%')
            .attr('height', '100%')
            .call(d3.zoom()
                .scaleExtent([0.1, 4])
                .on('zoom', (event) => {
                    currentZoom = event.transform.k;
                    g.attr('transform', event.transform);
                }))
            .append('g');

        const g = svg.append('g')
            .attr('transform', `translate(${width / 2}, 60)`);

        // Build hierarchy
        const rootNode = d3.stratify()
            .id(d => d.id)
            .parentId(d => d.parent)
            (treeData.nodes);

        const treeLayout = d3.tree()
            .size([width - 200, height - 150])
            .separation((a, b) => (a.parent === b.parent ? 1.5 : 2.5));

        const tree = treeLayout(rootNode);

        // Draw links
        g.selectAll('.link')
            .data(tree.links())
            .enter()
            .append('path')
            .attr('class', 'link')
            .attr('d', d3.linkVertical()
                .x(d => d.x - width / 2 + 100)
                .y(d => d.y + 60));

        // Draw nodes
        const nodes = g.selectAll('.node')
            .data(tree.descendants())
            .enter()
            .append('g')
            .attr('class', 'node')
            .attr('transform', d => `translate(${d.x - width / 2 + 100}, ${d.y + 60})`);

        nodes.append('circle')
            .attr('r', 8)
            .attr('fill', d => familyColors[d.data.family] || '#7c3aed')
            .on('mouseover', showTooltip)
            .on('mouseout', hideTooltip)
            .on('click', d => {
                window.open(`https://huggingface.co/${d.data.id}`, '_blank');
            });

        nodes.append('text')
            .attr('dy', 25)
            .text(d => d.data.shortName || d.data.name)
            .style('opacity', showLabels ? 1 : 0);

        // Tooltip functions
        function showTooltip(event, d) {
            const tooltip = document.getElementById('tooltip');
            document.getElementById('tooltip-title').textContent = d.data.name;
            document.getElementById('tooltip-family').textContent = d.data.family + ' Family';

            // Params
            const params = d.data.params || {};
            const paramsHtml = Object.entries(params).slice(0, 6).map(([k, v]) => `
                <div class="param">
                    <span>${k}</span><br>
                    <strong>${v || 'N/A'}</strong>
                </div>
            `).join('');
            document.getElementById('tooltip-params').innerHTML = paramsHtml;

            // Innovations
            const innovations = d.data.innovations || [];
            const innHtml = innovations.map(i =>
                `<span class="innovation">${i.name}</span>`
            ).join('');
            document.getElementById('tooltip-innovations').innerHTML =
                innovations.length ? `<strong>Innovations:</strong><br>${innHtml}` : '';

            tooltip.style.left = (event.pageX + 15) + 'px';
            tooltip.style.top = (event.pageY - 10) + 'px';
            tooltip.classList.add('visible');
        }

        function hideTooltip() {
            document.getElementById('tooltip').classList.remove('visible');
        }

        // Control functions
        function zoomIn() {
            svg.transition().call(d3.zoom().scaleBy, 1.3);
        }

        function zoomOut() {
            svg.transition().call(d3.zoom().scaleBy, 0.7);
        }

        function resetZoom() {
            svg.transition().call(d3.zoom().transform, d3.zoomIdentity);
        }

        function toggleLabels() {
            showLabels = !showLabels;
            g.selectAll('.node text')
                .transition()
                .style('opacity', showLabels ? 1 : 0);
        }
    </script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Tree Visualizer
# ─────────────────────────────────────────────────────────────────────────────

class TreeVisualizer:
    """
    Visualize architecture evolution trees as interactive HTML.

    Usage:
        viz = TreeVisualizer(tree)
        viz.generate_html("output/evolution_tree.html")
        # Or for web UI:
        html_content = viz.generate_html_string()
    """

    # Family colors for visualization
    DEFAULT_COLORS = {
        "Qwen": "#06b6d4",
        "LLaMA": "#f59e0b",
        "DeepSeek": "#ef4444",
        "Mistral": "#8b5cf6",
        "GLM": "#10b981",
        "Kimi": "#ec4899",
        "GPT": "#6366f1",
        "Yi": "#f97316",
        "Phi": "#14b8a6",
        "Gemma": "#8b5cf6",
        "Starcoder": "#3b82f6",
        "BLOOM": "#84cc16",
        "Falcon": "#f43f5e",
        "Other": "#7c3aed",
    }

    def __init__(self, tree: "EvolutionTree"):
        """
        Initialize the visualizer.

        Args:
            tree: EvolutionTree instance to visualize
        """
        self.tree = tree

    def generate_html(
        self,
        output_path: str,
        title: str = "Architecture Evolution Tree",
        description: str = "LLM Model Family Relationships and Innovations",
        show_innovations: bool = True,
    ) -> str:
        """
        Generate an interactive HTML visualization of the evolution tree.

        Args:
            output_path: Path to save the HTML file
            title: Title for the visualization
            description: Description subtitle
            show_innovations: Whether to show innovations in tooltips

        Returns:
            Path to the generated HTML file
        """
        html_content = self.generate_html_string(
            title=title,
            description=description,
            show_innovations=show_innovations,
        )

        # Write to file
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Generated evolution tree HTML: {output_path}")
        return output_path

    def generate_html_string(
        self,
        title: str = "Architecture Evolution Tree",
        description: str = "LLM Model Family Relationships and Innovations",
        show_innovations: bool = True,
    ) -> str:
        """
        Generate HTML content as a string (useful for web frameworks like Gradio).

        Args:
            title: Title for the visualization
            description: Description subtitle
            show_innovations: Whether to show innovations in tooltips

        Returns:
            HTML content as string
        """
        # Prepare tree data for D3.js
        nodes_data = []
        links_data = []

        # Find root nodes (no parent)
        for model_id, node in self.tree.nodes.items():
            node_dict = {
                "id": model_id,
                "name": node.model_name,
                "shortName": self._truncate_name(node.model_name),
                "family": node.family,
                "params": node.get_key_params(),
                "innovations": [
                    {"name": i.name, "description": i.description}
                    for i in node.innovations
                ],
            }
            nodes_data.append(node_dict)

            if node.parent:
                links_data.append({
                    "id": model_id,
                    "parent": node.parent,
                })


        # Count innovations
        total_innovations = sum(
            len(node.innovations) for node in self.tree.nodes.values()
        )

        # Prepare template variables
        template_vars = {
            "title": title,
            "description": description,
            "tree_json": json.dumps(nodes_data),
            "family_colors": json.dumps(self.DEFAULT_COLORS),
            "total_models": len(self.tree.nodes),
            "total_families": len(set(node.family for node in self.tree.nodes.values())),
            "total_innovations": total_innovations,
        }

        # Generate HTML
        html_content = EVOLUTION_TREE_HTML
        for key, value in template_vars.items():
            placeholder = "{{" + key + "}}"
            if isinstance(value, str):
                html_content = html_content.replace(placeholder, value)
            elif isinstance(value, (int, float)):
                html_content = html_content.replace(placeholder, str(value))

        return html_content

    def _truncate_name(self, name: str, max_len: int = 20) -> str:
        """Truncate model name for display."""
        if len(name) <= max_len:
            return name
        return name[: max_len - 3] + "..."

    def generate_markdown_report(self) -> str:
        """
        Generate a Markdown summary of the evolution tree.

        Returns:
            Markdown formatted report
        """
        lines = [
            "# Architecture Evolution Summary",
            "",
            f"**Total Models:** {len(self.tree.nodes)}",
            f"**Families:** {len(set(n.family for n in self.tree.nodes.values()))}",
            "",
            "## Model Families",
            "",
        ]

        # Group by family
        families: Dict[str, List["ArchNode"]] = {}
        for node in self.tree.nodes.values():
            if node.family not in families:
                families[node.family] = []
            families[node.family].append(node)

        for family, nodes in sorted(families.items()):
            lines.append(f"### {family}")
            lines.append("")
            for node in nodes:
                parent = f" ← {node.parent.split('/')[-1]}" if node.parent else ""
                innovations = (
                    " | ".join(f"`{i.name}`" for i in node.innovations) or "—"
                )
                lines.append(
                    f"- **{node.model_name}**{parent} "
                    f"(Innovations: {innovations})"
                )
            lines.append("")

        # Innovation timeline
        lines.append("## Innovation Timeline")
        lines.append("")
        timeline = self.tree.get_innovation_timeline()
        if timeline:
            for item in timeline:
                lines.append(
                    f"- **{item['year']}** {item['family']}: "
                    f"`{item['innovation']}` — {item['description']}"
                )
        else:
            lines.append("No innovations recorded.")

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Module Exports
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "TreeVisualizer",
]
