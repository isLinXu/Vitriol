"""
Innovation Timeline
==================

Visualizes architecture innovations over time.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from .tree_builder import EvolutionTree

logger = logging.getLogger(__name__)


@dataclass
class TimelineEvent:
    """A single event in the innovation timeline."""
    year: int
    month: int
    innovation: str
    description: str
    model_id: str
    family: str
    impact: str  # high, medium, low


class InnovationTimeline:
    """
    Builds and visualizes the timeline of architecture innovations.
    """

    def __init__(self, evolution_tree: Optional[EvolutionTree] = None):
        self.tree = evolution_tree or EvolutionTree()
        self.tree.load_builtin_families()
        self.tree.build()
        self.events: List[TimelineEvent] = []

    def build_events(self) -> List[TimelineEvent]:
        """Extract all innovation events from the evolution tree."""
        self.events = []

        for _node_id, node in self.tree.nodes.items():
            for innovation in node.innovations:
                event = TimelineEvent(
                    year=innovation.year,
                    month=6,
                    innovation=innovation.name,
                    description=innovation.description,
                    model_id=innovation.introduced_in,
                    family=node.family,
                    impact=self._assess_impact(innovation.name),
                )
                self.events.append(event)

        self.events.sort(key=lambda e: (e.year, e.month))
        return self.events

    def _assess_impact(self, innovation_name: str) -> str:
        """Assess the impact level of an innovation."""
        high_impact = {
            "MoE", "GQA", "MLA", "SSM", "Long Context", "128K Context", "1M Context",
            "Multi-Token Prediction", "Hybrid MoE", "FP8 Training", "WKV",
            "RNN-Transformer", "Linear Complexity", "Multi-lingual", "176B Params",
        }
        medium_impact = {
            "RoPE", "SwiGLU", "Sliding Window", "Rope Scaling",
            "Fine-grained Expert", "DeepSeek MoE", "GLM Embedding",
            "Multi-Query Attention", "Dynamic NTK", "Textbooks",
            "VL", "Tool Use", "Open Weights", "FIM",
        }

        if innovation_name in high_impact:
            return "high"
        elif innovation_name in medium_impact:
            return "medium"
        return "low"

    def get_innovation_by_year(self) -> Dict[int, List[TimelineEvent]]:
        """Group innovations by year."""
        by_year: Dict[int, List[TimelineEvent]] = defaultdict(list)
        for event in self.events:
            by_year[event.year].append(event)
        return dict(by_year)

    def generate_html(self, title: str = "LLM Architecture Innovation Timeline") -> str:
        """Generate an interactive HTML timeline visualization."""
        if not self.events:
            self.build_events()

        by_year = self.get_innovation_by_year()
        years = sorted(by_year.keys())

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #eee;
            padding-bottom: 60px;
        }}
        .header {{
            text-align: center;
            padding: 40px 20px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            margin-bottom: 30px;
        }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header p {{ font-size: 1.2em; opacity: 0.9; }}
        .timeline-container {{ max-width: 1400px; margin: 0 auto; padding: 0 20px; }}
        .stats {{
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}
        .stat {{
            text-align: center;
            background: rgba(255, 255, 255, 0.05);
            padding: 20px 30px;
            border-radius: 12px;
        }}
        .stat-value {{ font-size: 2.5em; font-weight: bold; color: #667eea; }}
        .stat-label {{ font-size: 0.9em; color: #aaa; margin-top: 5px; }}
        .legend {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; }}
        .legend-color {{ width: 20px; height: 20px; border-radius: 4px; }}
        .legend-color.high {{ background: #ef4444; }}
        .legend-color.medium {{ background: #f59e0b; }}
        .legend-color.low {{ background: #10b981; }}
        .timeline {{ position: relative; padding: 20px 0; }}
        .timeline::before {{
            content: '';
            position: absolute;
            left: 50%;
            transform: translateX(-50%);
            width: 4px;
            height: 100%;
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
            border-radius: 2px;
        }}
        .year-section {{ margin-bottom: 40px; }}
        .year-label {{
            text-align: center;
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 20px;
            text-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
        }}
        .events-row {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; }}
        .event-card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 20px;
            width: 300px;
            border-left: 4px solid;
            transition: transform 0.3s, box-shadow 0.3s;
            backdrop-filter: blur(10px);
        }}
        .event-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }}
        .event-card.high {{ border-color: #ef4444; }}
        .event-card.medium {{ border-color: #f59e0b; }}
        .event-card.low {{ border-color: #10b981; }}
        .event-innovation {{ font-size: 1.2em; font-weight: bold; margin-bottom: 8px; }}
        .event-card.high .event-innovation {{ color: #ef4444; }}
        .event-card.medium .event-innovation {{ color: #f59e0b; }}
        .event-card.low .event-innovation {{ color: #10b981; }}
        .event-model {{ font-size: 0.9em; color: #aaa; margin-bottom: 8px; }}
        .event-description {{ font-size: 0.95em; color: #ccc; margin-bottom: 10px; line-height: 1.4; }}
        .event-meta {{ display: flex; justify-content: space-between; align-items: center; }}
        .event-family {{ font-size: 0.8em; background: rgba(102, 126, 234, 0.3); padding: 4px 10px; border-radius: 20px; }}
        .event-impact {{ font-size: 0.75em; text-transform: uppercase; letter-spacing: 1px; }}
        .event-card.high .event-impact {{ color: #ef4444; }}
        .event-card.medium .event-impact {{ color: #f59e0b; }}
        .event-card.low .event-impact {{ color: #10b981; }}
        @media (max-width: 768px) {{
            .timeline::before {{ left: 20px; }}
            .event-card {{ width: calc(100% - 50px); margin-left: 40px; }}
            .events-row {{ justify-content: flex-start; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p>Tracing the evolution of LLM architectures through key innovations</p>
    </div>
    <div class="timeline-container">
        <div class="stats">
            <div class="stat"><div class="stat-value">{len(self.events)}</div><div class="stat-label">Total Innovations</div></div>
            <div class="stat"><div class="stat-value">{len(years)}</div><div class="stat-label">Years of Evolution</div></div>
            <div class="stat"><div class="stat-value">{len(self.tree.families)}</div><div class="stat-label">Model Families</div></div>
        </div>
        <div class="legend">
            <div class="legend-item"><div class="legend-color high"></div><span>High Impact (MoE, GQA, SSM)</span></div>
            <div class="legend-item"><div class="legend-color medium"></div><span>Medium Impact (RoPE, SwiGLU)</span></div>
            <div class="legend-item"><div class="legend-color low"></div><span>Standard Features</span></div>
        </div>
        <div class="timeline">
'''

        for year in years:
            events = by_year[year]
            html += f'''
            <div class="year-section">
                <div class="year-label">{year}</div>
                <div class="events-row">
'''
            for event in events:
                html += f'''
                    <div class="event-card {event.impact}">
                        <div class="event-innovation">{event.innovation}</div>
                        <div class="event-model">{event.model_id}</div>
                        <div class="event-description">{event.description}</div>
                        <div class="event-meta">
                            <span class="event-family">{event.family}</span>
                            <span class="event-impact">{event.impact}</span>
                        </div>
                    </div>
'''
            html += '''
                </div>
            </div>
'''

        html += '''
        </div>
    </div>
</body>
</html>
'''
        return html

    def save_html(self, output_path: str) -> None:
        """Save the timeline HTML to a file."""
        html = self.generate_html()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        logger.info(f"Timeline saved to: {output_path}")
