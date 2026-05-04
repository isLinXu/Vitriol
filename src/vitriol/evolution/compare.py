"""
Architecture Comparison Report Generator
========================================

Generate detailed comparison reports between model architectures.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .tree_builder import ArchNode

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComparisonResult:
    """Result of architecture comparison."""
    model1_id: str
    model2_id: str
    similarity_score: float  # 0-100

    # Basic parameters
    param_differences: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Architecture differences
    shared_features: List[str] = field(default_factory=list)
    unique_to_model1: List[str] = field(default_factory=list)
    unique_to_model2: List[str] = field(default_factory=list)

    # Innovation comparison
    model1_innovations: List[str] = field(default_factory=list)
    model2_innovations: List[str] = field(default_factory=list)

    # Summary
    summary: str = ""
    pros_model1: List[str] = field(default_factory=list)
    pros_model2: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model1": self.model1_id,
            "model2": self.model2_id,
            "similarity_score": self.similarity_score,
            "param_differences": self.param_differences,
            "shared_features": self.shared_features,
            "unique_to_model1": self.unique_to_model1,
            "unique_to_model2": self.unique_to_model2,
            "model1_innovations": self.model1_innovations,
            "model2_innovations": self.model2_innovations,
            "summary": self.summary,
            "pros_model1": self.pros_model1,
            "pros_model2": self.pros_model2,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Feature Definitions
# ─────────────────────────────────────────────────────────────────────────────

ATTENTION_TYPES = {
    "multi_head": "Multi-Head Attention (MHA)",
    "multi_query": "Multi-Query Attention (MQA)",
    "grouped_query": "Grouped Query Attention (GQA)",
    "multi_latent": "Multi-head Latent Attention (MLA)",
}

FFN_TYPES = {
    "standard": "Standard FFN",
    "swiglu": "SwiGLU Activation",
    "geglu": "GeGLU Activation",
    "moe": "Mixture of Experts (MoE)",
    "dense_moe": "Hybrid Dense + MoE",
}

POSITION_ENCODING = {
    "rope": "Rotary Position Embedding (RoPE)",
    "alibi": "ALiBi Position Encoding",
    "absolute": "Absolute Position Embedding",
    "relative": "Relative Position Encoding",
}


# ─────────────────────────────────────────────────────────────────────────────
# Comparator
# ─────────────────────────────────────────────────────────────────────────────

class ArchComparator:
    """
    Compare two model architectures and generate detailed reports.

    Usage:
        comparator = ArchComparator()
        result = comparator.compare(node1, node2)
        print(result.to_markdown())
    """

    def __init__(self):
        """Initialize the comparator."""
        pass

    def compare(
        self,
        model1: ArchNode,
        model2: ArchNode,
    ) -> ComparisonResult:
        """
        Compare two architecture nodes.

        Args:
            model1: First architecture node
            model2: Second architecture node

        Returns:
            ComparisonResult with detailed comparison
        """
        config1 = model1.config
        config2 = model2.config

        # Calculate similarity
        similarity = self._calculate_similarity(model1, model2)

        # Compare parameters
        param_diffs = self._compare_params(config1, config2)

        # Analyze features
        features1 = self._extract_features(config1)
        features2 = self._extract_features(config2)
        shared = features1 & features2
        unique1 = features1 - features2
        unique2 = features2 - features1

        # Innovation comparison
        innovations1 = [i.name for i in model1.innovations]
        innovations2 = [i.name for i in model2.innovations]

        # Generate summary
        summary = self._generate_summary(
            model1, model2, similarity, param_diffs,
            shared, unique1, unique2, innovations1, innovations2
        )

        # Identify pros
        pros1 = self._identify_pros(model1, model2, unique1, innovations1)
        pros2 = self._identify_pros(model2, model1, unique2, innovations2)

        return ComparisonResult(
            model1_id=model1.model_id,
            model2_id=model2.model_id,
            similarity_score=similarity * 100,
            param_differences=param_diffs,
            shared_features=sorted(shared),
            unique_to_model1=sorted(unique1),
            unique_to_model2=sorted(unique2),
            model1_innovations=innovations1,
            model2_innovations=innovations2,
            summary=summary,
            pros_model1=pros1,
            pros_model2=pros2,
        )

    def compare_params(
        self,
        config1: Dict[str, Any],
        config2: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Public API: compare key parameter differences between two configs (tests/external callers)."""
        diffs = self._compare_params(config1, config2)
        return {
            "params_match": len(diffs) == 0,
            "differences": diffs,
        }

    def compare_attention(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
        """Public API: compare attention configuration (MHA/GQA/MQA)."""
        def _attn_type(cfg: Dict[str, Any]) -> str:
            num_heads = cfg.get("num_attention_heads", 0) or 0
            num_kv = cfg.get("num_key_value_heads", 0) or 0
            if num_heads <= 0:
                return "unknown"
            if num_kv in (0, num_heads):
                return "MHA"
            if num_kv == 1:
                return "MQA"
            if 1 < num_kv < num_heads:
                return "GQA"
            return "unknown"

        return {
            "attention_type_1": _attn_type(config1),
            "attention_type_2": _attn_type(config2),
            "num_attention_heads_1": config1.get("num_attention_heads"),
            "num_key_value_heads_1": config1.get("num_key_value_heads"),
            "num_attention_heads_2": config2.get("num_attention_heads"),
            "num_key_value_heads_2": config2.get("num_key_value_heads"),
        }

    def _calculate_similarity(self, model1: ArchNode, model2: ArchNode) -> float:
        """Calculate architecture similarity score (0-1)."""
        params1 = model1.get_key_params()
        params2 = model2.get_key_params()

        score = 0.0
        total = 0

        for key in ["hidden_size", "num_hidden_layers", "num_attention_heads"]:
            v1 = params1.get(key, 0) or 0
            v2 = params2.get(key, 0) or 0
            if v1 > 0 and v2 > 0:
                ratio = min(v1, v2) / max(v1, v2)
                score += ratio
                total += 1

        if params1.get("model_type") == params2.get("model_type"):
            score += 1
        total += 1

        if params1.get("is_moe") == params2.get("is_moe"):
            score += 1
        total += 1

        attn1 = self._get_attention_type(params1)
        attn2 = self._get_attention_type(params2)
        if attn1 == attn2:
            score += 1
        total += 1

        return score / total if total > 0 else 0.0

    def _get_attention_type(self, params: Dict) -> str:
        """Determine attention type from parameters."""
        num_kv = params.get("num_key_value_heads", 0) or 0
        num_heads = params.get("num_attention_heads", 0) or 1

        if num_kv == 0:
            return "multi_head"
        elif num_kv == 1:
            return "multi_query"
        elif num_kv < num_heads:
            return "grouped_query"
        return "multi_head"

    def _compare_params(
        self,
        config1: Dict[str, Any],
        config2: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Compare key parameters between two configs."""
        keys = [
            "hidden_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "intermediate_size", "vocab_size",
            "max_position_embeddings", "rope_theta", "num_local_experts",
        ]

        differences = {}
        for key in keys:
            v1 = config1.get(key)
            v2 = config2.get(key)

            if v1 is not None and v2 is not None and v1 != v2:
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    if v2 != 0:
                        pct_diff = ((v1 - v2) / v2) * 100
                        diff_str = f"{pct_diff:+.1f}%"
                    else:
                        diff_str = f"{v1} vs {v2}"
                else:
                    diff_str = f"{v1} vs {v2}"

                differences[key] = {"model1": v1, "model2": v2, "difference": diff_str}

        return differences

    def _extract_features(self, config: Dict[str, Any]) -> set:
        """Extract architectural features from config."""
        features = set()

        num_kv = config.get("num_key_value_heads", 0) or 0
        num_heads = config.get("num_attention_heads", 0) or 1

        if num_kv == 0 or num_kv == num_heads:
            features.add("Multi-Head Attention")
        elif num_kv == 1:
            features.add("Multi-Query Attention")
        elif num_kv < num_heads:
            features.add("Grouped Query Attention")

        num_experts = config.get("num_local_experts", 0) or 0
        if num_experts > 1:
            features.add(f"Mixture of Experts ({num_experts} experts)")
            if config.get("n_routed_experts"):
                features.add("Hybrid Dense + MoE")

        if config.get("activation_function"):
            act = config.get("activation_function", "").lower()
            if "swiglu" in act:
                features.add("SwiGLU Activation")
            elif "geglu" in act:
                features.add("GeGLU Activation")

        rope_type = config.get("rope_type", "default")
        if rope_type != "default":
            features.add(f"RoPE ({rope_type})")
        else:
            features.add("RoPE")

        if config.get("tie_word_embeddings"):
            features.add("Tied Embeddings")
        if config.get("use_cache"):
            features.add("KV Cache")
        if config.get("sliding_window"):
            features.add(f"Sliding Window ({config['sliding_window']})")

        return features

    def _generate_summary(
        self,
        model1: ArchNode,
        model2: ArchNode,
        similarity: float,
        param_diffs: Dict,
        shared: set,
        unique1: set,
        unique2: set,
        innovations1: List[str],
        innovations2: List[str],
    ) -> str:
        """Generate a text summary of the comparison."""
        lines = []

        if similarity > 0.8:
            lines.append(
                f"**{model1.model_name}** and **{model2.model_name}** have highly "
                f"similar architectures ({similarity * 100:.0f}% similarity)."
            )
        elif similarity > 0.5:
            lines.append(
                f"**{model1.model_name}** and **{model2.model_name}** share "
                f"moderate architectural similarity ({similarity * 100:.0f}%)."
            )
        else:
            lines.append(
                f"**{model1.model_name}** and **{model2.model_name}** have "
                f"distinct architectures ({similarity * 100:.0f}% similarity)."
            )

        if unique1:
            lines.append(f"\n**{model1.model_name}** features: {', '.join(sorted(unique1))}")
        if unique2:
            lines.append(f"\n**{model2.model_name}** features: {', '.join(sorted(unique2))}")

        if innovations1 or innovations2:
            lines.append("\n**Architectural Innovations:**")
            if innovations1:
                lines.append(f"- {model1.model_name}: {', '.join(innovations1)}")
            if innovations2:
                lines.append(f"- {model2.model_name}: {', '.join(innovations2)}")

        return "\n".join(lines)

    def _identify_pros(
        self,
        model: ArchNode,
        other: ArchNode,
        unique_features: set,
        innovations: List[str],
    ) -> List[str]:
        """Identify advantages of a model."""
        pros = []
        params = model.get_key_params()

        if "Grouped Query Attention" in unique_features:
            pros.append("Memory-efficient attention (GQA reduces KV cache)")
        if "Sliding Window" in str(unique_features):
            pros.append("Long context efficiency (Sliding Window Attention)")
        if "Multi-head Latent Attention" in unique_features:
            pros.append("Superior memory efficiency (MLA)")

        if "Mixture of Experts" in str(unique_features):
            num_experts = params.get("num_experts", 0)
            pros.append(f"Sparse activation ({num_experts} experts) for better compute efficiency")

        other_params = other.get_key_params()
        if params.get("hidden_size", 0) > (other_params.get("hidden_size", 0) or 0) * 1.5:
            pros.append("Larger hidden dimension for better representational capacity")

        if params.get("num_hidden_layers", 0) > (other_params.get("num_hidden_layers", 0) or 0):
            pros.append("Deeper network for more complex tasks")

        for inn in innovations:
            if inn == "MLA":
                pros.append("Latest attention innovation (MLA)")
            elif inn == "MoE":
                pros.append("State-of-the-art sparse routing")

        return pros


# ─────────────────────────────────────────────────────────────────────────────
# Report Formatters
# ─────────────────────────────────────────────────────────────────────────────

class ComparisonReport:
    """Format comparison results into various output formats."""

    @staticmethod
    def to_markdown(result: ComparisonResult) -> str:
        """Format result as Markdown report."""
        m1_name = result.model1_id.split('/')[-1]
        m2_name = result.model2_id.split('/')[-1]

        lines = [
            "# Architecture Comparison Report",
            "",
            f"## {m1_name} vs {m2_name}",
            "",
            f"**Similarity Score: {result.similarity_score:.1f}%**",
            "",
            "## Parameter Differences",
            "",
            "| Parameter | Model 1 | Model 2 | Difference |",
            "|-----------|---------|---------|-----------|",
        ]

        for param, diff in result.param_differences.items():
            lines.append(
                f"| {param} | {diff['model1']} | {diff['model2']} | {diff['difference']} |"
            )

        lines.extend(["", "## Shared Features", ""])
        if result.shared_features:
            for feat in result.shared_features:
                lines.append(f"- {feat}")
        else:
            lines.append("_No shared architectural features_")

        lines.extend(["", f"## Unique to {m1_name}", ""])
        if result.unique_to_model1:
            for feat in result.unique_to_model1:
                lines.append(f"- {feat}")
        else:
            lines.append("_None_")

        lines.extend(["", f"## Unique to {m2_name}", ""])
        if result.unique_to_model2:
            for feat in result.unique_to_model2:
                lines.append(f"- {feat}")
        else:
            lines.append("_None_")

        if result.model1_innovations or result.model2_innovations:
            lines.extend(["", "## Innovations", ""])
            if result.model1_innovations:
                lines.append(f"**{m1_name}**: {', '.join(result.model1_innovations)}")
            if result.model2_innovations:
                lines.append(f"**{m2_name}**: {', '.join(result.model2_innovations)}")

        if result.pros_model1 or result.pros_model2:
            lines.extend(["", "## Analysis", ""])
            if result.pros_model1:
                lines.append(f"### {m1_name} Advantages")
                for pro in result.pros_model1:
                    lines.append(f"- {pro}")
            if result.pros_model2:
                lines.append(f"### {m2_name} Advantages")
                for pro in result.pros_model2:
                    lines.append(f"- {pro}")

        if result.summary:
            lines.extend(["", "## Summary", "", result.summary])

        return "\n".join(lines)

    @staticmethod
    def to_json(result: ComparisonResult, indent: int = 2) -> str:
        """Format result as JSON string."""
        return json.dumps(result.to_dict(), indent=indent, ensure_ascii=False)

    @staticmethod
    def to_html(result: ComparisonResult) -> str:
        """Format result as HTML report."""
        m1_name = result.model1_id.split('/')[-1]
        m2_name = result.model2_id.split('/')[-1]

        params_rows = "".join(
            f"<tr><td>{k}</td><td>{v['model1']}</td><td>{v['model2']}</td></tr>"
            for k, v in result.param_differences.items()
        )

        shared_features = "".join(
            f'<span class="feature">{f}</span>' for f in result.shared_features
        ) if result.shared_features else "<p>None</p>"

        unique1 = "".join(
            f'<span class="feature">{f}</span>' for f in result.unique_to_model1
        ) if result.unique_to_model1 else "<p>None</p>"

        unique2 = "".join(
            f'<span class="feature">{f}</span>' for f in result.unique_to_model2
        ) if result.unique_to_model2 else "<p>None</p>"

        pros1 = "".join(f"<li>{p}</li>" for p in result.pros_model1)
        pros2 = "".join(f"<li>{p}</li>" for p in result.pros_model2)

        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Architecture Comparison: {m1_name} vs {m2_name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #1a1a2e; border-bottom: 2px solid #7c3aed; padding-bottom: 10px; }}
        h2 {{ color: #7c3aed; margin-top: 30px; }}
        .score {{ background: linear-gradient(90deg, #7c3aed, #00d4ff); color: white; padding: 10px 20px; border-radius: 8px; display: inline-block; font-size: 18px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ border: 1px solid #e0e0e0; padding: 10px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        .pros {{ background: #f0fdf4; border-left: 4px solid #22c55e; padding: 15px; margin: 10px 0; }}
        .feature {{ display: inline-block; background: #e0e0e0; padding: 4px 10px; border-radius: 15px; margin: 3px; font-size: 12px; }}
    </style>
</head>
<body>
    <h1>Architecture Comparison</h1>
    <h2>{m1_name} vs {m2_name}</h2>
    <p>Similarity Score: <span class="score">{result.similarity_score:.1f}%</span></p>

    <h2>Parameter Comparison</h2>
    <table>
        <tr><th>Parameter</th><th>{m1_name}</th><th>{m2_name}</th></tr>
        {params_rows}
    </table>

    <h2>Features</h2>
    <h3>Shared</h3>
    {shared_features}

    <h3>Unique to {m1_name}</h3>
    {unique1}

    <h3>Unique to {m2_name}</h3>
    {unique2}

    <h2>Analysis</h2>
    <div class="pros">
        <strong>{m1_name} Advantages:</strong>
        <ul>{pros1}</ul>
    </div>
    <div class="pros">
        <strong>{m2_name} Advantages:</strong>
        <ul>{pros2}</ul>
    </div>

    <h2>Summary</h2>
    <p>{result.summary}</p>
</body>
</html>
"""


__all__ = [
    "ArchComparator",
    "ComparisonResult",
    "ComparisonReport",
]
