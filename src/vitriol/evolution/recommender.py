"""
Architecture Recommender
========================

Recommends suitable LLM architectures based on user requirements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .tree_builder import EvolutionTree, ArchNode

logger = logging.getLogger(__name__)


class UseCase(str, Enum):
    """Common use cases for LLM architectures."""
    CHAT = "chat"
    CODE = "code"
    EMBEDDING = "embedding"
    LONG_CONTEXT = "long_context"
    MULTIMODAL = "multimodal"
    GENERAL = "general"


@dataclass
class RecommendationCriteria:
    """User-defined criteria for architecture recommendation."""
    max_params: Optional[float] = None
    max_vram: Optional[float] = None
    min_params: Optional[float] = None
    use_case: UseCase = UseCase.GENERAL
    prefer_moe: bool = False
    prefer_small: bool = False
    require_gqa: bool = False
    require_long_context: bool = False
    preferred_families: Optional[List[str]] = None
    excluded_families: Optional[List[str]] = None


@dataclass
class ArchitectureRecommendation:
    """A single architecture recommendation."""
    model_id: str
    family: str
    params_b: float
    vram_gb: float
    score: float
    match_reasons: List[str] = field(default_factory=list)
    innovations: List[str] = field(default_factory=list)


class ArchitectureRecommender:
    """
    Recommends suitable LLM architectures based on user requirements.
    """

    VRAM_COEFFICIENT = 2.0  # GB per billion parameters (bfloat16)

    def __init__(self, evolution_tree: Optional[EvolutionTree] = None):
        self.tree = evolution_tree or EvolutionTree()
        self.tree.load_builtin_families()
        self.tree.build()
        self._build_cache()

    def _build_cache(self) -> None:
        """Build internal caches."""
        self._arch_cache: Dict[str, Dict[str, Any]] = {}

        for node_id, node in self.tree.nodes.items():
            self._arch_cache[node_id] = {
                "params": self._estimate_params(node),
                "vram": self._estimate_vram(node),
                "family": node.family,
                "is_moe": node.config.get("num_local_experts", 0) > 1,
                "has_gqa": (node.config.get("num_key_value_heads", 0) <
                           node.config.get("num_attention_heads", 0)),
                "context_length": node.config.get("max_position_embeddings", 4096),
                "innovations": [i.name for i in node.innovations],
                "node": node,
            }

    def _estimate_params(self, node: ArchNode) -> float:
        """Estimate parameter count in billions."""
        hidden_size = node.config.get("hidden_size", 4096)
        num_layers = node.config.get("num_hidden_layers", 32)
        vocab_size = node.config.get("vocab_size", 32000)
        intermediate_size = node.config.get("intermediate_size", 11008)

        embedding_params = vocab_size * hidden_size
        attention_params = 4 * hidden_size * hidden_size * num_layers
        ffn_params = 2 * hidden_size * intermediate_size * num_layers

        num_experts = node.config.get("num_local_experts", 1)
        if num_experts > 1:
            ffn_params *= num_experts

        total = embedding_params + attention_params + ffn_params
        return total / 1e9

    def _estimate_vram(self, node: ArchNode) -> float:
        """Estimate VRAM usage in GB."""
        return self._estimate_params(node) * self.VRAM_COEFFICIENT

    def recommend(
        self,
        max_params: Optional[float] = None,
        max_vram: Optional[float] = None,
        use_case: UseCase = UseCase.GENERAL,
        prefer_moe: bool = False,
        require_gqa: bool = False,
        require_long_context: bool = False,
        preferred_families: Optional[List[str]] = None,
    ) -> List[ArchitectureRecommendation]:
        """
        Recommend architectures based on criteria.

        Args:
            max_params: Maximum parameters in billions
            max_vram: Maximum VRAM in GB
            use_case: chat, code, embedding, long_context, multimodal, general
            prefer_moe: Prefer MoE architectures
            require_gqa: Require GQA support
            require_long_context: Require 128K+ context
            preferred_families: List of preferred families

        Returns:
            List of ranked recommendations
        """
        candidates: List[Tuple[str, float, Dict[str, Any]]] = []

        for model_id, arch_info in self._arch_cache.items():
            score, reasons = self._score_architecture(model_id, arch_info, max_params,
                                                       max_vram, use_case, prefer_moe,
                                                       require_gqa, require_long_context,
                                                       preferred_families)

            if score > 0:
                candidates.append((model_id, score, {**arch_info, "reasons": reasons}))

        candidates.sort(key=lambda x: x[1], reverse=True)

        recommendations = []
        for model_id, score, info in candidates[:10]:
            node = info["node"]
            recommendations.append(ArchitectureRecommendation(
                model_id=model_id,
                family=node.family,
                params_b=info["params"],
                vram_gb=info["vram"],
                score=score,
                match_reasons=info["reasons"],
                innovations=info["innovations"],
            ))

        return recommendations

    def _score_architecture(
        self,
        model_id: str,
        arch_info: Dict[str, Any],
        max_params: Optional[float],
        max_vram: Optional[float],
        use_case: UseCase,
        prefer_moe: bool,
        require_gqa: bool,
        require_long_context: bool,
        preferred_families: Optional[List[str]],
    ) -> Tuple[float, List[str]]:
        """Score an architecture."""
        score = 1.0
        reasons = []
        params = arch_info["params"]
        vram = arch_info["vram"]

        # Hard constraints
        if max_params and params > max_params:
            return 0, []
        if max_vram and vram > max_vram:
            return 0, []

        # Parameter scoring
        if max_params:
            if params <= max_params * 0.7:
                score *= 1.2
                reasons.append(f"✓ Efficient params ({params:.1f}B vs {max_params}B limit)")
            elif params <= max_params:
                reasons.append(f"✓ Within budget ({params:.1f}B)")

        # VRAM scoring
        if max_vram:
            if vram <= max_vram * 0.8:
                score *= 1.2
                reasons.append(f"✓ Low VRAM ({vram:.1f}GB)")

        # MoE preference
        if prefer_moe:
            if arch_info["is_moe"]:
                score *= 1.3
                reasons.append("✓ MoE architecture")
            else:
                score *= 0.8

        # GQA requirement
        if require_gqa:
            if arch_info["has_gqa"]:
                score *= 1.2
                reasons.append("✓ GQA support")
            else:
                score *= 0.7

        # Long context requirement
        if require_long_context:
            ctx = arch_info["context_length"]
            if ctx >= 128000:
                score *= 1.5
                reasons.append(f"✓ 128K+ context ({ctx:,})")
            elif ctx >= 32000:
                score *= 1.2
                reasons.append(f"✓ 32K+ context ({ctx:,})")
            else:
                score *= 0.6

        # Use case specific
        if use_case == UseCase.CODE:
            if "FIM" in arch_info["innovations"] or "Code" in str(arch_info["innovations"]):
                score *= 1.3
                reasons.append("✓ Good for code")

        # Family preference
        if preferred_families and arch_info["family"] in preferred_families:
            score *= 1.3
            reasons.append(f"✓ {arch_info['family']} family")

        return score, reasons

    def get_family_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all families."""
        summary = {}
        for family_name, family_data in self.tree.families.items():
            nodes = [n for n in self.tree.nodes.values() if n.family == family_name]
            if nodes:
                params_list = [self._arch_cache.get(n.model_id, {}).get("params", 0) for n in nodes]
                summary[family_name] = {
                    "members": len(nodes),
                    "min_params": min(params_list) if params_list else 0,
                    "max_params": max(params_list) if params_list else 0,
                    "has_moe": any(self._arch_cache.get(n.model_id, {}).get("is_moe") for n in nodes),
                    "has_gqa": any(self._arch_cache.get(n.model_id, {}).get("has_gqa") for n in nodes),
                }
        return summary
