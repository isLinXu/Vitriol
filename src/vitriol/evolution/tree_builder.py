"""
Architecture Node and Evolution Tree Builder
===========================================

Build model family trees based on architecture similarities and known relationships.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..utils.hf_loading import load_config as hf_load_config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ArchInnovation:
    """Represents an architectural innovation at a specific node."""
    name: str  # e.g., "GQA", "MLA", "SwiGLU", "MoE"
    description: str
    introduced_in: str  # Model that first introduced this
    year: int


@dataclass
class ArchNode:
    """
    A node in the architecture evolution tree.

    Represents a model architecture with its configuration and
    relationship to other architectures.
    """
    model_id: str                    # HuggingFace model ID
    config: Dict[str, Any]           # Model configuration
    parent: Optional[str] = None      # Parent model ID (if known)
    children: List[str] = field(default_factory=list)
    innovations: List[ArchInnovation] = field(default_factory=list)
    similarity_score: float = 1.0    # Similarity to parent (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def model_name(self) -> str:
        """Extract clean model name from model_id."""
        return self.model_id.split("/")[-1] if "/" in self.model_id else self.model_id

    @property
    def family(self) -> str:
        """Extract model family (e.g., 'Qwen', 'LLaMA', 'DeepSeek')."""
        # Prefer canonical families over org/user names (e.g. "deepseek-ai" -> "DeepSeek").
        if "/" in self.model_id:
            org = self.model_id.split("/")[0]
            org_lower = org.lower()
            model_id_lower = self.model_id.lower()

            org_family_map = {
                # DeepSeek
                "deepseek-ai": "DeepSeek",
                "deepseek": "DeepSeek",
                # LLaMA / Mistral orgs
                "meta-llama": "LLaMA",
                "mistralai": "LLaMA",
                # Qwen
                "qwen": "Qwen",
                # GLM family often lives under multiple orgs; keep keyword fallback below
            }
            if org_lower in org_family_map:
                return org_family_map[org_lower]

            # Keyword-based fallback using full model_id (covers "org/model" patterns)
            if "qwen" in model_id_lower:
                return "Qwen"
            if "llama" in model_id_lower or "mistral" in model_id_lower:
                return "LLaMA"
            if "deepseek" in model_id_lower:
                return "DeepSeek"
            if "gpt" in model_id_lower:
                return "GPT"
            if "glm" in model_id_lower:
                return "GLM"
            if "kimi" in model_id_lower or "moonshot" in model_id_lower:
                return "Kimi"

            # Unknown org: keep the org string as a "family-like" label.
            return org
        # Infer from model name patterns
        name = self.model_name.lower()
        if "qwen" in name:
            return "Qwen"
        elif "llama" in name or "mistral" in name:
            return "LLaMA"
        elif "deepseek" in name:
            return "DeepSeek"
        elif "gpt" in name:
            return "GPT"
        elif "glm" in name:
            return "GLM"
        elif "kimi" in name or "moonshot" in name:
            return "Kimi"
        return "Other"

    def get_key_params(self) -> Dict[str, Any]:
        """Extract key architecture parameters for comparison."""
        return {
            "hidden_size": self.config.get("hidden_size"),
            "num_hidden_layers": self.config.get("num_hidden_layers"),
            "num_attention_heads": self.config.get("num_attention_heads"),
            "num_key_value_heads": self.config.get("num_key_value_heads"),
            "intermediate_size": self.config.get("intermediate_size"),
            "vocab_size": self.config.get("vocab_size"),
            "max_position_embeddings": self.config.get("max_position_embeddings"),
            "model_type": self.config.get("model_type"),
            "rope_type": self.config.get("rope_type", "default"),
            "is_moe": self.config.get("num_local_experts", 0) > 1,
            "num_experts": self.config.get("num_local_experts", 1),
        }


@dataclass
class ArchitectureMetrics:
    """Computed architecture metrics for a node."""
    total_params: int
    trainable_params: int
    flops_per_token: float
    vram_estimate_gb: float  # Estimated VRAM usage in GB
    inference_latency_ms: float  # Estimated latency per token (ms)
    memory_bandwidth_gbs: float  # GB/s required

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
            "flops_per_token": self.flops_per_token,
            "vram_estimate_gb": self.vram_estimate_gb,
            "inference_latency_ms": self.inference_latency_ms,
            "memory_bandwidth_gbs": self.memory_bandwidth_gbs,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Built-in Model Family Relationships
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Fallback Architecture Parameters
# ─────────────────────────────────────────────────────────────────────────────
# For models that cannot be fetched from HuggingFace (gated, removed, or
# proprietary), we embed known architecture parameters here. These are sourced
# from official publications, blog posts, and model cards.

FALLBACK_PARAMS: Dict[str, Dict[str, Any]] = {
    # ── LLaMA ────────────────────────────────────────────────
    "meta-llama/Llama-2-7b": {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 32, "intermediate_size": 11008, "vocab_size": 32000, "max_position_embeddings": 4096, "model_type": "llama"},
    "meta-llama/Llama-2-13b": {"hidden_size": 5120, "num_hidden_layers": 40, "num_attention_heads": 40, "num_key_value_heads": 40, "intermediate_size": 13824, "vocab_size": 32000, "max_position_embeddings": 4096, "model_type": "llama"},
    "meta-llama/Llama-2-70b": {"hidden_size": 8192, "num_hidden_layers": 80, "num_attention_heads": 64, "num_key_value_heads": 8, "intermediate_size": 28672, "vocab_size": 32000, "max_position_embeddings": 4096, "model_type": "llama"},
    "meta-llama/Llama-3-8b": {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "intermediate_size": 14336, "vocab_size": 128256, "max_position_embeddings": 8192, "model_type": "llama"},
    "meta-llama/Llama-3.1-8B": {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "intermediate_size": 14336, "vocab_size": 128256, "max_position_embeddings": 131072, "model_type": "llama"},
    "meta-llama/Llama-3.2-8B": {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "intermediate_size": 14336, "vocab_size": 128256, "max_position_embeddings": 131072, "model_type": "llama"},
    "meta-llama/Llama-3.3-70B": {"hidden_size": 8192, "num_hidden_layers": 80, "num_attention_heads": 64, "num_key_value_heads": 8, "intermediate_size": 28672, "vocab_size": 128256, "max_position_embeddings": 131072, "model_type": "llama"},
    "meta-llama/Llama-3.2-90B": {"hidden_size": 8192, "num_hidden_layers": 80, "num_attention_heads": 64, "num_key_value_heads": 8, "intermediate_size": 28672, "vocab_size": 128256, "max_position_embeddings": 131072, "model_type": "llama"},
    "meta-llama/Mistral-7B-v0.1": {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "intermediate_size": 14336, "vocab_size": 32000, "max_position_embeddings": 32768, "model_type": "mistral"},
    # ── Qwen (gated or not fetched) ──────────────────────────
    "Qwen/Qwen3-72B": {"hidden_size": 8192, "num_hidden_layers": 80, "num_attention_heads": 64, "num_key_value_heads": 8, "intermediate_size": 29568, "vocab_size": 152064, "max_position_embeddings": 131072, "model_type": "qwen2"},
    # ── DeepSeek ─────────────────────────────────────────────
    "deepseek-ai/deepseek-7b": {"hidden_size": 4096, "num_hidden_layers": 30, "num_attention_heads": 32, "num_key_value_heads": 32, "intermediate_size": 11008, "vocab_size": 102400, "max_position_embeddings": 4096, "model_type": "llama"},
    "deepseek-ai/DeepSeek-LLM-7B": {"hidden_size": 4096, "num_hidden_layers": 30, "num_attention_heads": 32, "num_key_value_heads": 32, "intermediate_size": 11008, "vocab_size": 102400, "max_position_embeddings": 4096, "model_type": "llama"},
    "deepseek-ai/DeepSeek-LLM-67B": {"hidden_size": 8192, "num_hidden_layers": 95, "num_attention_heads": 64, "num_key_value_heads": 64, "intermediate_size": 22016, "vocab_size": 102400, "max_position_embeddings": 4096, "model_type": "llama"},
    "deepseek-ai/deepseek-moe-16b": {"hidden_size": 2048, "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 16, "intermediate_size": 10944, "vocab_size": 102400, "max_position_embeddings": 4096, "model_type": "deepseek", "num_local_experts": 64},
    "deepseek-ai/DeepSeek-Coder-V2": {"hidden_size": 5120, "num_hidden_layers": 60, "num_attention_heads": 128, "num_key_value_heads": 128, "intermediate_size": 12288, "vocab_size": 102400, "max_position_embeddings": 163840, "model_type": "deepseek_v2"},
    # ── Kimi / Moonshot (proprietary, approximate) ───────────
    "moonshotai/kimi-7b": {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 32, "intermediate_size": 11008, "vocab_size": 64000, "max_position_embeddings": 131072, "model_type": "kimi"},
    "moonshotai/kimi-20b": {"hidden_size": 5120, "num_hidden_layers": 40, "num_attention_heads": 40, "num_key_value_heads": 40, "intermediate_size": 13824, "vocab_size": 64000, "max_position_embeddings": 131072, "model_type": "kimi"},
    "moonshotai/kimi-v1": {"hidden_size": 8192, "num_hidden_layers": 64, "num_attention_heads": 64, "num_key_value_heads": 8, "intermediate_size": 22016, "vocab_size": 64000, "max_position_embeddings": 1048576, "model_type": "kimi"},
    "moonshotai/kimi-v1-32k": {"hidden_size": 8192, "num_hidden_layers": 64, "num_attention_heads": 64, "num_key_value_heads": 8, "intermediate_size": 22016, "vocab_size": 64000, "max_position_embeddings": 32768, "model_type": "kimi"},
    # ── Phi (Microsoft) ──────────────────────────────────────
    "microsoft/Phi-3-mini": {"hidden_size": 3072, "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 32, "intermediate_size": 8192, "vocab_size": 32064, "max_position_embeddings": 131072, "model_type": "phi3"},
    "microsoft/Phi-3-small": {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "intermediate_size": 14336, "vocab_size": 100352, "max_position_embeddings": 131072, "model_type": "phi3"},
    "microsoft/Phi-3-medium": {"hidden_size": 5120, "num_hidden_layers": 40, "num_attention_heads": 40, "num_key_value_heads": 10, "intermediate_size": 17920, "vocab_size": 32064, "max_position_embeddings": 131072, "model_type": "phi3"},
    "microsoft/Phi-3-small-128k": {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "intermediate_size": 14336, "vocab_size": 100352, "max_position_embeddings": 131072, "model_type": "phi3"},
    # ── Mistral ──────────────────────────────────────────────
    "mistralai/Mistral-7B-v0.3-24B": {"hidden_size": 5120, "num_hidden_layers": 40, "num_attention_heads": 32, "num_key_value_heads": 8, "intermediate_size": 14336, "vocab_size": 32768, "max_position_embeddings": 32768, "model_type": "mistral"},
    "mistralai/Mistral-Nemo-Instruct-12B": {"hidden_size": 5120, "num_hidden_layers": 40, "num_attention_heads": 32, "num_key_value_heads": 8, "intermediate_size": 14336, "vocab_size": 131072, "max_position_embeddings": 131072, "model_type": "mistral"},
    # ── Mamba (SSM — not Transformer, params are approximate) ─
    "state-spaces/mamba-2.8b": {"hidden_size": 2560, "num_hidden_layers": 64, "num_attention_heads": 1, "vocab_size": 50280, "model_type": "mamba"},
    "state-spaces/mamba-2.8b-slimp": {"hidden_size": 2560, "num_hidden_layers": 64, "num_attention_heads": 1, "vocab_size": 50280, "model_type": "mamba"},
    "state-spaces/mamba-1.4b": {"hidden_size": 2048, "num_hidden_layers": 48, "num_attention_heads": 1, "vocab_size": 50280, "model_type": "mamba"},
    "state-spaces/mamba-1.8b": {"hidden_size": 2048, "num_hidden_layers": 48, "num_attention_heads": 1, "vocab_size": 50280, "model_type": "mamba"},
    "state-spaces/mamba-670m": {"hidden_size": 1024, "num_hidden_layers": 48, "num_attention_heads": 1, "vocab_size": 50280, "model_type": "mamba"},
    # ── RWKV ─────────────────────────────────────────────────
    "RWKV/rwkv-4-1b5": {"hidden_size": 2048, "num_hidden_layers": 24, "num_attention_heads": 1, "vocab_size": 50277, "model_type": "rwkv"},
    "RWKV/rwkv-4-2b5": {"hidden_size": 2560, "num_hidden_layers": 32, "num_attention_heads": 1, "vocab_size": 50277, "model_type": "rwkv"},
    "RWKV/rwkv-4-3b": {"hidden_size": 2560, "num_hidden_layers": 32, "num_attention_heads": 1, "vocab_size": 50277, "model_type": "rwkv"},
    "RWKV/rwkv-4-7b": {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 1, "vocab_size": 50277, "model_type": "rwkv"},
    "RWKV/rwkv-5-3b": {"hidden_size": 2560, "num_hidden_layers": 32, "num_attention_heads": 1, "vocab_size": 65536, "model_type": "rwkv"},
    "RWKV/rwkv-5-7b": {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 1, "vocab_size": 65536, "model_type": "rwkv"},
    "RWKV/rwkv-6-13b": {"hidden_size": 5120, "num_hidden_layers": 40, "num_attention_heads": 1, "vocab_size": 65536, "model_type": "rwkv"},
    # ── Falcon ───────────────────────────────────────────────
    "tiiuae/falcon-instruct-7b": {"hidden_size": 4544, "num_hidden_layers": 32, "num_attention_heads": 71, "vocab_size": 65024, "model_type": "falcon"},
    "tiiuae/falcon-instruct-40b": {"hidden_size": 8192, "num_hidden_layers": 60, "num_attention_heads": 128, "vocab_size": 65024, "model_type": "falcon"},
    # ── Starcoder ────────────────────────────────────────────
    "bigcode/starcoderbase": {"hidden_size": 6144, "num_hidden_layers": 40, "num_attention_heads": 48, "intermediate_size": 24576, "vocab_size": 49152, "max_position_embeddings": 8192, "model_type": "gpt_bigcode"},
    "bigcode/starcoderbase-15b": {"hidden_size": 6144, "num_hidden_layers": 40, "num_attention_heads": 48, "intermediate_size": 24576, "vocab_size": 49152, "max_position_embeddings": 8192, "model_type": "gpt_bigcode"},
    "bigcode/starcoder-15b": {"hidden_size": 6144, "num_hidden_layers": 40, "num_attention_heads": 48, "intermediate_size": 24576, "vocab_size": 49152, "max_position_embeddings": 8192, "model_type": "gpt_bigcode"},
    "bigcode/starcoder2-1b": {"hidden_size": 2048, "num_hidden_layers": 24, "num_attention_heads": 16, "num_key_value_heads": 4, "intermediate_size": 8192, "vocab_size": 49152, "max_position_embeddings": 16384, "model_type": "starcoder2"},
    # ── BLOOM ────────────────────────────────────────────────
    "bigscience/bloom-176b": {"hidden_size": 14336, "num_hidden_layers": 70, "num_attention_heads": 112, "vocab_size": 250880, "model_type": "bloom"},
    # ── GLM ──────────────────────────────────────────────────
    "THUDM/GLM-4-70B": {"hidden_size": 8192, "num_hidden_layers": 80, "num_attention_heads": 64, "num_key_value_heads": 8, "intermediate_size": 24576, "vocab_size": 151552, "model_type": "chatglm"},
    "THUDM/GLM-4-32B": {"hidden_size": 6144, "num_hidden_layers": 48, "num_attention_heads": 48, "num_key_value_heads": 8, "intermediate_size": 16384, "vocab_size": 151552, "model_type": "chatglm"},
    # ── Yi ───────────────────────────────────────────────────
    "01-ai/Yi-1.5-70B": {"hidden_size": 8192, "num_hidden_layers": 64, "num_attention_heads": 64, "num_key_value_heads": 8, "intermediate_size": 28672, "vocab_size": 64000, "max_position_embeddings": 4096, "model_type": "llama"},
    # ── InternLM ─────────────────────────────────────────────
    "internlm/internlm2.5-7b": {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "intermediate_size": 14336, "vocab_size": 92544, "max_position_embeddings": 32768, "model_type": "internlm2"},
    "internlm/internlm2.5-20b": {"hidden_size": 6144, "num_hidden_layers": 48, "num_attention_heads": 48, "num_key_value_heads": 8, "intermediate_size": 16384, "vocab_size": 92544, "max_position_embeddings": 32768, "model_type": "internlm2"},
    "internlm/internlm3-7b": {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "intermediate_size": 14336, "vocab_size": 92544, "max_position_embeddings": 32768, "model_type": "internlm2"},
    "internlm/internlm3-20b": {"hidden_size": 6144, "num_hidden_layers": 48, "num_attention_heads": 48, "num_key_value_heads": 8, "intermediate_size": 16384, "vocab_size": 92544, "max_position_embeddings": 32768, "model_type": "internlm2"},
    # ── BaiChuan ─────────────────────────────────────────────
    "baichuan-inc/Baichuan2-7B": {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32, "intermediate_size": 11008, "vocab_size": 125696, "max_position_embeddings": 4096, "model_type": "baichuan"},
    "baichuan-inc/Baichuan2-13B": {"hidden_size": 5120, "num_hidden_layers": 40, "num_attention_heads": 40, "intermediate_size": 13696, "vocab_size": 125696, "model_type": "baichuan"},
    "baichuan-inc/Baichuan2-53B": {"hidden_size": 8192, "num_hidden_layers": 64, "num_attention_heads": 64, "intermediate_size": 22016, "vocab_size": 125696, "model_type": "baichuan"},
    "baichuan-inc/Baichuan2-53B-chat": {"hidden_size": 8192, "num_hidden_layers": 64, "num_attention_heads": 64, "intermediate_size": 22016, "vocab_size": 125696, "model_type": "baichuan"},
    # ── Skywork ──────────────────────────────────────────────
    "Skywork/Skywork-13B": {"hidden_size": 4608, "num_hidden_layers": 52, "num_attention_heads": 36, "intermediate_size": 12288, "vocab_size": 65536, "max_position_embeddings": 4096, "model_type": "llama"},
    "Skywork/Skywork-13B-chat": {"hidden_size": 4608, "num_hidden_layers": 52, "num_attention_heads": 36, "intermediate_size": 12288, "vocab_size": 65536, "max_position_embeddings": 4096, "model_type": "llama"},
    "Skywork/Skywork-MoE-13B": {"hidden_size": 4608, "num_hidden_layers": 52, "num_attention_heads": 36, "intermediate_size": 12288, "vocab_size": 65536, "max_position_embeddings": 4096, "model_type": "llama"},
    "Skywork/Skywork-13B-3k": {"hidden_size": 4608, "num_hidden_layers": 52, "num_attention_heads": 36, "intermediate_size": 12288, "vocab_size": 65536, "max_position_embeddings": 3072, "model_type": "llama"},
    "Skywork/Skywork-13B-16K": {"hidden_size": 4608, "num_hidden_layers": 52, "num_attention_heads": 36, "intermediate_size": 12288, "vocab_size": 65536, "max_position_embeddings": 16384, "model_type": "llama"},
}


DEFAULT_FAMILIES = {
    "LLaMA": {
        "root": "meta-llama/Llama-2-7b",
        "members": {
            "meta-llama/Llama-2-7b": {
                "children": ["meta-llama/Llama-2-13b", "meta-llama/Llama-2-70b"],
            },
            "meta-llama/Llama-2-13b": {
                "children": ["meta-llama/Llama-3-8b", "meta-llama/Mistral-7B-v0.1"],
            },
            "meta-llama/Llama-3-8b": {
                "children": ["meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.2-8B"],
            },
            "meta-llama/Llama-3.1-8B": {
                "children": ["meta-llama/Llama-3.3-70B"],
            },
            "meta-llama/Llama-3.2-8B": {
                "children": ["meta-llama/Llama-3.2-90B"],
            },
        },
        "innovations": {
            "meta-llama/Llama-2-7b": [
                ArchInnovation("RoPE", "Rotary Position Embedding", "LLaMA-2", 2023),
                ArchInnovation("SwiGLU", "SwiGLU Activation", "LLaMA-2", 2023),
            ],
            "meta-llama/Llama-3-8b": [
                ArchInnovation("GQA", "Grouped Query Attention", "LLaMA-3", 2024),
                ArchInnovation("128K Context", "Extended Context Length", "LLaMA-3", 2024),
            ],
            "meta-llama/Llama-3.1-8B": [
                ArchInnovation("Long Context", "128K Extended Context", "LLaMA-3.1", 2024),
            ],
        },
    },
    "Qwen": {
        "root": "Qwen/Qwen-7B",
        "members": {
            "Qwen/Qwen-7B": {
                "children": ["Qwen/Qwen-14B", "Qwen/Qwen1.5-0.5B", "Qwen/Qwen1.5-1.8B"],
            },
            "Qwen/Qwen-14B": {
                "children": ["Qwen/Qwen1.5-7B", "Qwen/Qwen1.5-14B"],
            },
            "Qwen/Qwen1.5-7B": {
                "children": ["Qwen/Qwen2-7B", "Qwen/Qwen2.5-7B"],
            },
            "Qwen/Qwen2-7B": {
                "children": ["Qwen/Qwen2.5-72B", "Qwen/Qwen2.5-32B"],
            },
            "Qwen/Qwen2.5-7B": {
                "children": ["Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-1.5B"],
            },
            "Qwen/Qwen2.5-72B": {
                "children": ["Qwen/Qwen3-72B"],
            },
        },
        "innovations": {
            "Qwen/Qwen-7B": [
                ArchInnovation("RoPE", "Rotary Position Embedding", "Qwen-7B", 2023),
            ],
            "Qwen/Qwen1.5-7B": [
                ArchInnovation("GQA", "Grouped Query Attention", "Qwen1.5", 2024),
            ],
            "Qwen/Qwen2-7B": [
                ArchInnovation("GQA", "Grouped Query Attention", "Qwen2", 2024),
                ArchInnovation("BF16", "BFloat16 Support", "Qwen2", 2024),
            ],
            "Qwen/Qwen2.5-72B": [
                ArchInnovation("MoE", "Mixture of Experts (optional)", "Qwen2.5", 2024),
                ArchInnovation("Long Context", "128K Context", "Qwen2.5", 2024),
            ],
        },
    },
    "DeepSeek": {
        "root": "deepseek-ai/deepseek-7b",
        "members": {
            "deepseek-ai/deepseek-7b": {
                "children": ["deepseek-ai/DeepSeek-LLM-7B", "deepseek-ai/deepseek-moe-16b"],
            },
            "deepseek-ai/DeepSeek-LLM-7B": {
                "children": ["deepseek-ai/DeepSeek-LLM-67B"],
            },
            "deepseek-ai/deepseek-moe-16b": {
                "children": ["deepseek-ai/DeepSeek-V2", "deepseek-ai/DeepSeek-V2.5"],
            },
            "deepseek-ai/DeepSeek-V2": {
                "children": ["deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-Coder-V2"],
            },
            "deepseek-ai/DeepSeek-V2.5": {
                "children": ["deepseek-ai/DeepSeek-V3"],
            },
        },
        "innovations": {
            "deepseek-ai/deepseek-moe-16b": [
                ArchInnovation("MoE", "Mixture of Experts", "DeepSeek-MoE", 2024),
                ArchInnovation("Fine-grained Expert", "Fine-grained Expert Partitioning", "DeepSeek-MoE", 2024),
            ],
            "deepseek-ai/DeepSeek-V2": [
                ArchInnovation("MLA", "Multi-head Latent Attention", "DeepSeek-V2", 2024),
                ArchInnovation("DeepSeek MoE", "Custom MoE Architecture", "DeepSeek-V2", 2024),
            ],
            "deepseek-ai/DeepSeek-V2.5": [
                ArchInnovation("VL", "Vision-Language Integration", "DeepSeek-V2.5", 2024),
            ],
            "deepseek-ai/DeepSeek-V3": [
                ArchInnovation("Hybrid MoE", "Hybrid Dense + MoE", "DeepSeek-V3", 2024),
                ArchInnovation("Multi-Token Prediction", "MTP Auxiliary Loss", "DeepSeek-V3", 2024),
                ArchInnovation("FP8 Training", "FP8 Mixed Precision Training", "DeepSeek-V3", 2024),
            ],
        },
    },
    "Mistral": {
        "root": "mistralai/Mistral-7B-v0.1",
        "members": {
            "mistralai/Mistral-7B-v0.1": {
                "children": ["mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mistral-7B-v0.3"],
            },
            "mistralai/Mistral-7B-Instruct-v0.2": {
                "children": ["mistralai/Mistral-Nemo-Instruct-12B", "mistralai/Mistral-7B-Instruct-v0.3"],
            },
            "mistralai/Mistral-7B-v0.3": {
                "children": ["mistralai/Mistral-7B-v0.3-24B"],
            },
        },
        "innovations": {
            "mistralai/Mistral-7B-v0.1": [
                ArchInnovation("Sliding Window", "Sliding Window Attention", "Mistral-7B", 2023),
                ArchInnovation("RoPE", "Rotary Position Embedding", "Mistral-7B", 2023),
            ],
            "mistralai/Mistral-7B-Instruct-v0.2": [
                ArchInnovation("Rope Scaling", "Extended Context via RoPE Scaling", "Mistral-Instruct", 2023),
            ],
            "mistralai/Mistral-Nemo-Instruct-12B": [
                ArchInnovation("Mistral Small", "Efficient Mixed-Expert", "Mistral-Nemo", 2024),
            ],
        },
    },
    # Added: GLM family
    "GLM": {
        "root": "THUDM/chatglm-6b",
        "members": {
            "THUDM/chatglm-6b": {
                "children": ["THUDM/chatglm2-6b", "THUDM/chatglm3-6b"],
            },
            "THUDM/chatglm2-6b": {
                "children": ["THUDM/chatglm3-6b", "THUDM/glm-10b"],
            },
            "THUDM/chatglm3-6b": {
                "children": ["THUDM/GLM-4-9B", "THUDM/GLM-4V-9B"],
            },
            "THUDM/GLM-4-9B": {
                "children": ["THUDM/GLM-4-70B", "THUDM/GLM-4-32B"],
            },
        },
        "innovations": {
            "THUDM/chatglm-6b": [
                ArchInnovation("GLM Embedding", "General Language Model Pretraining", "ChatGLM", 2023),
            ],
            "THUDM/chatglm2-6b": [
                ArchInnovation("Multi-Query Attention", "Multi-Query Attention", "ChatGLM2", 2023),
                ArchInnovation("Long Context", "32K Context", "ChatGLM2", 2023),
            ],
            "THUDM/chatglm3-6b": [
                ArchInnovation("GQA", "Grouped Query Attention", "ChatGLM3", 2023),
                ArchInnovation("Self-Extension", "Extended Context 128K", "ChatGLM3", 2023),
            ],
            "THUDM/GLM-4-9B": [
                ArchInnovation("GLM-4", "Full GLA Architecture", "GLM-4", 2024),
                ArchInnovation("Tool Use", "Function Calling", "GLM-4", 2024),
            ],
        },
    },
    # Added: Yi family (01.AI)
    "Yi": {
        "root": "01-ai/Yi-6B",
        "members": {
            "01-ai/Yi-6B": {
                "children": ["01-ai/Yi-34B", "01-ai/Yi-6B-chat"],
            },
            "01-ai/Yi-34B": {
                "children": ["01-ai/Yi-34B-chat", "01-ai/Yi-1.5-34B"],
            },
            "01-ai/Yi-1.5-34B": {
                "children": ["01-ai/Yi-1.5-70B"],
            },
        },
        "innovations": {
            "01-ai/Yi-6B": [
                ArchInnovation("Long Context", "200K Context Window", "Yi", 2023),
                ArchInnovation("RoPE", "Rotary Position Embedding", "Yi", 2023),
            ],
            "01-ai/Yi-1.5-34B": [
                ArchInnovation("GQA", "Grouped Query Attention", "Yi-1.5", 2024),
                ArchInnovation("Stronger Base", "Improved Pretraining", "Yi-1.5", 2024),
            ],
        },
    },
    # Added: Kimi (Moonshot) family
    "Kimi": {
        "root": "moonshotai/kimi-7b",
        "members": {
            "moonshotai/kimi-7b": {
                "children": ["moonshotai/kimi-20b", "moonshotai/kimi-v1"],
            },
            "moonshotai/kimi-20b": {
                "children": ["moonshotai/kimi-v1-32k"],
            },
        },
        "innovations": {
            "moonshotai/kimi-7b": [
                ArchInnovation("Long Context", "128K Context Window", "Kimi", 2023),
            ],
            "moonshotai/kimi-v1": [
                ArchInnovation("VL", "Vision-Language Support", "Kimi-V1", 2024),
                ArchInnovation("1M Context", "1M Token Context", "Kimi-V1", 2024),
            ],
        },
    },
    # Added: Phi (Microsoft) family
    "Phi": {
        "root": "microsoft/phi-1",
        "members": {
            "microsoft/phi-1": {
                "children": ["microsoft/phi-1.5", "microsoft/phi-2"],
            },
            "microsoft/phi-1.5": {
                "children": ["microsoft/phi-2"],
            },
            "microsoft/phi-2": {
                "children": ["microsoft/Phi-3-mini", "microsoft/Phi-3-small"],
            },
            "microsoft/Phi-3-mini": {
                "children": ["microsoft/Phi-3-medium", "microsoft/Phi-3-small-128k"],
            },
        },
        "innovations": {
            "microsoft/phi-1": [
                ArchInnovation("Textbooks", "High-Quality Textbook Data", "Phi-1", 2023),
                ArchInnovation("Code Data", "Synthetic Code Generation", "Phi-1", 2023),
            ],
            "microsoft/phi-2": [
                ArchInnovation("Small Scale", "2.7B Parameter Efficiency", "Phi-2", 2023),
            ],
            "microsoft/Phi-3-mini": [
                ArchInnovation("3.8B > 7B", "Outperform Larger Models", "Phi-3", 2024),
                ArchInnovation("Long Context", "128K Context", "Phi-3", 2024),
                ArchInnovation("GQA", "Grouped Query Attention", "Phi-3", 2024),
            ],
        },
    },
    # Added: Gemma (Google) family
    "Gemma": {
        "root": "google/gemma-2b",
        "members": {
            "google/gemma-2b": {
                "children": ["google/gemma-7b", "google/gemma-2b-it"],
            },
            "google/gemma-7b": {
                "children": ["google/gemma-7b-it", "google/gemma-2-9b"],
            },
            "google/gemma-2-9b": {
                "children": ["google/gemma-2-27b"],
            },
        },
        "innovations": {
            "google/gemma-2b": [
                ArchInnovation("Gemini Tech", "Based on Gemini Research", "Gemma", 2024),
                ArchInnovation("Open Weights", "Open Model Weights", "Gemma", 2024),
            ],
            "google/gemma-2-9b": [
                ArchInnovation("Gemma 2", "Improved Architecture", "Gemma-2", 2024),
                ArchInnovation("GQA", "Grouped Query Attention", "Gemma-2", 2024),
            ],
        },
    },
    # Added: StarCoder family
    "Starcoder": {
        "root": "bigcode/starcoderbase",
        "members": {
            "bigcode/starcoderbase": {
                "children": ["bigcode/starcoderbase-15b", "bigcode/starcoder-15b"],
            },
            "bigcode/starcoder-15b": {
                "children": ["bigcode/starcoder2-15b", "bigcode/starcoder2-7b"],
            },
            "bigcode/starcoder2-7b": {
                "children": ["bigcode/starcoder2-3b", "bigcode/starcoder2-1b"],
            },
        },
        "innovations": {
            "bigcode/starcoderbase": [
                ArchInnovation("FIM", "Fill-in-the-Middle", "Starcoder", 2023),
                ArchInnovation("Long Context", "8K Context", "Starcoder", 2023),
            ],
            "bigcode/starcoder2-15b": [
                ArchInnovation("96K Context", "Extended 96K Context", "Starcoder2", 2024),
                ArchInnovation("More Languages", "35+ Programming Languages", "Starcoder2", 2024),
            ],
        },
    },
    # Added: BLOOM family
    "BLOOM": {
        "root": "bigscience/bloom-560m",
        "members": {
            "bigscience/bloom-560m": {
                "children": ["bigscience/bloom-1b1", "bigscience/bloom-1b7"],
            },
            "bigscience/bloom-1b7": {
                "children": ["bigscience/bloom-3b", "bigscience/bloom-7b1"],
            },
            "bigscience/bloom-7b1": {
                "children": ["bigscience/bloomz-7b1", "bigscience/bloom-176b"],
            },
        },
        "innovations": {
            "bigscience/bloom-560m": [
                ArchInnovation("Multi-lingual", "46 Languages", "BLOOM", 2022),
                ArchInnovation("ODP", "Open Scientific Preprint License", "BLOOM", 2022),
            ],
            "bigscience/bloom-7b1": [
                ArchInnovation("176B Params", "Largest Open Multilingual Model", "BLOOM", 2022),
            ],
        },
    },
    # Added: Falcon family
    "Falcon": {
        "root": "tiiuae/falcon-rw-1b",
        "members": {
            "tiiuae/falcon-rw-1b": {
                "children": ["tiiuae/falcon-7b", "tiiuae/falcon-rw-7b"],
            },
            "tiiuae/falcon-7b": {
                "children": ["tiiuae/falcon-40b", "tiiuae/falcon-instruct-7b"],
            },
            "tiiuae/falcon-40b": {
                "children": ["tiiuae/falcon-instruct-40b"],
            },
        },
        "innovations": {
            "tiiuae/falcon-7b": [
                ArchInnovation("Billion-scale", "Web Data Filtering", "Falcon", 2023),
                ArchInnovation("LLM", "FlashAttention", "Falcon", 2023),
            ],
            "tiiuae/falcon-40b": [
                ArchInnovation("GQA", "Grouped Query Attention (40B)", "Falcon-40B", 2023),
            ],
        },
    },
    # Added: Mamba family (state-space model, SSM)
    "Mamba": {
        "root": "state-spaces/mamba-2.8b",
        "members": {
            "state-spaces/mamba-2.8b": {
                "children": ["state-spaces/mamba-2.8b-slimp", "state-spaces/mamba-1.4b"],
            },
            "state-spaces/mamba-1.4b": {
                "children": ["state-spaces/mamba-670m"],
            },
            "state-spaces/mamba-2.8b-slimp": {
                "children": ["state-spaces/mamba-1.8b"],
            },
        },
        "innovations": {
            "state-spaces/mamba-2.8b": [
                ArchInnovation("SSM", "State Space Model", "Mamba", 2024),
                ArchInnovation("Hardware-Aware", "Hardware-Aware Selection Scan", "Mamba", 2024),
                ArchInnovation("Linear Complexity", "O(n) vs O(n²) Attention", "Mamba", 2024),
            ],
            "state-spaces/mamba-2.8b-slimp": [
                ArchInnovation("Slimp", "Compression for Deployment", "Mamba-Slimp", 2024),
            ],
        },
    },
    # Added: RWKV family (RNN-Transformer hybrid)
    "RWKV": {
        "root": "RWKV/rwkv-4-1b5",
        "members": {
            "RWKV/rwkv-4-1b5": {
                "children": ["RWKV/rwkv-4-2b5", "RWKV/rwkv-4-3b"],
            },
            "RWKV/rwkv-4-2b5": {
                "children": ["RWKV/rwkv-4-7b", "RWKV/rwkv-5-3b"],
            },
            "RWKV/rwkv-4-3b": {
                "children": ["RWKV/rwkv-4-7b"],
            },
            "RWKV/rwkv-4-7b": {
                "children": ["RWKV/rwkv-5-7b", "RWKV/rwkv-6-13b"],
            },
        },
        "innovations": {
            "RWKV/rwkv-4-1b5": [
                ArchInnovation("WKV", "Weighted Key-Value", "RWKV", 2023),
                ArchInnovation("RNN-Transformer", "RNN-Transformer Hybrid", "RWKV", 2023),
                ArchInnovation("Linear Complexity", "O(n) for Long Context", "RWKV", 2023),
            ],
            "RWKV/rwkv-5-3b": [
                ArchInnovation("Emoji Support", "Better Multi-lingual", "RWKV-5", 2024),
            ],
            "RWKV/rwkv-6-13b": [
                ArchInnovation("RWKV-6", "Enhanced Positional Encoding", "RWKV-6", 2024),
            ],
        },
    },
    # Added: InternLM family (Shanghai AI Lab)
    "InternLM": {
        "root": "internlm/internlm-7b",
        "members": {
            "internlm/internlm-7b": {
                "children": ["internlm/internlm-20b", "internlm/internlm2-7b"],
            },
            "internlm/internlm-20b": {
                "children": ["internlm/internlm2-20b", "internlm/internlm2.5-20b"],
            },
            "internlm/internlm2-7b": {
                "children": ["internlm/internlm2.5-7b", "internlm/internlm3-7b"],
            },
            "internlm/internlm2-20b": {
                "children": ["internlm/internlm2.5-20b", "internlm/internlm3-20b"],
            },
        },
        "innovations": {
            "internlm/internlm-7b": [
                ArchInnovation("Long Context", "8K-32K Context", "InternLM", 2023),
                ArchInnovation("Open Weights", "Fully Open Source", "InternLM", 2023),
            ],
            "internlm/internlm2-7b": [
                ArchInnovation("MoE", "Mixture of Experts", "InternLM2", 2024),
                ArchInnovation("100K Context", "Extended to 100K", "InternLM2", 2024),
            ],
            "internlm/internlm2.5-20b": [
                ArchInnovation("GQA", "Grouped Query Attention", "InternLM2.5", 2024),
                ArchInnovation("Code Model", "InternLM-Coder", "InternLM2.5", 2024),
            ],
        },
    },
    # Added: Baichuan family
    "BaiChuan": {
        "root": "baichuan-inc/Baichuan-7B",
        "members": {
            "baichuan-inc/Baichuan-7B": {
                "children": ["baichuan-inc/Baichuan-13B", "baichuan-inc/Baichuan2-7B"],
            },
            "baichuan-inc/Baichuan-13B": {
                "children": ["baichuan-inc/Baichuan2-13B", "baichuan-inc/Baichuan2-13B-chat"],
            },
            "baichuan-inc/Baichuan2-7B": {
                "children": ["baichuan-inc/Baichuan2-7B-chat", "baichuan-inc/Baichuan2-53B"],
            },
            "baichuan-inc/Baichuan2-13B": {
                "children": ["baichuan-inc/Baichuan2-13B-chat", "baichuan-inc/Baichuan2-53B-chat"],
            },
        },
        "innovations": {
            "baichuan-inc/Baichuan-7B": [
                ArchInnovation("BaiChuan", "Bilingual (ZH/EN)", "Baichuan", 2023),
                ArchInnovation("Dynamic NTK", "Dynamic NTK Scaling", "Baichuan", 2023),
            ],
            "baichuan-inc/Baichuan2-7B": [
                ArchInnovation("2.0", "Improved Training Data", "Baichuan2", 2023),
                ArchInnovation("GQA", "Grouped Query Attention", "Baichuan2", 2023),
            ],
            "baichuan-inc/Baichuan2-53B": [
                ArchInnovation("53B", "Large Parameter Model", "Baichuan2-53B", 2023),
            ],
        },
    },
    # Added: Skywork family
    "Skywork": {
        "root": "Skywork/Skywork-13B",
        "members": {
            "Skywork/Skywork-13B": {
                "children": ["Skywork/Skywork-13B-chat", "Skywork/Skywork-MoE-13B"],
            },
            "Skywork/Skywork-13B-chat": {
                "children": ["Skywork/Skywork-13B-3k"],
            },
            "Skywork/Skywork-MoE-13B": {
                "children": ["Skywork/Skywork-13B-16K"],
            },
        },
        "innovations": {
            "Skywork/Skywork-13B": [
                ArchInnovation("Open Source", "Fully Open Weights", "Skywork", 2023),
                ArchInnovation("Long Context", "4K-16K Context", "Skywork", 2023),
            ],
            "Skywork/Skywork-MoE-13B": [
                ArchInnovation("MoE", "Mixture of Experts", "Skywork-MoE", 2024),
                ArchInnovation("SFT", "Supervised Fine-Tuning", "Skywork-MoE", 2024),
            ],
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Evolution Tree Builder
# ─────────────────────────────────────────────────────────────────────────────

class EvolutionTree:
    """
    Builds and manages architecture evolution trees.

    Usage:
        tree = EvolutionTree()
        tree.load_builtin_families()
        tree.add_model("Qwen/Qwen2.5-72B")
        tree.build()
        tree.visualize("output/evolution_tree.html")
    """

    def __init__(self, custom_families: Optional[Dict] = None, trust_remote_code: bool = False):
        """
        Initialize the evolution tree builder.

        Args:
            custom_families: Optional custom family definitions to merge
        """
        self.nodes: Dict[str, ArchNode] = {}
        self.families: Dict[str, Dict] = {}

        # Merge custom families with defaults
        if custom_families:
            self._merge_families(DEFAULT_FAMILIES, custom_families)
        else:
            self.families = DEFAULT_FAMILIES.copy()

        self.trust_remote_code = trust_remote_code

    def _merge_families(self, base: Dict, override: Dict) -> None:
        """Merge custom families into base."""
        for family_name, family_data in override.items():
            if family_name in base:
                base[family_name]["members"].update(family_data.get("members", {}))
            else:
                base[family_name] = family_data
        self.families = base

    def load_builtin_families(self) -> None:
        """Load the default model family relationships."""
        self.families = DEFAULT_FAMILIES.copy()
        logger.info(f"Loaded {len(self.families)} builtin families")

    def add_model(
        self,
        model_id: str,
        config: Optional[Dict[str, Any]] = None,
        parent: Optional[str] = None,
        innovations: Optional[List[ArchInnovation]] = None,
    ) -> ArchNode:
        """
        Add a model to the evolution tree.

        Args:
            model_id: HuggingFace model ID
            config: Model configuration dict (will fetch if not provided)
            parent: Parent model ID (inferred if not provided)
            innovations: List of architectural innovations

        Returns:
            Created ArchNode
        """
        if model_id in self.nodes:
            logger.warning(f"Model {model_id} already exists, skipping")
            return self.nodes[model_id]

        # Fetch config if not provided
        if config is None:
            try:
                config = hf_load_config(
                    model_id,
                    security={
                        "trust_remote_code": self.trust_remote_code,
                        "allow_network": True,
                        "local_files_only": False,
                    },
                )
                config = config.to_dict()
            except Exception as e:
                logger.warning(f"Failed to fetch config for {model_id}: {e}")
                # Use fallback parameters if available
                config = FALLBACK_PARAMS.get(model_id, {})
                if config:
                    logger.info(f"Using fallback params for {model_id}")
                else:
                    logger.warning(f"No fallback params for {model_id}")

        # Infer parent if not provided
        if parent is None:
            parent = self._infer_parent(model_id)

        node = ArchNode(
            model_id=model_id,
            config=config,
            parent=parent,
            innovations=innovations or [],
        )
        self.nodes[model_id] = node

        # Update parent's children
        if parent and parent in self.nodes:
            if model_id not in self.nodes[parent].children:
                self.nodes[parent].children.append(model_id)

        logger.info(f"Added node: {model_id} (parent: {parent})")
        return node

    def _infer_parent(self, model_id: str) -> Optional[str]:
        """
        Infer the parent model based on naming patterns and known relationships.

        This is a heuristic-based inference using:
        1. Version numbers (e.g., Qwen2 → Qwen1.5 → Qwen)
        2. Size variants (e.g., 72B → 7B)
        3. Known family relationships
        """
        name = model_id.lower()
        model_name = model_id.split("/")[-1] if "/" in model_id else model_id

        # Check against known families
        for _family_name, family_data in self.families.items():
            for known_model, _info in family_data.get("members", {}).items():
                if known_model.lower() in name or known_model.split("/")[-1].lower() in name:
                    return known_model

        # Infer from version patterns
        version_patterns = [
            # Qwen pattern: Qwen3 → Qwen2.5 → Qwen2 → Qwen1.5 → Qwen
            (r"qwen3[.-](\d+b?)", "Qwen2.5", "Qwen/Qwen2.5-{size}"),
            (r"qwen2[.-](\d+b?)", "Qwen1.5", "Qwen/Qwen1.5-{size}"),
            (r"qwen1\.5[.-](\d+b?)", "Qwen", "Qwen/Qwen-{size}"),
            # LLaMA pattern: Llama-3.1 → Llama-3 → Llama-2
            (r"llama-3\.1[.-](\d+b?)", "meta-llama/Llama-3-{size}", None),
            (r"llama-3[.-](\d+b?)", "meta-llama/Llama-2-{size}", None),
            (r"llama-2[.-](\d+b?)", "meta-llama/Llama-{size}", None),
            # DeepSeek pattern
            (r"deepseek-v3", "deepseek-ai/DeepSeek-V2", None),
            (r"deepseek-v2", "deepseek-ai/deepseek-moe-16b", None),
        ]

        for pattern, parent_pattern, _ in version_patterns:
            import re
            if re.search(pattern, name):
                # Try to extract size
                size_match = re.search(r"(\d+[bB])", model_name)
                if size_match and parent_pattern:
                    size = size_match.group(1)
                    parent_id = parent_pattern.format(size=size)
                    # Check if this parent exists in our nodes or known families
                    for known_model in self.nodes.keys():
                        if parent_id.lower() in known_model.lower():
                            return known_model

        return None

    def build(self) -> None:
        """Build the tree by establishing all parent-child relationships."""
        for _family_name, family_data in self.families.items():
            root = family_data.get("root")
            if root:
                # Ensure root node exists
                if root not in self.nodes:
                    self.add_model(root)

            # Process all known members
            for model_id, info in family_data.get("members", {}).items():
                if model_id not in self.nodes:
                    self.add_model(model_id)

                # Set children from family data
                children = info.get("children", [])
                for child in children:
                    if child not in self.nodes:
                        self.add_model(child)
                    if child not in self.nodes[model_id].children:
                        self.nodes[model_id].children.append(child)
                    self.nodes[child].parent = model_id

                # Add innovations
                innovations = family_data.get("innovations", {}).get(model_id, [])
                for innovation in innovations:
                    if innovation not in self.nodes[model_id].innovations:
                        self.nodes[model_id].innovations.append(innovation)

        logger.info(f"Built tree with {len(self.nodes)} nodes")

    def get_subtree(self, root_model_id: str) -> EvolutionTree:
        """Extract a subtree rooted at the specified model."""
        subtree = EvolutionTree()
        subtree.families = {}

        def collect_children(model_id: str) -> None:
            if model_id not in self.nodes:
                return
            node = self.nodes[model_id]
            subtree.add_model(
                model_id=node.model_id,
                config=node.config,
                parent=node.parent,
                innovations=node.innovations,
            )
            for child_id in node.children:
                collect_children(child_id)

        collect_children(root_model_id)
        subtree.build()
        return subtree

    def to_dict(self) -> Dict[str, Any]:
        """Export tree to dictionary format."""
        return {
            "nodes": {
                model_id: {
                    "model_id": node.model_id,
                    "model_name": node.model_name,
                    "family": node.family,
                    "parent": node.parent,
                    "children": node.children,
                    "innovations": [
                        {
                            "name": i.name,
                            "description": i.description,
                            "introduced_in": i.introduced_in,
                        }
                        for i in node.innovations
                    ],
                    "key_params": node.get_key_params(),
                }
                for model_id, node in self.nodes.items()
            },
            "families": list(self.families.keys()),
            "total_nodes": len(self.nodes),
        }

    def get_family(self, model_id: str) -> Optional[str]:
        """Get the family name for a model."""
        if model_id in self.nodes:
            return self.nodes[model_id].family
        return None

    def get_innovation_timeline(self) -> List[Dict[str, Any]]:
        """Get a timeline of all innovations in the tree."""
        innovations = []
        for node in self.nodes.values():
            for inn in node.innovations:
                innovations.append({
                    "model": node.model_id,
                    "family": node.family,
                    "innovation": inn.name,
                    "description": inn.description,
                    "year": inn.year,
                })
        return sorted(innovations, key=lambda x: x["year"])

    def find_common_ancestor(self, model1: str, model2: str) -> Optional[str]:
        """Find the common ancestor of two models."""
        if model1 not in self.nodes or model2 not in self.nodes:
            return None

        # Build ancestor chains
        ancestors1 = set()
        current = model1
        visited1 = set()
        while current:
            if current in visited1:
                # Guard against accidental cycles in parent pointers
                logger.warning("Cycle detected while building ancestor chain for %s at %s", model1, current)
                break
            visited1.add(current)
            ancestors1.add(current)
            current = self.nodes[current].parent if current in self.nodes else None

        # Find common ancestor
        current = model2
        visited2 = set()
        while current:
            if current in visited2:
                logger.warning("Cycle detected while walking ancestor chain for %s at %s", model2, current)
                break
            visited2.add(current)
            if current in ancestors1:
                return current
            current = self.nodes[current].parent if current in self.nodes else None

        return None

    def compute_similarity(self, model1: str, model2: str) -> float:
        """
        Compute architecture similarity between two models (0-1).

        Uses cosine similarity on key architecture parameters.
        """
        if model1 == model2 and model1 in self.nodes:
            return 1.0
        if model1 not in self.nodes or model2 not in self.nodes:
            return 0.0

        params1 = self.nodes[model1].get_key_params()
        params2 = self.nodes[model2].get_key_params()

        # Extract numeric features
        features1 = []
        features2 = []
        feature_names = []

        for key in ["hidden_size", "num_hidden_layers", "num_attention_heads",
                    "num_key_value_heads", "intermediate_size", "vocab_size"]:
            v1 = params1.get(key) or 0
            v2 = params2.get(key) or 0
            # Normalize by log scale for large numbers
            import math
            features1.append(math.log1p(v1))
            features2.append(math.log1p(v2))
            feature_names.append(key)

        # Add categorical features
        features1.append(1.0 if params1.get("is_moe") else 0.0)
        features2.append(1.0 if params2.get("is_moe") else 0.0)
        feature_names.append("is_moe")

        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(features1, features2))
        norm1 = math.sqrt(sum(a * a for a in features1))
        norm2 = math.sqrt(sum(b * b for b in features2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        sim = dot_product / (norm1 * norm2)
        # Numerical safety: keep within [0, 1]
        if sim < 0.0:
            return 0.0
        if sim > 1.0:
            return 1.0
        return sim

    def save(self, path: str) -> None:
        """Save tree to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved tree to {path}")

    def load(self, path: str) -> None:
        """Load tree from JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self.nodes = {}
        for _model_id, node_data in data.get("nodes", {}).items():
            innovations = [
                ArchInnovation(
                    name=i["name"],
                    description=i["description"],
                    introduced_in=i["introduced_in"],
                    year=0,  # Not stored
                )
                for i in node_data.get("innovations", [])
            ]
            self.add_model(
                model_id=node_data["model_id"],
                config={},
                parent=node_data.get("parent"),
                innovations=innovations,
            )

        logger.info(f"Loaded tree from {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Module Exports
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "ArchNode",
    "ArchInnovation",
    "ArchitectureMetrics",
    "EvolutionTree",
    "DEFAULT_FAMILIES",
]
