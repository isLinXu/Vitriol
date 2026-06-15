"""Type-safe TypedDict definitions for Vitriol core modules.

Centralising TypedDict definitions avoids re-declaring them in every module
and gives a single source of truth for the shape of dicts that cross
module boundaries (e.g. HuggingFace configs, benchmark results).
"""
from __future__ import annotations

from typing import Dict, List, Optional, TypedDict


class HFConfigDict(TypedDict, total=False):
    """HuggingFace-style model config dictionary.

    Covers the keys commonly used by Vitriol's architecture analysis
    and weight generation.  Fields are optional so that partial configs
    (e.g. after shrinking) are valid.
    """
    model_type: str
    architectures: List[str]
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    hidden_act: str
    rms_norm_eps: float
    attention_dropout: float
    hidden_dropout_prob: float
    use_bias: bool
    rope_theta: float
    use_cache: bool
    # MLA
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    kv_lora_rank: int
    q_lora_rank: int
    # MoE
    num_experts: int
    n_routed_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    # Mamba
    d_state: int
    d_conv: int
    expand_factor: int
    # Vitriol marker
    _vitriol_nas_gene: Dict[str, object]


class BenchResultDict(TypedDict, total=False):
    """Structured benchmark result returned by runner functions."""
    model_id: str
    preset: str
    prompt_tokens: int
    max_new_tokens: int
    ok: bool
    error: Optional[str]
    # Memory
    tuned_memory: Dict[str, object]
    estimated_kv_megabytes: float
    peak_device_megabytes: float
    # Timing
    timing_ms: float
    # Quality metrics
    key_mse: float
    value_mse: float
    logits_mse: float
    output_mse: float
    residual_gain_k: float
    residual_gain_v: float
    k_cosine: float
    v_cosine: float
    # Policy
    preset_name: str
    quantized_kv_start: int
    quantized_layers: int
    policy_insights: Dict[str, object]
    # Comparison
    case_diffs: List[Dict[str, object]]
    delta_speedup: float
    changed_layers: List[Dict[str, object]]


class GenerationResultDict(TypedDict):
    """JSON-serialisable result from MinimalWeightGenerator."""
    output_dir: str
    manifest_path: Optional[str]
    index_path: Optional[str]
    total_size: int
    generated_at: str


class ShardMapDict(TypedDict, total=False):
    """Shard index mapping parameter names to shard filenames."""
    metadata: Dict[str, object]
    weight_map: Dict[str, str]
