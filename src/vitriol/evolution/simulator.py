"""
Architecture Performance Simulator
=================================

Estimate performance metrics (FLOPs, VRAM, latency) for LLM architectures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    """Result of architecture simulation."""
    model_id: str
    config: Dict[str, Any]

    # Computed metrics
    total_params: int
    trainable_params: int
    active_params_per_token: int  # For MoE models
    flops_per_token: float
    flops_per_second: float  # Theoretical throughput

    # Memory estimates (GB)
    vram_full_model: float
    vram_inference: float
    vram_training: float
    kv_cache_estimate: float

    # Performance estimates
    inference_latency_ms: float  # Per token
    tokens_per_second: float
    memory_bandwidth_gbs: float

    # Efficiency metrics
    params_per_vram: float  # Params per GB VRAM
    flops_per_param: float  # Compute intensity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
            "active_params_per_token": self.active_params_per_token,
            "flops_per_token": self.flops_per_token,
            "flops_per_second": self.flops_per_second,
            "vram_full_model": round(self.vram_full_model, 2),
            "vram_inference": round(self.vram_inference, 2),
            "vram_training": round(self.vram_training, 2),
            "kv_cache_estimate": round(self.kv_cache_estimate, 2),
            "inference_latency_ms": round(self.inference_latency_ms, 2),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "memory_bandwidth_gbs": round(self.memory_bandwidth_gbs, 2),
            "params_per_vram": round(self.params_per_vram, 2),
            "flops_per_param": round(self.flops_per_param, 2),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Constants and Coefficients
# ─────────────────────────────────────────────────────────────────────────────

# Memory constants (bytes per parameter)
BYTES_FP32 = 4
BYTES_FP16 = 2
BYTES_BF16 = 2
BYTES_INT8 = 1
BYTES_Q4 = 0.5

# GPU memory bandwidth (GB/s) - typical values
BANDWIDTH_A100 = 2039  # GB/s
BANDWIDTH_H100 = 3350  # GB/s
BANDWIDTH_V100 = 900   # GB/s
BANDWIDTH_A10 = 400    # GB/s

# GPU compute (TFLOPS) - theoretical
FLOPS_A100 = 312e12  # FP32
FLOPS_H100 = 989e12   # FP32
FLOPS_A10 = 125e12   # FP32

# Activation memory factor (relative to model size)
ACTIVATION_FACTOR_TRAINING = 3.0  # Training needs more for activations
ACTIVATION_FACTOR_INFERENCE = 0.2  # Inference is more memory-efficient


# ─────────────────────────────────────────────────────────────────────────────
# Simulator
# ─────────────────────────────────────────────────────────────────────────────

class ArchSimulator:
    """
    Simulate and estimate performance metrics for LLM architectures.

    Uses analytical models based on:
    - Total parameter count
    - Architecture type (dense vs MoE)
    - Attention configuration (MHA, GQA, MQA, MLA)
    - Batch size and sequence length

    Usage:
        simulator = ArchSimulator()
        result = simulator.simulate(config)
        logger.info("VRAM needed: %.1f GB", result.vram_inference)
    """

    def __init__(
        self,
        dtype: str = "bfloat16",
        device: str = "cuda",
        gpu_model: str = "A100",
    ):
        """
        Initialize the simulator.

        Args:
            dtype: Data type for computation (fp32, fp16, bfloat16)
            device: Device type (cuda, mps)
            gpu_model: GPU model for bandwidth/compute estimates
        """
        self.dtype = dtype
        self.device = device
        self.gpu_model = gpu_model

        # Get GPU characteristics
        self.bandwidth = self._get_bandwidth(gpu_model)
        self.peak_flops = self._get_peak_flops(gpu_model)

        # Bytes per parameter based on dtype
        self.bytes_per_param = self._get_bytes_per_param(dtype)

    def _get_bandwidth(self, gpu_model: str) -> float:
        """Get memory bandwidth for GPU model."""
        bandwidths = {
            "H100": BANDWIDTH_H100,
            "A100": BANDWIDTH_A100,
            "V100": BANDWIDTH_V100,
            "A10": BANDWIDTH_A10,
            "default": BANDWIDTH_A100,
        }
        return bandwidths.get(gpu_model, BANDWIDTH_A100)

    def _get_peak_flops(self, gpu_model: str) -> float:
        """Get peak FLOPs for GPU model."""
        flops = {
            "H100": FLOPS_H100,
            "A100": FLOPS_A100,
            "A10": FLOPS_A10,
            "default": FLOPS_A100,
        }
        return flops.get(gpu_model, FLOPS_A100)

    def _get_bytes_per_param(self, dtype: str) -> float:
        """Get bytes per parameter for dtype."""
        mapping = {
            "fp32": BYTES_FP32,
            "fp16": BYTES_FP16,
            "bfloat16": BYTES_BF16,
            "int8": BYTES_INT8,
            "q4": BYTES_Q4,
        }
        return mapping.get(dtype, BYTES_BF16)

    def simulate(
        self,
        model_id: str,
        config: Dict[str, Any],
        batch_size: int = 1,
        seq_length: int = 512,
    ) -> SimulationResult:
        """
        Simulate performance metrics for a model configuration.

        Args:
            model_id: Model identifier
            config: Model configuration dict
            batch_size: Batch size for simulation
            seq_length: Sequence length for simulation

        Returns:
            SimulationResult with estimated metrics
        """
        # Extract architecture parameters
        hidden_size = config.get("hidden_size", 4096)
        num_layers = config.get("num_hidden_layers", 32)
        num_heads = config.get("num_attention_heads", 32)
        num_kv_heads = config.get("num_key_value_heads", num_heads) or num_heads
        intermediate_size = config.get("intermediate_size", hidden_size * 4)
        vocab_size = config.get("vocab_size", 32000)

        # Check for MoE
        num_experts = config.get("num_local_experts", 0) or 0
        is_moe = num_experts > 1
        num_routed = config.get("n_routed_experts", num_experts) if is_moe else 0
        topk = config.get("num_experts_per_tok", 2) if is_moe else 1

        # Attention type
        attn_type = self._detect_attention_type(num_heads, num_kv_heads, config)

        # Calculate parameters
        total_params = self._estimate_params(
            hidden_size, num_layers, num_heads, num_kv_heads,
            intermediate_size, vocab_size, num_experts, is_moe
        )

        # Calculate FLOPs
        flops_per_token = self._estimate_flops(
            hidden_size, num_layers, num_heads, num_kv_heads,
            intermediate_size, seq_length, is_moe, num_routed, topk, attn_type
        )

        # Calculate memory
        vram_model = total_params * self.bytes_per_param / (1024 ** 3)

        kv_quant = config.get("kv_codec") or config.get("kv_quant") or config.get("kv_cache_type")
        kv_cache = self._estimate_kv_cache(
            hidden_size, num_layers, num_kv_heads, batch_size, seq_length, kv_quant=kv_quant
        )

        sparse_v_speedup = 1.0
        if config.get("use_sparse_v", False):
            sparse_v_speedup = 1.228
            flops_per_token /= sparse_v_speedup

        compute_skip_speedup = 1.0
        compute_skip = config.get("compute_skip")
        if compute_skip:
            if isinstance(compute_skip, dict):
                kept_fraction = float(compute_skip.get("kept_fraction", 0.7))
                if "epsilon" in compute_skip and "kept_fraction" not in compute_skip:
                    eps = float(compute_skip.get("epsilon", 0.02))
                    kept_fraction = max(0.2, 1.0 - 8.0 * eps)
            else:
                kept_fraction = 0.7
            kept_fraction = max(0.1, min(1.0, kept_fraction))
            compute_skip_speedup = min(2.0, 1.0 / kept_fraction)
            flops_per_token /= compute_skip_speedup

        vram_inference = vram_model + kv_cache
        vram_training = vram_model * ACTIVATION_FACTOR_TRAINING + kv_cache

        # Calculate throughput
        memory_bandwidth = self.bandwidth * (1024 ** 3)  # bytes/s
        bytes_per_token = vram_model * (1024 ** 3) / total_params  # bytes
        max_tokens_per_sec = memory_bandwidth / bytes_per_token if bytes_per_token > 0 else 0
        max_tokens_per_sec *= (sparse_v_speedup * compute_skip_speedup)

        # FLOPs-based throughput
        flops_per_sec = min(
            self.peak_flops,
            max_tokens_per_sec * flops_per_token
        )
        tokens_per_second = flops_per_sec / flops_per_token if flops_per_token > 0 else 0

        # Latency estimate
        latency_ms = 1000 / tokens_per_second if tokens_per_second > 0 else 0

        # Efficiency metrics
        params_per_vram = total_params / (vram_model * (1024 ** 3)) if vram_model > 0 else 0
        flops_per_param = flops_per_token / total_params if total_params > 0 else 0

        # For MoE, calculate active params
        active_params = total_params
        if is_moe and num_routed > 0:
            # Only topk experts are active per token
            expert_params = intermediate_size * num_experts
            active_params = total_params - expert_params + (expert_params / num_experts * topk)

        return SimulationResult(
            model_id=model_id,
            config=config,
            total_params=total_params,
            trainable_params=total_params,
            active_params_per_token=int(active_params),
            flops_per_token=flops_per_token,
            flops_per_second=flops_per_sec,
            vram_full_model=vram_model,
            vram_inference=vram_inference,
            vram_training=vram_training,
            kv_cache_estimate=kv_cache,
            inference_latency_ms=latency_ms,
            tokens_per_second=tokens_per_second,
            memory_bandwidth_gbs=self.bandwidth,
            params_per_vram=params_per_vram,
            flops_per_param=flops_per_param,
        )

    def _estimate_vram(self, config: Dict[str, Any], batch_size: int = 1, seq_length: int = 512) -> float:
        """Estimate inference VRAM (model params + KV cache), in GB.

        This is a lightweight helper for tests/quick sizing. It reuses the same
        estimation logic as `simulate()`.
        """
        hidden_size = config.get("hidden_size", 4096)
        num_layers = config.get("num_hidden_layers", 32)
        num_heads = config.get("num_attention_heads", 32)
        num_kv_heads = config.get("num_key_value_heads", num_heads) or num_heads
        intermediate_size = config.get("intermediate_size", hidden_size * 4)
        vocab_size = config.get("vocab_size", 32000)

        num_experts = config.get("num_local_experts", 0) or 0
        is_moe = num_experts > 1

        total_params = self._estimate_params(
            hidden_size,
            num_layers,
            num_heads,
            num_kv_heads,
            intermediate_size,
            vocab_size,
            num_experts,
            is_moe,
        )

        vram_model = total_params * self.bytes_per_param / (1024 ** 3)
        kv_quant = config.get("kv_codec") or config.get("kv_quant") or config.get("kv_cache_type")
        kv_cache = self._estimate_kv_cache(
            hidden_size, num_layers, num_kv_heads, batch_size, seq_length, kv_quant=kv_quant
        )
        return float(vram_model + kv_cache)

    def _detect_attention_type(
        self,
        num_heads: int,
        num_kv_heads: int,
        config: Dict[str, Any],
    ) -> str:
        """Detect attention type from config."""
        if config.get("num_kv_heads") == 1:
            return "MQA"
        elif num_kv_heads < num_heads:
            return "GQA"
        elif config.get("multi_head_latent_attention"):
            return "MLA"
        else:
            return "MHA"

    def _estimate_params(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        vocab_size: int,
        num_experts: int,
        is_moe: bool,
    ) -> int:
        """Estimate total parameter count."""
        # Embedding
        vocab_params = vocab_size * hidden_size

        # Attention projection
        q_params = hidden_size * hidden_size
        k_params = hidden_size * hidden_size * (num_kv_heads / num_heads)
        v_params = hidden_size * hidden_size * (num_kv_heads / num_heads)
        o_params = hidden_size * hidden_size

        # Attention total per layer
        attn_params_per_layer = q_params + k_params + v_params + o_params

        # FFN per layer
        if is_moe:
            # MoE FFN: shared experts + routed experts
            ffn_params_per_layer = 3 * hidden_size * hidden_size  # Gate/Up/Down per expert
            ffn_params_per_layer * num_experts
        else:
            # Standard FFN
            ffn_params_per_layer = 2 * hidden_size * intermediate_size

        # Layer norms
        ln_params = 4 * hidden_size  # 2 RMSNorm + 2 LayerNorms

        # Total per transformer layer
        params_per_layer = attn_params_per_layer + ffn_params_per_layer + ln_params

        # All layers
        transformer_params = params_per_layer * num_layers

        # Output head
        head_params = hidden_size * vocab_size

        # Final norm
        final_norm = 2 * hidden_size

        total = vocab_params + transformer_params + head_params + final_norm

        # Account for tied embeddings (approximate reduction)
        if not is_moe:
            total = int(total * 0.95)  # ~5% tied embeddings

        return total

    def _estimate_flops(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        seq_length: int,
        is_moe: bool,
        num_routed: int,
        topk: int,
        attn_type: str,
    ) -> float:
        """
        Estimate FLOPs per token for forward pass.

        Based on the Megatron formula:
        FLOPs = 2 * [attention_params * seq + MLP_params * seq]
        """
        # Attention FLOPs
        # Q, K, V projection: 3 * hidden^2 * seq
        # Attention scores: 2 * hidden * hidden * seq
        # Output projection: hidden^2 * seq

        if attn_type == "GQA":
            # GQA reduces K/V computation
            kv_ratio = num_kv_heads / num_heads
            attn_flops = (2 + 2 * kv_ratio + 1) * hidden_size ** 2 * seq_length
        elif attn_type == "MQA":
            attn_flops = (2 + 2 * (1 / num_heads) + 1) * hidden_size ** 2 * seq_length
        else:
            attn_flops = 6 * hidden_size ** 2 * seq_length

        attn_flops *= num_layers

        # FFN FLOPs
        if is_moe:
            # MoE: topk experts are activated
            ffn_flops = 2 * hidden_size * intermediate_size * seq_length * num_routed * topk / max(num_routed, 1)
        else:
            ffn_flops = 4 * hidden_size * intermediate_size * seq_length

        ffn_flops *= num_layers

        total_flops = attn_flops + ffn_flops
        return float(total_flops)

    def _estimate_kv_cache(
        self,
        hidden_size: int,
        num_layers: int,
        num_kv_heads: int,
        batch_size: int,
        seq_length: int,
        kv_quant: Optional[Any] = None,
    ) -> float:
        """Estimate KV cache memory in GB."""
        # KV cache per layer: 2 * batch * seq * num_kv_heads * head_dim
        head_dim = hidden_size // max(num_kv_heads, 1)
        kv_per_layer = 2 * batch_size * seq_length * num_kv_heads * head_dim

        # All layers
        kv_total = kv_per_layer * num_layers

        from ..kv.codec import kv_bytes_per_value

        bytes_total = kv_total * kv_bytes_per_value(kv_quant)

        return bytes_total / (1024 ** 3)

    def compare_simulations(
        self,
        configs: List[Dict[str, Any]],
        model_ids: Optional[List[str]] = None,
    ) -> List[SimulationResult]:
        """Simulate multiple configurations and compare."""
        if model_ids is None:
            model_ids = [f"Model_{i}" for i in range(len(configs))]

        results = []
        for config, model_id in zip(configs, model_ids):
            result = self.simulate(model_id, config)
            results.append(result)

        return results


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────

def quick_estimate(
    hidden_size: int,
    num_layers: int,
    vocab_size: int = 32000,
    is_moe: bool = False,
    num_experts: int = 0,
) -> Dict[str, Any]:
    """
    Quick parameter and memory estimate without full simulation.

    Args:
        hidden_size: Hidden dimension
        num_layers: Number of layers
        vocab_size: Vocabulary size
        is_moe: Whether model is MoE
        num_experts: Number of experts (for MoE)

    Returns:
        Dictionary with estimates
    """
    # Rough parameter estimate
    params_per_layer = (
        4 * hidden_size ** 2  # Attention + FFN
    )
    total_params = params_per_layer * num_layers + vocab_size * hidden_size

    if is_moe and num_experts > 0:
        # Add expert parameters
        total_params += hidden_size * num_experts

    # Memory (bfloat16)
    vram_gb = total_params * 2 / (1024 ** 3)

    return {
        "estimated_params": total_params,
        "estimated_vram_gb": round(vram_gb, 2),
        "params_per_layer": params_per_layer,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Module Exports
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "ArchSimulator",
    "SimulationResult",
    "quick_estimate",
]
