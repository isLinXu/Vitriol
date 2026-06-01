"""Hunyuan v3 family analyzer."""

from typing import Any

from ..core import Architecture, Layer
from ._helpers import (
    _cfg_get,
    _finalize_architecture,
    _safe_float,
)
from .base import TransformerAnalyzer


class Hy3Analyzer(TransformerAnalyzer):
    """Tencent Hy3 / hy_v3 analyzer with dense-prefix MoE and GQA metadata."""

    def analyze(self, config: Any) -> Architecture:
        vocab_size = int(getattr(config, 'vocab_size', 0) or 0)
        hidden_size = int(getattr(config, 'hidden_size', 0) or 0)
        num_layers = int(getattr(config, 'num_hidden_layers', 0) or 0)
        num_heads = int(getattr(config, 'num_attention_heads', 0) or 0)
        if num_heads <= 0 and hidden_size:
            num_heads = 1
        num_kv_heads = int(getattr(config, 'num_key_value_heads', num_heads) or 0) or num_heads
        head_dim = int(getattr(config, 'head_dim', 0) or 0)
        if head_dim <= 0:
            head_dim = (hidden_size // num_heads) if num_heads else 0

        dense_intermediate = int(getattr(config, 'intermediate_size', 0) or 0)
        moe_intermediate = int(getattr(config, 'moe_intermediate_size', dense_intermediate) or 0)
        expert_hidden_dim = int(getattr(config, 'expert_hidden_dim', moe_intermediate) or 0)
        num_experts = int(getattr(config, 'num_experts', 0) or 0)
        active_experts = int(getattr(config, 'num_experts_per_tok', 0) or 0)
        shared_experts = int(getattr(config, 'num_shared_experts', 0) or 0)
        first_dense = int(getattr(config, 'first_k_dense_replace', 0) or 0)
        mtp_layers = int(getattr(config, 'num_nextn_predict_layers', 0) or 0)
        max_position = int(getattr(config, 'max_position_embeddings', 0) or 0)
        route_norm = bool(getattr(config, 'route_norm', False))
        router_sigmoid = bool(getattr(config, 'moe_router_use_sigmoid', False))
        router_bias = bool(getattr(config, 'moe_router_enable_expert_bias', False))
        output_router_logits = bool(getattr(config, 'output_router_logits', False))
        router_scaling_factor = float(getattr(config, 'router_scaling_factor', 0.0) or 0.0)
        rope_theta = 0.0
        rope_parameters = getattr(config, 'rope_parameters', None)
        if rope_parameters is not None:
            rope_theta = _safe_float(_cfg_get(rope_parameters, 'rope_theta', 0.0), 0.0)

        features = ["Hy3", "MoE", "GQA", "RMSNorm", "SwiGLU"]
        if getattr(config, 'qk_norm', False):
            features.append("QK-Norm")
        if max_position >= 131072:
            features.append("Long Context")
        if mtp_layers > 0:
            features.append(f"MTP ({mtp_layers})")
        if first_dense > 0:
            features.append(f"Dense Prefix ({first_dense})")
        if rope_theta:
            features.append("RoPE")
        if route_norm:
            features.append("RouteNorm")
        if router_sigmoid:
            features.append("Sigmoid Router")
        if router_bias:
            features.append("Router Bias")

        layers = []
        total_params = 0

        emb_params = vocab_size * hidden_size
        layers.append(
            Layer(
                "Token Embedding",
                "embedding",
                emb_params,
                (vocab_size, hidden_size),
                f"Vocab: {vocab_size}, Max ctx: {max_position}",
            )
        )
        total_params += emb_params

        ln_params = hidden_size
        q_params = hidden_size * (num_heads * head_dim)
        k_params = hidden_size * (num_kv_heads * head_dim)
        v_params = hidden_size * (num_kv_heads * head_dim)
        o_params = (num_heads * head_dim) * hidden_size
        attn_params = q_params + k_params + v_params + o_params

        for i in range(num_layers):
            layers.append(Layer(f"Block {i}", "block_start", 0, (), ""))
            layers.append(Layer(f"Block {i} - Norm 1", "normalization", ln_params, (hidden_size,)))
            total_params += ln_params

            attn_desc = f"GQA ({num_heads}Q/{num_kv_heads}KV, head_dim={head_dim})"
            if getattr(config, 'qk_norm', False):
                attn_desc += ", QK-Norm"
            layers.append(
                Layer(
                    f"Block {i} - Attention",
                    "attention",
                    attn_params,
                    (hidden_size, num_heads * head_dim),
                    attn_desc,
                )
            )
            total_params += attn_params

            layers.append(Layer(f"Block {i} - Norm 2", "normalization", ln_params, (hidden_size,)))
            total_params += ln_params

            if i < first_dense:
                ffn_params = 3 * hidden_size * dense_intermediate
                desc = f"Dense SwiGLU (Inter: {dense_intermediate})"
                ff_shape = (hidden_size, dense_intermediate)
            else:
                gate_params = hidden_size * num_experts
                routed_params = num_experts * (3 * hidden_size * moe_intermediate)
                shared_params = shared_experts * (3 * hidden_size * moe_intermediate)
                ffn_params = gate_params + routed_params + shared_params
                router_bits = []
                if router_sigmoid:
                    router_bits.append("sigmoid router")
                if route_norm:
                    router_bits.append("RouteNorm")
                if router_bias:
                    router_bits.append("expert bias")
                if router_scaling_factor:
                    router_bits.append(f"scale={router_scaling_factor:g}")
                router_suffix = f", {', '.join(router_bits)}" if router_bits else ""
                desc = (
                    f"MoE (Experts: {num_experts}, TopK: {active_experts}, "
                    f"Shared: {shared_experts}, Inter: {moe_intermediate}{router_suffix})"
                )
                ff_shape = (hidden_size, moe_intermediate)
            layers.append(Layer(f"Block {i} - FFN", "feedforward", ffn_params, ff_shape, desc))
            total_params += ffn_params
            layers.append(Layer(f"Block {i}", "block_end", 0, (), ""))

        layers.append(Layer("Final Norm", "normalization", hidden_size, (hidden_size,), "RMSNorm"))
        total_params += hidden_size

        if getattr(config, 'tie_word_embeddings', False):
            head_params = 0
            head_desc = "Tied"
        else:
            head_params = vocab_size * hidden_size
            head_desc = "Untied"
        layers.append(Layer("LM Head", "output", head_params, (hidden_size, vocab_size), head_desc))
        total_params += head_params

        if mtp_layers > 0:
            layers.append(Layer("MTP Head", "adapter", 0, (mtp_layers,), f"Next-N prediction layers: {mtp_layers}"))

        return _finalize_architecture(
            Architecture(
                model_type="hy_v3",
                arch_type="decoder-only",
                total_layers=num_layers,
                total_params=total_params,
                memory_fp16_gb=total_params * 2 / (1024**3),
                parameters={
                    "vocab_size": vocab_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "num_kv_heads": num_kv_heads,
                    "head_dim": head_dim,
                    "max_position": max_position,
                    "num_experts": num_experts,
                    "top_k_experts": active_experts,
                    "num_shared_experts": shared_experts,
                    "dense_prefix_layers": first_dense,
                    "mtp_layers": mtp_layers,
                    "rope_theta": rope_theta,
                    "expert_hidden_dim": expert_hidden_dim,
                    "route_norm": route_norm,
                    "router_sigmoid": router_sigmoid,
                    "router_bias": router_bias,
                    "output_router_logits": output_router_logits,
                    "router_scaling_factor": router_scaling_factor,
                },
                features=features,
                layers=layers,
            ),
            total_layers=num_layers,
        )
