"""DeepSeek-V3 family analyzers (incl. Kimi variant)."""

from typing import Any

from ..core import Architecture, Layer
from ._helpers import (
    _append_feature,
    _as_int_list,
    _cfg_get,
    _finalize_architecture,
    _head_dim,
    _infer_ffn_feature,
    _infer_norm_feature,
    _project_subconfig,
    _safe_float,
    _safe_int,
)
from .base import TransformerAnalyzer


class DeepSeekAnalyzer(TransformerAnalyzer):
    """DeepSeek analyzer covering MLA, MoE, and DeepSeek-V4 CSA/HCA metadata."""

    def _detect_features(self, config, num_heads, num_kv_heads):
        features = super()._detect_features(config, num_heads, num_kv_heads)
        model_type = str(_cfg_get(config, 'model_type', '') or '').lower()
        quant_cfg = _cfg_get(config, 'quantization_config', {}) or {}
        rope_scaling = _cfg_get(config, 'rope_scaling', {}) or {}
        compress_ratios = _as_int_list(_cfg_get(config, 'compress_ratios', []))

        if model_type in ['deepseek_v3', 'kimi_k25']:
            _append_feature(features, "MLA")
        elif model_type == 'deepseek_v4':
            _append_feature(features, "DeepSeek-V4")
            _append_feature(features, "CSA/HCA")
            _append_feature(features, "mHC")
            if _safe_int(_cfg_get(config, 'num_hash_layers', 0), 0) > 0:
                _append_feature(features, "Hash Attention")
            if any(ratio > 0 for ratio in compress_ratios):
                _append_feature(features, "Compressed Attention")
            if _safe_float(_cfg_get(config, 'compress_rope_theta', 0.0), 0.0) > 0.0:
                _append_feature(features, "Compressed RoPE")
            if str(_cfg_get(rope_scaling, 'type', '') or '').lower() == 'yarn':
                _append_feature(features, "YARN RoPE")
            if str(_cfg_get(quant_cfg, 'quant_method', '') or '').lower() == 'fp8':
                _append_feature(features, "FP8")
            if _safe_int(_cfg_get(config, 'num_nextn_predict_layers', 0), 0) > 0:
                _append_feature(features, f"MTP ({_safe_int(_cfg_get(config, 'num_nextn_predict_layers', 0), 0)})")

        _append_feature(features, _infer_norm_feature(config))
        _append_feature(features, _infer_ffn_feature(config))
        return features

    def _deepseek_v4_attention_params(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        q_lora_rank: int,
        o_lora_rank: int,
        index_n_heads: int,
        index_head_dim: int,
    ) -> int:
        q_out = num_heads * head_dim
        kv_out = num_kv_heads * head_dim
        q_params = hidden_size * q_lora_rank + q_lora_rank * q_out if q_lora_rank > 0 else hidden_size * q_out
        k_params = hidden_size * kv_out
        v_params = hidden_size * kv_out
        o_params = q_out * o_lora_rank + o_lora_rank * hidden_size if o_lora_rank > 0 else q_out * hidden_size
        index_params = hidden_size * (index_n_heads * index_head_dim) if index_n_heads and index_head_dim else 0
        return int(q_params + k_params + v_params + o_params + index_params)

    def _deepseek_v4_attention_desc(self, layer_idx: int, compress_ratios: list[int], num_hash_layers: int, sliding_window: int) -> str:
        if layer_idx < num_hash_layers:
            desc = "Hash Attention"
        else:
            ratio = compress_ratios[layer_idx] if layer_idx < len(compress_ratios) else 0
            desc = f"CSA/HCA compressed x{ratio}" if ratio > 0 else "CSA/HCA full"
        if sliding_window:
            desc += f", SWA={sliding_window}"
        return desc

    def analyze(self, config: Any) -> Architecture:
        _project_subconfig(config, "text_config")

        vocab_size = _safe_int(_cfg_get(config, 'vocab_size', 0), 0)
        hidden_size = _safe_int(_cfg_get(config, 'hidden_size', 0), 0)
        num_layers = _safe_int(_cfg_get(config, 'num_hidden_layers', 0), 0)
        num_heads = _safe_int(_cfg_get(config, 'num_attention_heads', 0), 0)
        if num_heads <= 0 and hidden_size:
            num_heads = 1
        num_kv_heads = _safe_int(_cfg_get(config, 'num_key_value_heads', num_heads), 0) or num_heads
        head_dim = _head_dim(config, hidden_size, num_heads)
        max_position = _safe_int(_cfg_get(config, 'max_position_embeddings', 0), 0)
        sliding_window = _safe_int(_cfg_get(config, 'sliding_window', 0), 0)

        first_k_dense = _safe_int(_cfg_get(config, 'first_k_dense_replace', 0), 0)
        moe_inter_size = _safe_int(_cfg_get(config, 'moe_intermediate_size', _cfg_get(config, 'intermediate_size', 0)), 0)
        dense_inter_size = _safe_int(_cfg_get(config, 'intermediate_size', moe_inter_size), moe_inter_size)

        n_routed = _safe_int(_cfg_get(config, 'n_routed_experts', _cfg_get(config, 'num_experts', 0)), 0)
        n_shared = _safe_int(_cfg_get(config, 'n_shared_experts', _cfg_get(config, 'num_shared_experts', 0)), 0)
        active_experts = _safe_int(_cfg_get(config, 'num_experts_per_tok', 0), 0)

        model_type = str(_cfg_get(config, 'model_type', 'unknown') or 'unknown').lower()
        is_v4 = model_type == 'deepseek_v4'
        compress_ratios = _as_int_list(_cfg_get(config, 'compress_ratios', []))
        num_hash_layers = _safe_int(_cfg_get(config, 'num_hash_layers', 0), 0)
        compressed_layers = sum(
            1
            for idx in range(num_hash_layers, num_layers)
            if idx < len(compress_ratios) and compress_ratios[idx] > 0
        )
        q_lora_rank = _safe_int(_cfg_get(config, 'q_lora_rank', 0), 0)
        o_lora_rank = _safe_int(_cfg_get(config, 'o_lora_rank', 0), 0)
        o_groups = _safe_int(_cfg_get(config, 'o_groups', 0), 0)
        qk_rope_head_dim = _safe_int(_cfg_get(config, 'qk_rope_head_dim', 0), 0)
        index_n_heads = _safe_int(_cfg_get(config, 'index_n_heads', 0), 0)
        index_head_dim = _safe_int(_cfg_get(config, 'index_head_dim', 0), 0)
        index_topk = _safe_int(_cfg_get(config, 'index_topk', 0), 0)
        compress_rope_theta = _safe_float(_cfg_get(config, 'compress_rope_theta', 0.0), 0.0)
        mtp_layers = _safe_int(_cfg_get(config, 'num_nextn_predict_layers', 0), 0)
        topk_method = _cfg_get(config, 'topk_method', '')
        scoring_func = _cfg_get(config, 'scoring_func', '')
        routed_scaling = _safe_float(_cfg_get(config, 'routed_scaling_factor', 0.0), 0.0)
        swiglu_limit = _safe_float(_cfg_get(config, 'swiglu_limit', 0.0), 0.0)
        quant_cfg = _cfg_get(config, 'quantization_config', {}) or {}
        quant_method = _cfg_get(quant_cfg, 'quant_method', None)
        weight_block_size = _cfg_get(quant_cfg, 'weight_block_size', None)
        layer_types = []
        if is_v4 and num_layers > 0:
            for idx in range(num_layers):
                ratio = compress_ratios[idx] if idx < len(compress_ratios) else 0
                if idx < num_hash_layers:
                    layer_types.append("hash_attention")
                elif ratio > 0:
                    layer_types.append("compressed_attention")
                elif sliding_window:
                    layer_types.append("sliding_window")
                else:
                    layer_types.append("full_attention")

        features = self._detect_features(config, num_heads, num_kv_heads)

        layers = []
        total_params = 0
        activated_params = 0

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
        activated_params += emb_params

        ln_params = hidden_size
        for i in range(num_layers):
            layers.append(Layer(f"Block {i}", "block_start", 0, (), ""))
            layers.append(Layer(f"Block {i} - Norm 1", "normalization", ln_params, (hidden_size,)))
            total_params += ln_params
            activated_params += ln_params

            if is_v4:
                attn_params = self._deepseek_v4_attention_params(
                    hidden_size,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    q_lora_rank,
                    o_lora_rank,
                    index_n_heads,
                    index_head_dim,
                )
                attn_desc = self._deepseek_v4_attention_desc(i, compress_ratios, num_hash_layers, sliding_window)
            else:
                q_params = hidden_size * (num_heads * head_dim)
                k_params = hidden_size * (num_kv_heads * head_dim)
                v_params = hidden_size * (num_kv_heads * head_dim)
                o_params = (num_heads * head_dim) * hidden_size
                attn_params = q_params + k_params + v_params + o_params
                attn_desc = f"MLA/GQA ({num_heads}Q/{num_kv_heads}KV, head_dim={head_dim})"

            layers.append(Layer(f"Block {i} - Attention", "attention", attn_params, (hidden_size, num_heads * head_dim), attn_desc))
            total_params += attn_params
            activated_params += attn_params

            layers.append(Layer(f"Block {i} - Norm 2", "normalization", ln_params, (hidden_size,)))
            total_params += ln_params
            activated_params += ln_params

            if i < first_k_dense:
                ffn_params = 3 * hidden_size * dense_inter_size
                active_ffn_params = ffn_params
                desc = f"Dense SwiGLU (Inter: {dense_inter_size})"
            else:
                gate_params = hidden_size * n_routed
                routed_params = n_routed * (3 * hidden_size * moe_inter_size)
                shared_params = n_shared * (3 * hidden_size * moe_inter_size)
                ffn_params = gate_params + routed_params + shared_params
                active_ffn_params = gate_params + (active_experts * 3 * hidden_size * moe_inter_size) + shared_params
                desc = f"MoE (Routed: {n_routed}, Active: {active_experts}, Shared: {n_shared})"
                if topk_method:
                    desc += f", topk={topk_method}"
                _append_feature(features, "MoE")

            layers.append(Layer(f"Block {i} - FFN", "feedforward", ffn_params, (hidden_size, moe_inter_size or dense_inter_size), desc))
            total_params += ffn_params
            activated_params += active_ffn_params
            layers.append(Layer(f"Block {i}", "block_end", 0, (), ""))

        layers.append(Layer("Final Norm", "normalization", hidden_size, (hidden_size,), "RMSNorm"))
        total_params += hidden_size
        activated_params += hidden_size

        if _cfg_get(config, 'tie_word_embeddings', False):
            head_params = 0
            desc = "Tied"
        else:
            head_params = vocab_size * hidden_size
            desc = "Untied"
        layers.append(Layer("LM Head", "output", head_params, (hidden_size, vocab_size), desc))
        total_params += head_params
        activated_params += head_params

        if mtp_layers > 0:
            layers.append(Layer("MTP Head", "adapter", 0, (mtp_layers,), f"Next-N prediction layers: {mtp_layers}"))

        parameters = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "max_position": max_position,
            "sliding_window": sliding_window,
            "activated_params": activated_params,
            "num_experts": n_routed,
            "top_k_experts": active_experts,
            "num_shared_experts": n_shared,
            "moe_intermediate_size": moe_inter_size,
            "dense_prefix_layers": first_k_dense,
            "mtp_layers": mtp_layers,
            "layer_types": layer_types,
        }
        if is_v4:
            parameters.update(
                {
                    "num_hash_layers": num_hash_layers,
                    "compressed_attention_layers": compressed_layers,
                    "compress_ratios": compress_ratios,
                    "compress_rope_theta": compress_rope_theta,
                    "index_n_heads": index_n_heads,
                    "index_head_dim": index_head_dim,
                    "index_topk": index_topk,
                    "q_lora_rank": q_lora_rank,
                    "o_lora_rank": o_lora_rank,
                    "o_groups": o_groups,
                    "qk_rope_head_dim": qk_rope_head_dim,
                    "topk_method": topk_method,
                    "scoring_func": scoring_func,
                    "routed_scaling_factor": routed_scaling,
                    "swiglu_limit": swiglu_limit,
                    "quant_method": quant_method,
                    "weight_block_size": weight_block_size,
                }
            )

        return _finalize_architecture(
            Architecture(
                model_type=model_type,
                arch_type="decoder-only",
                total_layers=num_layers,
                total_params=total_params,
                memory_fp16_gb=total_params * 2 / (1024**3),
                parameters=parameters,
                features=features,
                layers=layers,
            ),
            total_layers=num_layers,
        )

class KimiAnalyzer(DeepSeekAnalyzer):
    """Kimi K2.5 Analyzer (DeepSeek V3 based)."""
    pass
