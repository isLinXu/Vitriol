"""Base analyzer classes shared across model families."""

from typing import Any

from ..core import Architecture, Layer
from ._helpers import (
    _append_feature,
    _cfg_first,
    _cfg_get,
    _finalize_architecture,
    _head_dim,
    _infer_ffn_feature,
    _infer_norm_feature,
    _num_experts,
    _project_subconfig,
    _safe_float,
    _safe_int,
)


class ModelAnalyzer:
    """Base class for model-specific analysis."""
    def analyze(self, config: Any) -> Architecture:
        raise NotImplementedError

class TransformerAnalyzer(ModelAnalyzer):
    """Generic Transformer Analyzer."""
    def analyze(self, config: Any) -> Architecture:
        # Check if it is a multimodal or composite config (e.g. Kimi, Qwen-VL)
        _project_subconfig(config, "text_config")

        # Common attributes
        vocab_size = _safe_int(_cfg_first(config, ('vocab_size', 'padded_vocab_size'), 0), 0)
        hidden_size = _safe_int(_cfg_first(config, ('hidden_size', 'n_embd', 'd_model', 'dim', 'model_dim'), 0), 0)
        num_layers = _safe_int(_cfg_first(config, ('num_hidden_layers', 'n_layer', 'n_layers', 'num_layers', 'decoder_layers'), 0), 0)
        num_heads = _safe_int(_cfg_first(config, ('num_attention_heads', 'n_head', 'n_heads', 'num_heads'), 0), 0)
        num_kv_heads = _safe_int(_cfg_get(config, 'num_key_value_heads', num_heads), 0) or num_heads
        intermediate_size = _safe_int(
            _cfg_first(config, ('intermediate_size', 'ffn_hidden_size', 'd_ff', 'n_inner'), hidden_size * 4),
            hidden_size * 4,
        )
        num_heads = int(num_heads or 0)
        if num_heads <= 0 and hidden_size:
            num_heads = 1
        num_kv_heads = int(num_kv_heads or 0) or num_heads
        head_dim = _head_dim(config, hidden_size, num_heads)
        max_position = _safe_int(
            _cfg_get(config, 'max_position_embeddings', _cfg_get(config, 'n_positions', _cfg_get(config, 'n_ctx', 0))),
            0,
        )
        sliding_window = _safe_int(_cfg_get(config, 'sliding_window', 0), 0)
        rope_theta = _safe_float(_cfg_get(config, 'rope_theta', 0.0), 0.0)
        if rope_theta <= 0.0:
            rope_parameters = _cfg_get(config, 'rope_parameters', None)
            if isinstance(rope_parameters, dict):
                rope_theta = _safe_float(rope_parameters.get('rope_theta', 0.0), 0.0)
        num_experts = _num_experts(config)
        top_k_experts = _safe_int(_cfg_get(config, 'num_experts_per_tok', 0), 0)

        # Determine Architecture Type
        arch_type = "decoder-only"
        model_type = str(getattr(config, 'model_type', '') or '').lower()
        if getattr(config, 'is_encoder_decoder', False):
            arch_type = "encoder-decoder"
        elif any(k in model_type for k in ['t5', 'bart', 'bert', 'encoder']):
            if 'bert' in model_type or 'encoder' in model_type:
                arch_type = "encoder-only"
            else:
                arch_type = "encoder-decoder"

        features = self._detect_features(config, num_heads, num_kv_heads)
        _append_feature(features, _infer_norm_feature(config))
        _append_feature(features, _infer_ffn_feature(config))
        if "GQA" not in features and "MQA" not in features:
            _append_feature(features, "MHA")
        if arch_type == "encoder-only":
            _append_feature(features, "Bidirectional")
        elif arch_type == "encoder-decoder":
            _append_feature(features, "CrossAttn")
        else:
            _append_feature(features, "Causal")

        layers = []
        total_params = 0

        # 1. Embedding
        emb_params = vocab_size * hidden_size
        layers.append(Layer("Token Embedding", "embedding", emb_params, (vocab_size, hidden_size), f"Vocab: {vocab_size}, Dim: {hidden_size}"))
        total_params += emb_params

        # 2. Layers
        for i in range(num_layers):
            layers.append(Layer(f"Block {i}", "block_start", 0, (), ""))

            # LayerNorm 1
            ln_params = hidden_size # approx
            layers.append(Layer(f"Block {i} - Norm 1", "normalization", ln_params, (hidden_size,)))
            total_params += ln_params

            # Attention
            q_params = hidden_size * hidden_size
            k_params = hidden_size * (num_kv_heads * head_dim)
            v_params = hidden_size * (num_kv_heads * head_dim)
            o_params = hidden_size * hidden_size
            attn_params = q_params + k_params + v_params + o_params

            layers.append(Layer(f"Block {i} - Attention", "attention", attn_params, (hidden_size, hidden_size),
                                f"Heads: {num_heads}, KV Heads: {num_kv_heads}"))
            total_params += attn_params

            # LayerNorm 2
            layers.append(Layer(f"Block {i} - Norm 2", "normalization", ln_params, (hidden_size,)))
            total_params += ln_params

            # FFN / MoE
            block_ffn_params, desc = self._calculate_ffn_params(config, hidden_size, intermediate_size, features)
            layers.append(Layer(f"Block {i} - FFN", "feedforward", block_ffn_params, (hidden_size, intermediate_size), desc))
            total_params += block_ffn_params

            layers.append(Layer(f"Block {i}", "block_end", 0, (), ""))

        # 3. Final Norm
        layers.append(Layer("Final Norm", "normalization", hidden_size, (hidden_size,)))
        total_params += hidden_size

        # 4. Output Head
        if getattr(config, 'tie_word_embeddings', True):
             head_params = 0
             desc = "Tied"
        else:
             head_params = vocab_size * hidden_size
             desc = "Untied"
        layers.append(Layer("LM Head", "output", head_params, (hidden_size, vocab_size), desc))
        total_params += head_params

        return _finalize_architecture(
            Architecture(
                model_type=getattr(config, 'model_type', 'unknown'),
                arch_type=arch_type,
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
                    "intermediate_size": intermediate_size,
                    "max_position": max_position,
                    "sliding_window": sliding_window,
                    "num_experts": num_experts,
                    "top_k_experts": top_k_experts,
                    "rope_theta": rope_theta,
                },
                features=features,
                layers=layers,
            ),
            total_layers=num_layers,
            encoder_layers=num_layers if arch_type == "encoder-only" else 0,
        )

    def _detect_features(self, config, num_heads, num_kv_heads):
        features = []
        max_position = _safe_int(
            _cfg_get(config, 'max_position_embeddings', _cfg_get(config, 'n_positions', _cfg_get(config, 'n_ctx', 0))),
            0,
        )
        if num_heads > 1 and num_kv_heads == 1:
            features.append("MQA")
        elif num_kv_heads != num_heads:
            features.append("GQA")
        if _cfg_get(config, 'rope_scaling', None) or _cfg_get(config, 'rope_parameters', None) or _cfg_get(config, 'rope_theta', None):
            features.append("RoPE")
        if _cfg_get(config, 'sliding_window', None):
            features.append("Sliding Window")
        if max_position >= 32768:
            features.append("Long Context")
        if 'moe' in getattr(config, 'model_type', '').lower() or _safe_int(_cfg_get(config, 'num_experts_per_tok', 0), 0) > 0 or _num_experts(config) > 1:
            features.append("MoE")
        return features

    def _calculate_ffn_params(self, config, hidden, intermediate, features):
        num_experts = _num_experts(config)
        active_experts = _safe_int(_cfg_get(config, 'num_experts_per_tok', 2), 2)
        shared_intermediate = _safe_int(_cfg_get(config, 'shared_expert_intermediate_size', 0), 0)
        if "MoE" in features and num_experts > 0:
            # Gate + Experts
            gate = hidden * num_experts
            # SwiGLU 3 matrices
            expert = num_experts * (3 * hidden * intermediate)
            shared = 3 * hidden * shared_intermediate if shared_intermediate > 0 else 0
            desc = f"Experts: {num_experts}, Active: {active_experts}"
            if shared_intermediate > 0:
                desc += f", Shared Inter: {shared_intermediate}"
            return gate + expert + shared, desc
        else:
            # SwiGLU standard
            return 3 * hidden * intermediate, f"Inter: {intermediate}"
