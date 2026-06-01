"""Classic decoder-only families (GPT-2/NeoX, Bloom, StarCoder, Falcon, OPT)."""

from typing import Any

from ..core import Architecture, Layer
from ._helpers import (
    _append_feature,
    _cfg_get,
    _finalize_architecture,
    _head_dim,
    _infer_ffn_feature,
    _infer_norm_feature,
    _safe_int,
)
from .base import ModelAnalyzer, TransformerAnalyzer


class BloomAnalyzer(ModelAnalyzer):
    """BLOOM decoder analyzer with ALiBi-aware metadata."""

    def analyze(self, config: Any) -> Architecture:
        vocab_size = _safe_int(_cfg_get(config, "vocab_size", 250880), 250880)
        hidden_size = _safe_int(_cfg_get(config, "hidden_size", 1024), 1024)
        num_layers = _safe_int(_cfg_get(config, "num_hidden_layers", 24), 24)
        num_heads = _safe_int(_cfg_get(config, "num_attention_heads", 16), 16)
        intermediate_size = _safe_int(
            _cfg_get(config, "intermediate_size", _cfg_get(config, "n_inner", hidden_size * 4)),
            hidden_size * 4,
        )
        max_position = _safe_int(
            _cfg_get(config, "seq_length", _cfg_get(config, "max_position_embeddings", 2048)),
            2048,
        )
        head_dim = _head_dim(config, hidden_size, num_heads)

        features = ["Bloom", "LayerNorm", "GELU", "Causal", "MHA"]
        if _cfg_get(config, "alibi", True):
            _append_feature(features, "ALiBi")
        if max_position >= 32768:
            _append_feature(features, "Long Context")

        layers = []
        total_params = 0
        emb_params = vocab_size * hidden_size
        layers.append(Layer("Token Embedding", "embedding", emb_params, (vocab_size, hidden_size), f"Vocab: {vocab_size}"))
        total_params += emb_params

        attn_params = 4 * hidden_size * hidden_size
        ffn_params = 2 * hidden_size * intermediate_size
        ln_params = 2 * hidden_size
        for i in range(num_layers):
            layers.append(Layer(f"Block {i}", "block_start", 0, (), ""))
            layers.append(Layer(f"Block {i} - Norm 1", "normalization", ln_params, (hidden_size,), "LayerNorm"))
            total_params += ln_params
            layers.append(Layer(f"Block {i} - Attention", "attention", attn_params, (num_heads, head_dim), "ALiBi MHA"))
            total_params += attn_params
            layers.append(Layer(f"Block {i} - Norm 2", "normalization", ln_params, (hidden_size,), "LayerNorm"))
            total_params += ln_params
            layers.append(Layer(f"Block {i} - FFN", "feedforward", ffn_params, (hidden_size, intermediate_size), "GELU"))
            total_params += ffn_params
            layers.append(Layer(f"Block {i}", "block_end", 0, (), ""))

        layers.append(Layer("Final Norm", "normalization", ln_params, (hidden_size,), "LayerNorm"))
        total_params += ln_params

        if _cfg_get(config, "tie_word_embeddings", True):
            head_params = 0
            head_desc = "Tied"
        else:
            head_params = vocab_size * hidden_size
            head_desc = "Untied"
        layers.append(Layer("LM Head", "output", head_params, (hidden_size, vocab_size), head_desc))
        total_params += head_params

        return _finalize_architecture(
            Architecture(
                model_type=getattr(config, "model_type", "bloom"),
                arch_type="decoder-only",
                total_layers=num_layers,
                total_params=total_params,
                memory_fp16_gb=total_params * 2 / (1024**3),
                parameters={
                    "vocab_size": vocab_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "num_kv_heads": num_heads,
                    "head_dim": head_dim,
                    "intermediate_size": intermediate_size,
                    "max_position": max_position,
                },
                features=features,
                layers=layers,
            ),
            total_layers=num_layers,
        )


class GPTNeoXAnalyzer(ModelAnalyzer):
    """GPT-NeoX decoder analyzer with RoPE / parallel residual metadata."""

    def analyze(self, config: Any) -> Architecture:
        vocab_size = _safe_int(_cfg_get(config, "vocab_size", 50432), 50432)
        hidden_size = _safe_int(_cfg_get(config, "hidden_size", 6144), 6144)
        num_layers = _safe_int(_cfg_get(config, "num_hidden_layers", 44), 44)
        num_heads = _safe_int(_cfg_get(config, "num_attention_heads", 64), 64)
        num_kv_heads = _safe_int(
            _cfg_get(config, "num_key_value_heads", _cfg_get(config, "num_kv_heads", num_heads)),
            num_heads,
        ) or num_heads
        intermediate_size = _safe_int(_cfg_get(config, "intermediate_size", hidden_size * 4), hidden_size * 4)
        max_position = _safe_int(_cfg_get(config, "max_position_embeddings", 2048), 2048)
        head_dim = _head_dim(config, hidden_size, num_heads)

        features = ["GPT-NeoX", "LayerNorm", "GELU", "Causal"]
        if _cfg_get(config, "rotary_pct", None) is not None or _cfg_get(config, "rope_theta", None):
            _append_feature(features, "RoPE")
        if _cfg_get(config, "use_parallel_residual", False):
            _append_feature(features, "Parallel Residual")
        if num_heads > 1 and num_kv_heads == 1:
            _append_feature(features, "MQA")
        elif num_kv_heads != num_heads:
            _append_feature(features, "GQA")
        else:
            _append_feature(features, "MHA")
        if max_position >= 32768:
            _append_feature(features, "Long Context")

        layers = []
        total_params = 0
        emb_params = vocab_size * hidden_size
        layers.append(Layer("Token Embedding", "embedding", emb_params, (vocab_size, hidden_size), f"Vocab: {vocab_size}"))
        total_params += emb_params

        attn_params = (
            hidden_size * (num_heads * head_dim)
            + hidden_size * (num_kv_heads * head_dim)
            + hidden_size * (num_kv_heads * head_dim)
            + (num_heads * head_dim) * hidden_size
        )
        ffn_params = 2 * hidden_size * intermediate_size
        ln_params = 2 * hidden_size
        for i in range(num_layers):
            layers.append(Layer(f"Block {i}", "block_start", 0, (), ""))
            layers.append(Layer(f"Block {i} - Norm 1", "normalization", ln_params, (hidden_size,), "LayerNorm"))
            total_params += ln_params
            attn_desc = "RoPE"
            if "Parallel Residual" in features:
                attn_desc += " + Parallel Residual"
            layers.append(Layer(f"Block {i} - Attention", "attention", attn_params, (num_heads, head_dim), attn_desc))
            total_params += attn_params
            layers.append(Layer(f"Block {i} - Norm 2", "normalization", ln_params, (hidden_size,), "LayerNorm"))
            total_params += ln_params
            layers.append(Layer(f"Block {i} - FFN", "feedforward", ffn_params, (hidden_size, intermediate_size), "GELU"))
            total_params += ffn_params
            layers.append(Layer(f"Block {i}", "block_end", 0, (), ""))

        layers.append(Layer("Final Norm", "normalization", ln_params, (hidden_size,), "LayerNorm"))
        total_params += ln_params

        if _cfg_get(config, "tie_word_embeddings", False):
            head_params = 0
            head_desc = "Tied"
        else:
            head_params = vocab_size * hidden_size
            head_desc = "Untied"
        layers.append(Layer("LM Head", "output", head_params, (hidden_size, vocab_size), head_desc))
        total_params += head_params

        return _finalize_architecture(
            Architecture(
                model_type=getattr(config, "model_type", "gpt_neox"),
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
                    "intermediate_size": intermediate_size,
                    "max_position": max_position,
                },
                features=features,
                layers=layers,
            ),
            total_layers=num_layers,
        )


class StarCoderAnalyzer(ModelAnalyzer):
    """StarCoder / GPT-BigCode analyzer."""

    def analyze(self, config: Any) -> Architecture:
        vocab_size = _safe_int(_cfg_get(config, "vocab_size", 49152), 49152)
        hidden_size = _safe_int(_cfg_get(config, "hidden_size", _cfg_get(config, "n_embd", 6144)), 6144)
        num_layers = _safe_int(_cfg_get(config, "num_hidden_layers", _cfg_get(config, "n_layer", 40)), 40)
        num_heads = _safe_int(_cfg_get(config, "num_attention_heads", _cfg_get(config, "n_head", 48)), 48)
        intermediate_size = _safe_int(_cfg_get(config, "n_inner", _cfg_get(config, "intermediate_size", hidden_size * 4)), hidden_size * 4)
        max_position = _safe_int(_cfg_get(config, "max_position_embeddings", _cfg_get(config, "n_positions", 8192)), 8192)
        multi_query = bool(_cfg_get(config, "multi_query", False))
        num_kv_heads = 1 if multi_query else (
            _safe_int(_cfg_get(config, "num_key_value_heads", _cfg_get(config, "num_kv_heads", num_heads)), num_heads) or num_heads
        )
        head_dim = _head_dim(config, hidden_size, num_heads)

        features = ["StarCoder", "LearnedPos", "LayerNorm", "GELU", "Causal"]
        if num_heads > 1 and num_kv_heads == 1:
            _append_feature(features, "MQA")
        elif num_kv_heads != num_heads:
            _append_feature(features, "GQA")
        else:
            _append_feature(features, "MHA")

        layers = []
        total_params = 0
        token_emb = vocab_size * hidden_size
        pos_emb = max_position * hidden_size
        layers.append(Layer("Token Embedding", "embedding", token_emb, (vocab_size, hidden_size), f"Vocab: {vocab_size}"))
        layers.append(Layer("Position Embedding", "embedding", pos_emb, (max_position, hidden_size), f"Max pos: {max_position}"))
        total_params += token_emb + pos_emb

        attn_params = (
            hidden_size * (num_heads * head_dim)
            + hidden_size * (num_kv_heads * head_dim)
            + hidden_size * (num_kv_heads * head_dim)
            + (num_heads * head_dim) * hidden_size
        )
        ffn_params = 2 * hidden_size * intermediate_size
        ln_params = 2 * hidden_size
        for i in range(num_layers):
            layers.append(Layer(f"Block {i}", "block_start", 0, (), ""))
            layers.append(Layer(f"Block {i} - Norm 1", "normalization", ln_params, (hidden_size,), "LayerNorm"))
            total_params += ln_params
            attn_desc = "MQA" if num_kv_heads == 1 and num_heads > 1 else "Causal Attention"
            layers.append(Layer(f"Block {i} - Attention", "attention", attn_params, (num_heads, head_dim), attn_desc))
            total_params += attn_params
            layers.append(Layer(f"Block {i} - Norm 2", "normalization", ln_params, (hidden_size,), "LayerNorm"))
            total_params += ln_params
            layers.append(Layer(f"Block {i} - FFN", "feedforward", ffn_params, (hidden_size, intermediate_size), "GELU"))
            total_params += ffn_params
            layers.append(Layer(f"Block {i}", "block_end", 0, (), ""))

        layers.append(Layer("Final Norm", "normalization", ln_params, (hidden_size,), "LayerNorm"))
        total_params += ln_params

        if _cfg_get(config, "tie_word_embeddings", True):
            head_params = 0
            head_desc = "Tied"
        else:
            head_params = vocab_size * hidden_size
            head_desc = "Untied"
        layers.append(Layer("LM Head", "output", head_params, (hidden_size, vocab_size), head_desc))
        total_params += head_params

        return _finalize_architecture(
            Architecture(
                model_type=getattr(config, "model_type", "gpt_bigcode"),
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
                    "intermediate_size": intermediate_size,
                    "max_position": max_position,
                },
                features=features,
                layers=layers,
            ),
            total_layers=num_layers,
        )


class StarCoder2Analyzer(ModelAnalyzer):
    """StarCoder2 decoder analyzer with GQA / RoPE metadata."""

    def analyze(self, config: Any) -> Architecture:
        vocab_size = _safe_int(_cfg_get(config, "vocab_size", 49152), 49152)
        hidden_size = _safe_int(_cfg_get(config, "hidden_size", 3072), 3072)
        num_layers = _safe_int(_cfg_get(config, "num_hidden_layers", 30), 30)
        num_heads = _safe_int(_cfg_get(config, "num_attention_heads", 24), 24)
        num_kv_heads = _safe_int(_cfg_get(config, "num_key_value_heads", num_heads), num_heads) or num_heads
        intermediate_size = _safe_int(_cfg_get(config, "intermediate_size", hidden_size * 4), hidden_size * 4)
        max_position = _safe_int(_cfg_get(config, "max_position_embeddings", 16384), 16384)
        sliding_window = _safe_int(_cfg_get(config, "sliding_window", 0), 0)
        head_dim = _head_dim(config, hidden_size, num_heads)

        features = ["StarCoder2", "Causal"]
        _append_feature(features, _infer_norm_feature(config) or "LayerNorm")
        _append_feature(features, _infer_ffn_feature(config) or "GELU")
        if _cfg_get(config, "rope_theta", None) is not None or _cfg_get(config, "rope_scaling", None):
            _append_feature(features, "RoPE")
        if num_heads > 1 and num_kv_heads == 1:
            _append_feature(features, "MQA")
        elif num_kv_heads != num_heads:
            _append_feature(features, "GQA")
        else:
            _append_feature(features, "MHA")
        if sliding_window > 0:
            _append_feature(features, "Sliding Window")
        if max_position >= 32768:
            _append_feature(features, "Long Context")

        layers = []
        total_params = 0
        emb_params = vocab_size * hidden_size
        layers.append(Layer("Token Embedding", "embedding", emb_params, (vocab_size, hidden_size), f"Vocab: {vocab_size}"))
        total_params += emb_params

        attn_params = (
            hidden_size * (num_heads * head_dim)
            + hidden_size * (num_kv_heads * head_dim)
            + hidden_size * (num_kv_heads * head_dim)
            + (num_heads * head_dim) * hidden_size
        )
        ffn_params = 2 * hidden_size * intermediate_size
        norm_desc = "RMSNorm" if "RMSNorm" in features else "LayerNorm"
        ln_params = hidden_size if norm_desc == "RMSNorm" else 2 * hidden_size
        ffn_desc = "GELU"
        if "SwiGLU" in features:
            ffn_desc = "SwiGLU"
        elif "GeGLU" in features:
            ffn_desc = "GeGLU"
        for i in range(num_layers):
            layers.append(Layer(f"Block {i}", "block_start", 0, (), ""))
            layers.append(Layer(f"Block {i} - Norm 1", "normalization", ln_params, (hidden_size,), norm_desc))
            total_params += ln_params
            attn_desc = "RoPE Attention" if "RoPE" in features else "Causal Attention"
            layers.append(Layer(f"Block {i} - Attention", "attention", attn_params, (num_heads, head_dim), attn_desc))
            total_params += attn_params
            layers.append(Layer(f"Block {i} - Norm 2", "normalization", ln_params, (hidden_size,), norm_desc))
            total_params += ln_params
            layers.append(Layer(f"Block {i} - FFN", "feedforward", ffn_params, (hidden_size, intermediate_size), ffn_desc))
            total_params += ffn_params
            layers.append(Layer(f"Block {i}", "block_end", 0, (), ""))

        layers.append(Layer("Final Norm", "normalization", ln_params, (hidden_size,), norm_desc))
        total_params += ln_params

        if _cfg_get(config, "tie_word_embeddings", False):
            head_params = 0
            head_desc = "Tied"
        else:
            head_params = vocab_size * hidden_size
            head_desc = "Untied"
        layers.append(Layer("LM Head", "output", head_params, (hidden_size, vocab_size), head_desc))
        total_params += head_params

        return _finalize_architecture(
            Architecture(
                model_type=getattr(config, "model_type", "starcoder2"),
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
                    "intermediate_size": intermediate_size,
                    "max_position": max_position,
                    "sliding_window": sliding_window,
                },
                features=features,
                layers=layers,
            ),
            total_layers=num_layers,
        )

class GPT2Analyzer(TransformerAnalyzer):
    """GPT-2 Specific Analyzer."""

    def analyze(self, config: Any) -> Architecture:
        # GPT-2 uses different config names
        vocab_size = getattr(config, 'vocab_size', 0)
        hidden_size = getattr(config, 'n_embd', 768)
        num_layers = getattr(config, 'n_layer', 12)
        num_heads = getattr(config, 'n_head', 12)
        max_pos = getattr(config, 'n_ctx', 1024)

        # Intermediate size in GPT-2 is typically 4 * hidden
        intermediate_size = getattr(config, 'n_inner', None)
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size

        features = ["GPT-2", "Absolute Positional Embedding"]

        layers = []
        total_params = 0

        # 1. Embedding (Word + Position)
        wte_params = vocab_size * hidden_size
        wpe_params = max_pos * hidden_size
        emb_params = wte_params + wpe_params

        layers.append(Layer("Token Embedding", "embedding", wte_params, (vocab_size, hidden_size), f"Vocab: {vocab_size}"))
        layers.append(Layer("Position Embedding", "embedding", wpe_params, (max_pos, hidden_size), f"Ctx: {max_pos}"))
        total_params += emb_params

        # 2. Layers
        for i in range(num_layers):
            layers.append(Layer(f"Block {i}", "block_start", 0, (), ""))

            # LN 1
            ln_params = 2 * hidden_size # weight + bias
            layers.append(Layer(f"Block {i} - Norm 1", "normalization", ln_params, (hidden_size,)))
            total_params += ln_params

            # Attention (Conv1D implementation in HF, but logic same)
            # c_attn: 3 * hidden (Q,K,V) -> hidden * (3*hidden) + bias
            # c_proj: hidden -> hidden
            attn_weight = hidden_size * (3 * hidden_size)
            attn_bias = 3 * hidden_size
            proj_weight = hidden_size * hidden_size
            proj_bias = hidden_size

            attn_params = attn_weight + attn_bias + proj_weight + proj_bias
            layers.append(Layer(f"Block {i} - Attention", "attention", attn_params, (hidden_size, hidden_size), "MHA"))
            total_params += attn_params

            # LN 2
            layers.append(Layer(f"Block {i} - Norm 2", "normalization", ln_params, (hidden_size,)))
            total_params += ln_params

            # MLP
            # c_fc: hidden -> inner
            # c_proj: inner -> hidden
            fc_weight = hidden_size * intermediate_size
            fc_bias = intermediate_size
            proj_weight = intermediate_size * hidden_size
            proj_bias = hidden_size

            mlp_params = fc_weight + fc_bias + proj_weight + proj_bias
            layers.append(Layer(f"Block {i} - MLP", "feedforward", mlp_params, (hidden_size, intermediate_size), "GELU"))
            total_params += mlp_params

            layers.append(Layer(f"Block {i}", "block_end", 0, (), ""))

        # 3. Final Norm
        ln_params = 2 * hidden_size
        layers.append(Layer("Final Norm", "normalization", ln_params, (hidden_size,)))
        total_params += ln_params

        # 4. Head
        # GPT-2 ties weights
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
                model_type=getattr(config, 'model_type', 'gpt2'),
                arch_type="decoder-only",
                total_layers=num_layers,
                total_params=total_params,
                memory_fp16_gb=total_params * 4 / (1024**3), # GPT-2 often float32 default? use 4 bytes
                parameters={
                    "vocab_size": vocab_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "num_heads": num_heads
                },
                features=features,
                layers=layers,
            ),
            total_layers=num_layers,
        )


class FalconAnalyzer(ModelAnalyzer):
    """Falcon family analyzer with ALiBi / MQA metadata."""

    def analyze(self, config: Any) -> Architecture:
        vocab_size = _safe_int(_cfg_get(config, "vocab_size", 65024), 65024)
        hidden_size = _safe_int(_cfg_get(config, "hidden_size", 4544), 4544)
        num_layers = _safe_int(_cfg_get(config, "num_hidden_layers", 32), 32)
        num_heads = _safe_int(_cfg_get(config, "num_attention_heads", 71), 71)
        multi_query = bool(_cfg_get(config, "multi_query", False))
        num_kv_heads = 1 if multi_query else (_safe_int(_cfg_get(config, "num_kv_heads", num_heads), num_heads) or num_heads)
        intermediate_size = _safe_int(_cfg_get(config, "intermediate_size", hidden_size * 4), hidden_size * 4)
        max_position = _safe_int(_cfg_get(config, "max_position_embeddings", 2048), 2048)
        head_dim = _head_dim(config, hidden_size, num_heads)

        features = ["Falcon", "LayerNorm", "GELU", "Causal"]
        if _cfg_get(config, "alibi", False):
            features.append("ALiBi")
        if num_kv_heads == 1 and num_heads > 1:
            features.append("MQA")
        elif num_kv_heads != num_heads:
            features.append("GQA")
        else:
            features.append("MHA")

        layers = []
        total_params = 0
        emb_params = vocab_size * hidden_size
        layers.append(Layer("Token Embedding", "embedding", emb_params, (vocab_size, hidden_size), f"Vocab: {vocab_size}"))
        total_params += emb_params

        attn_params = hidden_size * (num_heads * head_dim) + 2 * hidden_size * (num_kv_heads * head_dim) + (num_heads * head_dim) * hidden_size
        ffn_params = 2 * hidden_size * intermediate_size
        ln_params = 2 * hidden_size
        for i in range(num_layers):
            layers.append(Layer(f"Block {i}", "block_start", 0, (), ""))
            layers.append(Layer(f"Block {i} - Attention", "attention", attn_params, (num_heads, head_dim), "Parallel Attention"))
            total_params += attn_params
            layers.append(Layer(f"Block {i} - FFN", "feedforward", ffn_params, (hidden_size, intermediate_size), "GELU"))
            total_params += ffn_params
            layers.append(Layer(f"Block {i} - Norm", "normalization", ln_params, (hidden_size,), "LayerNorm"))
            total_params += ln_params
            layers.append(Layer(f"Block {i}", "block_end", 0, (), ""))

        return _finalize_architecture(
            Architecture(
                model_type=getattr(config, "model_type", "falcon"),
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
                    "intermediate_size": intermediate_size,
                    "max_position": max_position,
                },
                features=features,
                layers=layers,
            ),
            total_layers=num_layers,
        )


class OPTAnalyzer(ModelAnalyzer):
    """OPT family analyzer with learned positional embeddings."""

    def analyze(self, config: Any) -> Architecture:
        vocab_size = _safe_int(_cfg_get(config, "vocab_size", 50272), 50272)
        hidden_size = _safe_int(_cfg_get(config, "hidden_size", 768), 768)
        num_layers = _safe_int(_cfg_get(config, "num_hidden_layers", 12), 12)
        num_heads = _safe_int(_cfg_get(config, "num_attention_heads", 12), 12)
        intermediate_size = _safe_int(_cfg_get(config, "ffn_dim", _cfg_get(config, "intermediate_size", hidden_size * 4)), hidden_size * 4)
        max_position = _safe_int(_cfg_get(config, "max_position_embeddings", 2048), 2048)
        head_dim = _head_dim(config, hidden_size, num_heads)

        features = ["OPT", "MHA", "LearnedPos", "LayerNorm", "ReLU", "Causal"]
        layers = []
        total_params = 0

        token_emb = vocab_size * hidden_size
        pos_emb = max_position * hidden_size
        layers.append(Layer("Token Embedding", "embedding", token_emb, (vocab_size, hidden_size), f"Vocab: {vocab_size}"))
        layers.append(Layer("Position Embedding", "embedding", pos_emb, (max_position, hidden_size), f"Max pos: {max_position}"))
        total_params += token_emb + pos_emb

        attn_params = 4 * hidden_size * hidden_size
        ffn_params = 2 * hidden_size * intermediate_size
        ln_params = 2 * hidden_size
        for i in range(num_layers):
            layers.append(Layer(f"Block {i}", "block_start", 0, (), ""))
            layers.append(Layer(f"Block {i} - Attention", "attention", attn_params, (num_heads, head_dim), "Causal MHA"))
            total_params += attn_params
            layers.append(Layer(f"Block {i} - FFN", "feedforward", ffn_params, (hidden_size, intermediate_size), "ReLU"))
            total_params += ffn_params
            layers.append(Layer(f"Block {i} - Norm", "normalization", ln_params, (hidden_size,), "LayerNorm"))
            total_params += ln_params
            layers.append(Layer(f"Block {i}", "block_end", 0, (), ""))

        return _finalize_architecture(
            Architecture(
                model_type=getattr(config, "model_type", "opt"),
                arch_type="decoder-only",
                total_layers=num_layers,
                total_params=total_params,
                memory_fp16_gb=total_params * 2 / (1024**3),
                parameters={
                    "vocab_size": vocab_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "num_kv_heads": num_heads,
                    "head_dim": head_dim,
                    "intermediate_size": intermediate_size,
                    "max_position": max_position,
                },
                features=features,
                layers=layers,
            ),
            total_layers=num_layers,
        )
