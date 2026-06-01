"""Encoder / encoder-decoder families (BERT, T5)."""

from typing import Any

from ..core import Architecture, Layer
from ._helpers import (
    _cfg_get,
    _finalize_architecture,
    _head_dim,
    _safe_int,
)
from .base import ModelAnalyzer


class BertAnalyzer(ModelAnalyzer):
    """BERT / RoBERTa style encoder-only analyzer."""

    def analyze(self, config: Any) -> Architecture:
        vocab_size = _safe_int(_cfg_get(config, "vocab_size", 30522), 30522)
        hidden_size = _safe_int(_cfg_get(config, "hidden_size", 768), 768)
        num_layers = _safe_int(_cfg_get(config, "num_hidden_layers", 12), 12)
        num_heads = _safe_int(_cfg_get(config, "num_attention_heads", 12), 12)
        intermediate_size = _safe_int(_cfg_get(config, "intermediate_size", hidden_size * 4), hidden_size * 4)
        max_position = _safe_int(_cfg_get(config, "max_position_embeddings", 512), 512)
        type_vocab_size = _safe_int(_cfg_get(config, "type_vocab_size", 2), 2)
        head_dim = _head_dim(config, hidden_size, num_heads)

        features = ["BERT", "MHA", "AbsPos", "LayerNorm", "GELU", "Bidirectional"]
        layers = []
        total_params = 0

        token_emb = vocab_size * hidden_size
        pos_emb = max_position * hidden_size
        type_emb = type_vocab_size * hidden_size
        emb_ln = 2 * hidden_size
        layers.append(Layer("Token Embedding", "embedding", token_emb, (vocab_size, hidden_size), f"Vocab: {vocab_size}"))
        layers.append(Layer("Position Embedding", "embedding", pos_emb, (max_position, hidden_size), f"Max pos: {max_position}"))
        layers.append(Layer("Token Type Embedding", "embedding", type_emb, (type_vocab_size, hidden_size), f"Segments: {type_vocab_size}"))
        layers.append(Layer("Embedding Norm", "normalization", emb_ln, (hidden_size,), "LayerNorm"))
        total_params += token_emb + pos_emb + type_emb + emb_ln

        attn_params = 4 * hidden_size * hidden_size
        ln_params = 2 * hidden_size
        ffn_params = (hidden_size * intermediate_size) + intermediate_size + (intermediate_size * hidden_size) + hidden_size
        for i in range(num_layers):
            layers.append(Layer(f"Encoder Block {i}", "block_start", 0, (), ""))
            layers.append(Layer(f"Encoder Block {i} - Attention", "attention", attn_params, (num_heads, head_dim), "Bidirectional MHA"))
            total_params += attn_params
            layers.append(Layer(f"Encoder Block {i} - Norm 1", "normalization", ln_params, (hidden_size,), "LayerNorm"))
            total_params += ln_params
            layers.append(Layer(f"Encoder Block {i} - FFN", "feedforward", ffn_params, (hidden_size, intermediate_size), "GELU"))
            total_params += ffn_params
            layers.append(Layer(f"Encoder Block {i} - Norm 2", "normalization", ln_params, (hidden_size,), "LayerNorm"))
            total_params += ln_params
            layers.append(Layer(f"Encoder Block {i}", "block_end", 0, (), ""))

        pooler_params = (hidden_size * hidden_size) + hidden_size
        layers.append(Layer("Pooler", "adapter", pooler_params, (hidden_size, hidden_size), "[CLS] projection"))
        total_params += pooler_params

        return _finalize_architecture(
            Architecture(
                model_type=getattr(config, "model_type", "bert"),
                arch_type="encoder-only",
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
                encoder_layers=num_layers,
            ),
            total_layers=num_layers,
            encoder_layers=num_layers,
        )


class T5Analyzer(ModelAnalyzer):
    """T5 / BART-style encoder-decoder analyzer."""

    def analyze(self, config: Any) -> Architecture:
        vocab_size = _safe_int(_cfg_get(config, "vocab_size", 32128), 32128)
        hidden_size = _safe_int(_cfg_get(config, "d_model", _cfg_get(config, "hidden_size", 512)), 512)
        num_layers = _safe_int(_cfg_get(config, "num_layers", _cfg_get(config, "num_hidden_layers", 12)), 12)
        num_decoder_layers = _safe_int(_cfg_get(config, "num_decoder_layers", num_layers), num_layers)
        num_heads = _safe_int(_cfg_get(config, "num_heads", _cfg_get(config, "num_attention_heads", 8)), 8)
        d_kv = _safe_int(_cfg_get(config, "d_kv", 0), 0) or _head_dim(config, hidden_size, num_heads)
        d_ff = _safe_int(_cfg_get(config, "d_ff", _cfg_get(config, "intermediate_size", hidden_size * 4)), hidden_size * 4)
        max_position = _safe_int(_cfg_get(config, "n_positions", _cfg_get(config, "max_position_embeddings", 512)), 512)
        ff_proj = str(_cfg_get(config, "feed_forward_proj", "relu")).lower()
        ff_mats = 3 if "gated" in ff_proj else 2
        ff_name = "GeGLU" if "gated" in ff_proj or "gelu" in ff_proj else "ReLU"

        features = ["T5", "MHA", "RelPos", "LayerNorm", "CrossAttn", ff_name]
        layers = []
        total_params = 0

        shared_embed = vocab_size * hidden_size
        layers.append(Layer("Shared Embedding", "embedding", shared_embed, (vocab_size, hidden_size), f"Vocab: {vocab_size}"))
        total_params += shared_embed

        self_attn_params = 4 * hidden_size * (num_heads * d_kv)
        cross_attn_params = 4 * hidden_size * (num_heads * d_kv)
        ffn_params = ff_mats * hidden_size * d_ff
        norm_params = hidden_size

        for i in range(num_layers):
            layers.append(Layer(f"Encoder Block {i}", "block_start", 0, (), ""))
            layers.append(Layer(f"Encoder Block {i} - Self Attention", "attention", self_attn_params, (num_heads, d_kv), "Relative Position Bias"))
            total_params += self_attn_params
            layers.append(Layer(f"Encoder Block {i} - FFN", "feedforward", ffn_params, (hidden_size, d_ff), ff_name))
            total_params += ffn_params
            layers.append(Layer(f"Encoder Block {i} - Norm", "normalization", 2 * norm_params, (hidden_size,), "LayerNorm"))
            total_params += 2 * norm_params
            layers.append(Layer(f"Encoder Block {i}", "block_end", 0, (), ""))

        for i in range(num_decoder_layers):
            layers.append(Layer(f"Decoder Block {i}", "block_start", 0, (), ""))
            layers.append(Layer(f"Decoder Block {i} - Self Attention", "attention", self_attn_params, (num_heads, d_kv), "Causal + Relative Position"))
            total_params += self_attn_params
            layers.append(Layer(f"Decoder Block {i} - Cross Attention", "attention", cross_attn_params, (num_heads, d_kv), "Encoder-Decoder Attention"))
            total_params += cross_attn_params
            layers.append(Layer(f"Decoder Block {i} - FFN", "feedforward", ffn_params, (hidden_size, d_ff), ff_name))
            total_params += ffn_params
            layers.append(Layer(f"Decoder Block {i} - Norm", "normalization", 3 * norm_params, (hidden_size,), "LayerNorm"))
            total_params += 3 * norm_params
            layers.append(Layer(f"Decoder Block {i}", "block_end", 0, (), ""))

        final_norm = hidden_size
        layers.append(Layer("Final Norm", "normalization", final_norm, (hidden_size,), "LayerNorm"))
        total_params += final_norm

        return _finalize_architecture(
            Architecture(
                model_type=getattr(config, "model_type", "t5"),
                arch_type="encoder-decoder",
                total_layers=num_layers + num_decoder_layers,
                total_params=total_params,
                memory_fp16_gb=total_params * 2 / (1024**3),
                parameters={
                    "vocab_size": vocab_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers + num_decoder_layers,
                    "encoder_layers": num_layers,
                    "decoder_layers": num_decoder_layers,
                    "num_heads": num_heads,
                    "num_kv_heads": num_heads,
                    "head_dim": d_kv,
                    "intermediate_size": d_ff,
                    "max_position": max_position,
                },
                features=features,
                layers=layers,
                encoder_layers=num_layers,
                decoder_layers=num_decoder_layers,
            ),
            total_layers=num_layers + num_decoder_layers,
            encoder_layers=num_layers,
            decoder_layers=num_decoder_layers,
        )
