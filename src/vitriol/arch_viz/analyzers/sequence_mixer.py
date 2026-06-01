"""Fallback analyzer for SSM / recurrent (non-KV) families."""

from typing import Any

from ..core import Architecture, Layer
from ._helpers import (
    _append_feature,
    _cfg_first,
    _cfg_get,
    _finalize_architecture,
    _infer_ffn_feature,
    _infer_norm_feature,
    _project_subconfig,
    _safe_int,
)
from .base import ModelAnalyzer


class SequenceMixerAnalyzer(ModelAnalyzer):
    """Fallback analyzer for SSM/recurrent families that do not expose Transformer KV cache."""

    def analyze(self, config: Any) -> Architecture:
        _project_subconfig(config, "text_config")

        model_type = str(_cfg_get(config, "model_type", "sequence_mixer") or "sequence_mixer").lower()
        vocab_size = _safe_int(_cfg_first(config, ("vocab_size", "padded_vocab_size"), 0), 0)
        hidden_size = _safe_int(_cfg_first(config, ("hidden_size", "d_model", "n_embd", "dim"), 0), 0)
        num_layers = _safe_int(_cfg_first(config, ("num_hidden_layers", "n_layer", "n_layers", "num_layers"), 0), 0)
        intermediate_size = _safe_int(
            _cfg_first(config, ("intermediate_size", "expand", "ffn_hidden_size"), hidden_size * 4),
            hidden_size * 4,
        )
        state_size = _safe_int(_cfg_first(config, ("state_size", "d_state", "ssm_state_size"), 0), 0)
        conv_kernel = _safe_int(_cfg_first(config, ("conv_kernel", "d_conv", "time_mix_extra_dim"), 0), 0)
        max_position = _safe_int(_cfg_first(config, ("max_position_embeddings", "n_positions", "seq_length"), 0), 0)

        family = "State Space"
        if "rwkv" in model_type:
            family = "RWKV"
        elif "mamba" in model_type:
            family = "Mamba"
        elif "retnet" in model_type:
            family = "RetNet"
        elif "hyena" in model_type:
            family = "Hyena"

        features = [family, "Sequence Mixer", "No KV Cache"]
        if max_position >= 32768:
            features.append("Long Context")
        _append_feature(features, _infer_norm_feature(config))
        _append_feature(features, _infer_ffn_feature(config))
        _append_feature(features, "Causal")

        layers = []
        total_params = 0
        emb_params = vocab_size * hidden_size
        layers.append(Layer("Token Embedding", "embedding", emb_params, (vocab_size, hidden_size), f"Vocab: {vocab_size}"))
        total_params += emb_params

        mixer_params = hidden_size * max(state_size, hidden_size)
        if conv_kernel:
            mixer_params += hidden_size * conv_kernel
        ffn_params = 2 * hidden_size * intermediate_size if intermediate_size else 0
        for i in range(num_layers):
            layers.append(Layer(f"Block {i}", "block_start", 0, (), ""))
            layers.append(Layer(f"Block {i} - Norm", "normalization", hidden_size, (hidden_size,)))
            total_params += hidden_size
            desc = f"{family} mixer"
            if state_size:
                desc += f", state={state_size}"
            if conv_kernel:
                desc += f", conv={conv_kernel}"
            layers.append(Layer(f"Block {i} - Sequence Mixer", "sequence_mixer", mixer_params, (hidden_size, state_size or hidden_size), desc))
            total_params += mixer_params
            if ffn_params:
                layers.append(Layer(f"Block {i} - FFN", "feedforward", ffn_params, (hidden_size, intermediate_size), f"Inter: {intermediate_size}"))
                total_params += ffn_params
            layers.append(Layer(f"Block {i}", "block_end", 0, (), ""))

        layers.append(Layer("Final Norm", "normalization", hidden_size, (hidden_size,)))
        total_params += hidden_size
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
                model_type=model_type,
                arch_type="decoder-only",
                total_layers=num_layers,
                total_params=total_params,
                memory_fp16_gb=total_params * 2 / (1024**3),
                parameters={
                    "vocab_size": vocab_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "intermediate_size": intermediate_size,
                    "state_size": state_size,
                    "conv_kernel": conv_kernel,
                    "max_position": max_position,
                    "supports_kv_cache": False,
                    "layer_types": ["other"] * num_layers,
                },
                features=features,
                layers=layers,
            ),
            total_layers=num_layers,
        )
