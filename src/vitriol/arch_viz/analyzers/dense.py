"""Dense Transformer family analyzers (Qwen, Llama, Mistral, Gemma, ...)."""

from typing import Any

from ..core import Architecture, Layer
from ._helpers import (
    _append_feature,
    _cfg_get,
    _finalize_architecture,
    _safe_float,
    _safe_int,
)
from .base import TransformerAnalyzer


class QwenAnalyzer(TransformerAnalyzer):
    """Qwen specific analyzer."""
    pass # Uses standard logic mostly, but handles specific config keys if needed


class Qwen2MoeAnalyzer(TransformerAnalyzer):
    """Qwen2-MoE analyzer with family-specific metadata."""

    def _detect_features(self, config, num_heads, num_kv_heads):
        features = super()._detect_features(config, num_heads, num_kv_heads)
        if "Qwen2-MoE" not in features:
            features.insert(0, "Qwen2-MoE")
        if _cfg_get(config, "decoder_sparse_step", None):
            features.append("Sparse Routing")
        return features


class LlamaAnalyzer(TransformerAnalyzer):
    """LLaMA family analyzer with GQA / RoPE metadata."""

    def _detect_features(self, config, num_heads, num_kv_heads):
        features = super()._detect_features(config, num_heads, num_kv_heads)
        if "LLaMA" not in features:
            features.insert(0, "LLaMA")
        if _cfg_get(config, "attention_bias", None) is False:
            features.append("Bias-Free Attention")
        return features


class MistralAnalyzer(TransformerAnalyzer):
    """Mistral / Mixtral analyzer."""

    def _detect_features(self, config, num_heads, num_kv_heads):
        features = super()._detect_features(config, num_heads, num_kv_heads)
        model_type = str(_cfg_get(config, "model_type", "")).lower()
        if "mixtral" in model_type:
            features.insert(0, "Mixtral")
            if "MoE" not in features:
                features.append("MoE")
        else:
            features.insert(0, "Mistral")
        return features


class GemmaAnalyzer(TransformerAnalyzer):
    """Gemma family analyzer, including multimodal Gemma variants."""

    def analyze(self, config: Any) -> Architecture:
        vision_config = getattr(config, "vision_config", None)
        if vision_config is None:
            arch = super().analyze(config)
        else:
            text_config = getattr(config, "text_config", config)
            arch = super().analyze(text_config)

            v_hidden = _safe_int(_cfg_get(vision_config, "hidden_size", 0), 0)
            v_layers = _safe_int(_cfg_get(vision_config, "num_hidden_layers", _cfg_get(vision_config, "depth", 0)), 0)
            v_patch = _safe_int(_cfg_get(vision_config, "patch_size", 14), 14)
            v_params = (3 * v_hidden * v_patch * v_patch) + (12 * v_layers * v_hidden * v_hidden)
            projector_params = v_hidden * arch.parameters.get("hidden_size", 0)
            prefix_layers = []
            if v_params > 0:
                prefix_layers.append(
                    Layer("Vision Encoder", "encoder", v_params, (v_layers, v_hidden), f"Vision Tower/{v_layers}")
                )
            if projector_params > 0:
                prefix_layers.append(
                    Layer(
                        "Vision Projector",
                        "adapter",
                        projector_params,
                        (v_hidden, arch.parameters.get("hidden_size", 0)),
                        "Projection",
                    )
                )
            arch.layers = prefix_layers + arch.layers
            arch.total_params += v_params + projector_params
            arch.memory_fp16_gb = arch.total_params * 2 / (1024**3)
            arch.parameters["vision_hidden_size"] = v_hidden
            arch.parameters["vision_layers"] = v_layers

        arch.model_type = getattr(config, "model_type", arch.model_type)
        if "Gemma" not in arch.features:
            arch.features.insert(0, "Gemma")
        if vision_config is not None and "Vision" not in arch.features:
            arch.features.append("Vision")
        return _finalize_architecture(
            arch,
            total_layers=int(arch.parameters.get("num_layers", arch.total_layers) or arch.total_layers),
            encoder_layers=arch.encoder_layers,
            decoder_layers=arch.decoder_layers,
        )


class PhiAnalyzer(TransformerAnalyzer):
    """Microsoft Phi family analyzer."""

    def _detect_features(self, config, num_heads, num_kv_heads):
        features = super()._detect_features(config, num_heads, num_kv_heads)
        if "Phi" not in features:
            features.insert(0, "Phi")
        if _safe_float(_cfg_get(config, "partial_rotary_factor", 0.0), 0.0) > 0.0:
            features.append("Partial RoPE")
        return features


class CohereAnalyzer(TransformerAnalyzer):
    """Cohere / Command-R family analyzer."""

    def _detect_features(self, config, num_heads, num_kv_heads):
        features = super()._detect_features(config, num_heads, num_kv_heads)
        if "Cohere" not in features:
            features.insert(0, "Cohere")
        if _cfg_get(config, "use_qk_norm", None):
            features.append("QK-Norm")
        return features


class StableLMAnalyzer(TransformerAnalyzer):
    """StableLM family analyzer."""

    def _detect_features(self, config, num_heads, num_kv_heads):
        features = super()._detect_features(config, num_heads, num_kv_heads)
        if "StableLM" not in features:
            features.insert(0, "StableLM")
        return features


class YiAnalyzer(TransformerAnalyzer):
    """Yi family analyzer."""

    def _detect_features(self, config, num_heads, num_kv_heads):
        features = super()._detect_features(config, num_heads, num_kv_heads)
        if "Yi" not in features:
            features.insert(0, "Yi")
        _append_feature(features, "RMSNorm")
        _append_feature(features, "SwiGLU")
        if _cfg_get(config, "rope_scaling", None):
            _append_feature(features, "Dynamic NTK")
        return features


class InternLMAnalyzer(TransformerAnalyzer):
    """InternLM / InternLM2 family analyzer."""

    def _detect_features(self, config, num_heads, num_kv_heads):
        features = super()._detect_features(config, num_heads, num_kv_heads)
        if "InternLM" not in features:
            features.insert(0, "InternLM")
        _append_feature(features, "RMSNorm")
        _append_feature(features, "SwiGLU")
        return features


class BaichuanAnalyzer(TransformerAnalyzer):
    """Baichuan family analyzer."""

    def _detect_features(self, config, num_heads, num_kv_heads):
        features = super()._detect_features(config, num_heads, num_kv_heads)
        if "Baichuan" not in features:
            features.insert(0, "Baichuan")
        _append_feature(features, "RMSNorm")
        _append_feature(features, "SwiGLU")
        if _cfg_get(config, "rope_scaling", None):
            _append_feature(features, "Dynamic NTK")
        return features
