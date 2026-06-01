import logging
from typing import Any, Optional

from .core import Architecture, Layer

logger = logging.getLogger(__name__)


def _cfg_get(obj: Any, key: str, default: Any = 0) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _cfg_first(obj: Any, keys: tuple[str, ...], default: Any = 0) -> Any:
    for key in keys:
        value = _cfg_get(obj, key, None)
        if value is not None:
            return value
    return default


def _cfg_items(obj: Any):
    if obj is None:
        return []
    if isinstance(obj, dict):
        return obj.items()
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict().items()
        except Exception:
            return []
    if hasattr(obj, "__dict__"):
        return vars(obj).items()
    return []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return default


def _head_dim(config: Any, hidden_size: int, num_heads: int) -> int:
    head_dim = _safe_int(_cfg_get(config, "head_dim", 0), 0)
    if head_dim > 0:
        return head_dim
    return (hidden_size // num_heads) if num_heads else 0


def _num_experts(config: Any) -> int:
    return _safe_int(
        _cfg_get(config, "num_local_experts", _cfg_get(config, "num_experts", _cfg_get(config, "n_routed_experts", 0))),
        0,
    )


def _project_subconfig(config: Any, attr: str) -> Any:
    sub_config = getattr(config, attr, None)
    if sub_config is None:
        return None
    for key, value in _cfg_items(sub_config):
        setattr(config, key, value)
    return sub_config


def _architectures(config: Any) -> list[str]:
    names: list[str] = []
    for source in (config, getattr(config, "text_config", None), getattr(config, "vision_config", None)):
        archs = _cfg_get(source, "architectures", []) if source is not None else []
        if archs:
            names.extend(str(arch).lower() for arch in archs)
    return names


def _append_feature(features: list[str], feature: Optional[str]) -> None:
    if feature and feature not in features:
        features.append(feature)


def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [_safe_int(item, 0) for item in value]
    return []


def _infer_norm_feature(config: Any) -> Optional[str]:
    norm_type = str(
        _cfg_get(
            config,
            "norm_type",
            _cfg_get(config, "normalization_type", ""),
        )
        or ""
    ).lower()
    if "rms" in norm_type:
        return "RMSNorm"
    if _cfg_get(config, "rms_norm_eps", None) is not None or _cfg_get(config, "rmsnorm", False):
        return "RMSNorm"
    if (
        _cfg_get(config, "layer_norm_epsilon", None) is not None
        or _cfg_get(config, "layer_norm_eps", None) is not None
        or "layernorm" in norm_type
    ):
        return "LayerNorm"
    return None


def _infer_ffn_feature(config: Any) -> Optional[str]:
    act = str(
        _cfg_get(
            config,
            "hidden_act",
            _cfg_get(
                config,
                "hidden_activation",
                _cfg_get(
                    config,
                    "activation_function",
                    _cfg_get(config, "feed_forward_proj", ""),
                ),
            ),
        )
        or ""
    ).lower()
    if not act:
        return None
    if "geglu" in act or ("gated" in act and "gelu" in act):
        return "GeGLU"
    if "swiglu" in act or act in {"silu", "swish"}:
        return "SwiGLU"
    if "gelu" in act:
        return "GELU"
    if "relu" in act:
        return "ReLU"
    return None


def _finalize_architecture(
    arch: Architecture,
    *,
    total_layers: Optional[int] = None,
    encoder_layers: int = 0,
    decoder_layers: int = 0,
) -> Architecture:
    if total_layers is not None:
        arch.total_layers = int(total_layers or 0)
    if encoder_layers:
        arch.encoder_layers = int(encoder_layers)
        arch.parameters.setdefault("encoder_layers", arch.encoder_layers)
    if decoder_layers:
        arch.decoder_layers = int(decoder_layers)
        arch.parameters.setdefault("decoder_layers", arch.decoder_layers)
    arch.parameters.setdefault("num_layers", arch.total_layers)
    arch.special_features = list(arch.features)
    return arch

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

class QwenAnalyzer(TransformerAnalyzer):
    """Qwen specific analyzer."""
    pass # Uses standard logic mostly, but handles specific config keys if needed


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

class GLMAnalyzer(TransformerAnalyzer):
    """GLM-5 / glm_moe_dsa specific analyzer."""
    def analyze(self, config: Any) -> Architecture:
        # GLM-5 has hybrid layers (dense + sparse MoE) defined by mlp_layer_types
        # We need to iterate carefully

        vocab_size = getattr(config, 'vocab_size', 0)
        hidden_size = getattr(config, 'hidden_size', 0)
        num_layers = getattr(config, 'num_hidden_layers', 0)
        num_heads = getattr(config, 'num_attention_heads', 0)
        num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)

        # GLM specific fields
        mlp_layer_types = getattr(config, 'mlp_layer_types', [])
        moe_intermediate_size = getattr(config, 'moe_intermediate_size', 0)
        intermediate_size = getattr(config, 'intermediate_size', 0)
        num_experts = getattr(config, 'n_routed_experts', 0)
        active_experts = getattr(config, 'num_experts_per_tok', 0)
        shared_experts = getattr(config, 'n_shared_experts', 0)

        features = ["GLM-5", "DSA"]
        if "sparse" in mlp_layer_types:
            features.append("MoE")

        layers = []
        total_params = 0

        # 1. Embedding
        emb_params = vocab_size * hidden_size
        layers.append(Layer("Token Embedding", "embedding", emb_params, (vocab_size, hidden_size), f"Vocab: {vocab_size}"))
        total_params += emb_params

        # 2. Layers
        for i in range(num_layers):
            layers.append(Layer(f"Block {i}", "block_start", 0, (), ""))

            # Norm 1
            ln_params = hidden_size
            layers.append(Layer(f"Block {i} - Norm 1", "normalization", ln_params, (hidden_size,)))
            total_params += ln_params

            # Attention (DSA?)
            # Simplified calculation for standard attention, adjust for DSA/MLA if needed
            # GLM-5 uses standard GQA + RoPE usually, DSA is sparsity pattern
            head_dim = hidden_size // num_heads
            # Q, K, V, O
            # Check for qk_nope_head_dim etc if needed for precise count
            # Standard GQA:
            q_params = hidden_size * hidden_size
            k_params = hidden_size * (num_kv_heads * head_dim)
            v_params = hidden_size * (num_kv_heads * head_dim)
            o_params = hidden_size * hidden_size
            attn_params = q_params + k_params + v_params + o_params

            layers.append(Layer(f"Block {i} - Attention", "attention", attn_params, (hidden_size, hidden_size), "DSA"))
            total_params += attn_params

            # Norm 2
            layers.append(Layer(f"Block {i} - Norm 2", "normalization", ln_params, (hidden_size,)))
            total_params += ln_params

            # FFN / MoE
            layer_type = mlp_layer_types[i] if i < len(mlp_layer_types) else "dense"

            if layer_type == "sparse":
                # MoE Layer
                # Gate
                gate_params = hidden_size * num_experts
                # Experts: SwiGLU 3 matrices
                # Routed Experts
                routed_params = num_experts * (3 * hidden_size * moe_intermediate_size)
                # Shared Experts
                shared_params = shared_experts * (3 * hidden_size * moe_intermediate_size)

                ffn_params = gate_params + routed_params + shared_params
                desc = f"MoE (Routed: {num_experts}, Active: {active_experts}, Shared: {shared_experts})"
            else:
                # Dense Layer
                # SwiGLU
                ffn_params = 3 * hidden_size * intermediate_size
                desc = f"Dense (Inter: {intermediate_size})"

            layers.append(Layer(f"Block {i} - FFN", "feedforward", ffn_params, (hidden_size, intermediate_size), desc))
            total_params += ffn_params

            layers.append(Layer(f"Block {i}", "block_end", 0, (), ""))

        # 3. Final Norm
        layers.append(Layer("Final Norm", "normalization", hidden_size, (hidden_size,)))
        total_params += hidden_size

        # 4. Head
        if getattr(config, 'tie_word_embeddings', False):
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
                arch_type="decoder-only",
                total_layers=num_layers,
                total_params=total_params,
                memory_fp16_gb=total_params * 2 / (1024**3),
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

class ErnieAnalyzer(TransformerAnalyzer):
    """ERNIE 4.5 VL (MoE + Vision) Analyzer."""

    def analyze(self, config: Any) -> Architecture:
        # Check for vision config
        vision_config = getattr(config, 'vision_config', None)

        # Base Transformer Analysis (Language Part)
        vocab_size = getattr(config, 'vocab_size', 0)
        hidden_size = getattr(config, 'hidden_size', 0)
        num_layers = getattr(config, 'num_hidden_layers', 0)
        num_heads = getattr(config, 'num_attention_heads', 0)

        # MoE Config
        moe_num_experts = getattr(config, 'moe_num_experts', [0])
        # config.moe_num_experts is list [64, 64] ? Or maybe different per layer?
        # Based on config: "moe_num_experts": [64, 64]
        # "moe_num_shared_experts": 2
        n_routed = moe_num_experts[0] if isinstance(moe_num_experts, list) else moe_num_experts
        n_shared = getattr(config, 'moe_num_shared_experts', 0)
        active_experts = getattr(config, 'moe_k', 0)

        moe_intermediate_list = getattr(config, 'moe_intermediate_size', [0, 0])
        # [1536, 512] ?
        # Usually [routed_inter, shared_inter] or similar
        routed_inter = moe_intermediate_list[0] if isinstance(moe_intermediate_list, list) else moe_intermediate_list
        shared_inter = moe_intermediate_list[1] if isinstance(moe_intermediate_list, list) and len(moe_intermediate_list) > 1 else routed_inter

        features = ["ERNIE-4.5", "MoE", "Vision-Language"]
        if getattr(config, 'rope_3d', False):
            features.append("3D-RoPE")

        layers = []
        total_params = 0

        # --- Vision Encoder ---
        if vision_config:
            v_hidden = getattr(vision_config, 'hidden_size', 0)
            v_layers = getattr(vision_config, 'depth', getattr(vision_config, 'num_hidden_layers', 0))
            v_patch = getattr(vision_config, 'patch_size', 14)

            # Simple Vision Encoder Estimate
            # ViT-like: PatchEmbed + Layers * (Attn + MLP)
            v_params = 0
            # Patch Embed: (3, hidden, patch, patch)
            v_params += 3 * v_hidden * v_patch * v_patch

            # Transformer Layers
            # Attn: 4 * hidden^2 (Q,K,V,O)
            # MLP: 2 * hidden * (4*hidden) = 8 * hidden^2
            # Norms: 2 * hidden
            layer_p = 12 * v_hidden * v_hidden
            v_params += v_layers * layer_p

            layers.append(Layer("Vision Encoder", "encoder", v_params, (v_layers, v_hidden), f"ViT-L/{v_layers}"))
            total_params += v_params

            # Projector (if any)
            mm_hidden = getattr(vision_config, 'mm_hidden_size', v_hidden)
            proj_params = v_hidden * mm_hidden # Linear
            layers.append(Layer("Vision Projector", "adapter", proj_params, (v_hidden, mm_hidden), "Linear"))
            total_params += proj_params

        # --- Language Model ---

        # 1. Embedding
        emb_params = vocab_size * hidden_size
        layers.append(Layer("Token Embedding", "embedding", emb_params, (vocab_size, hidden_size), f"Vocab: {vocab_size}"))
        total_params += emb_params

        # 2. Layers
        for i in range(num_layers):
            layers.append(Layer(f"Block {i}", "block_start", 0, (), ""))

            # Norm 1
            ln_params = hidden_size
            layers.append(Layer(f"Block {i} - Norm 1", "normalization", ln_params, (hidden_size,)))
            total_params += ln_params

            # Attention
            attn_params = 4 * hidden_size * hidden_size # Standard Approx
            layers.append(Layer(f"Block {i} - Attention", "attention", attn_params, (hidden_size, hidden_size), "FlashAttn"))
            total_params += attn_params

            # Norm 2
            layers.append(Layer(f"Block {i} - Norm 2", "normalization", ln_params, (hidden_size,)))
            total_params += ln_params

            # MoE FFN
            # Gate
            gate_params = hidden_size * n_routed
            # Routed Experts (SwiGLU 3 matrices)
            routed_params = n_routed * (3 * hidden_size * routed_inter)
            # Shared Experts (SwiGLU 3 matrices)
            shared_params = n_shared * (3 * hidden_size * shared_inter)

            ffn_params = gate_params + routed_params + shared_params
            desc = f"MoE (Routed: {n_routed}, Active: {active_experts}, Shared: {n_shared})"

            layers.append(Layer(f"Block {i} - FFN", "feedforward", ffn_params, (hidden_size, routed_inter), desc))
            total_params += ffn_params

            layers.append(Layer(f"Block {i}", "block_end", 0, (), ""))

        # 3. Final Norm
        layers.append(Layer("Final Norm", "normalization", hidden_size, (hidden_size,)))
        total_params += hidden_size

        # 4. Head
        if getattr(config, 'tie_word_embeddings', False):
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
                arch_type="encoder-decoder", # VL is effectively hybrid
                total_layers=num_layers,
                total_params=total_params,
                memory_fp16_gb=total_params * 2 / (1024**3),
                parameters={
                    "vocab_size": vocab_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "vision_layers": int(getattr(vision_config, "num_hidden_layers", getattr(vision_config, "depth", 0)) or 0),
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

class MiniMaxAnalyzer(TransformerAnalyzer):
    """MiniMax-M2.5 Analyzer."""

    def analyze(self, config: Any) -> Architecture:
        # MiniMax M2.5 has MTP modules (Multi-Token Prediction?) or similar auxiliary heads
        # config.use_mtp = True, config.num_mtp_modules = 3
        # It is an MoE model

        vocab_size = getattr(config, 'vocab_size', 0)
        hidden_size = getattr(config, 'hidden_size', 0)
        num_layers = getattr(config, 'num_hidden_layers', 0)
        num_heads = getattr(config, 'num_attention_heads', 0)
        num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)

        # MoE Params
        n_local_experts = getattr(config, 'num_local_experts', 0)
        active_experts = getattr(config, 'num_experts_per_tok', 0)

        features = ["MiniMax-M2.5", "MoE"]
        if getattr(config, 'use_mtp', False):
            features.append(f"MTP (Modules: {getattr(config, 'num_mtp_modules', 0)})")
        if getattr(config, 'attn_type_list', None):
            features.append("Hybrid Attention")

        layers = []
        total_params = 0

        # 1. Embedding
        emb_params = vocab_size * hidden_size
        layers.append(Layer("Token Embedding", "embedding", emb_params, (vocab_size, hidden_size), f"Vocab: {vocab_size}"))
        total_params += emb_params

        # 2. Layers
        # MiniMax uses attn_type_list to switch between different attention types?
        # Based on config: "attn_type_list": [1, 1, ..., 1] (all 1s)

        for i in range(num_layers):
            layers.append(Layer(f"Block {i}", "block_start", 0, (), ""))

            # Norm 1
            ln_params = hidden_size
            layers.append(Layer(f"Block {i} - Norm 1", "normalization", ln_params, (hidden_size,)))
            total_params += ln_params

            # Attention
            head_dim = getattr(config, 'head_dim', hidden_size // num_heads)
            # Standard GQA
            q_params = hidden_size * (num_heads * head_dim)
            k_params = hidden_size * (num_kv_heads * head_dim)
            v_params = hidden_size * (num_kv_heads * head_dim)
            o_params = (num_heads * head_dim) * hidden_size

            attn_params = q_params + k_params + v_params + o_params
            layers.append(Layer(f"Block {i} - Attention", "attention", attn_params, (hidden_size, hidden_size), "GQA"))
            total_params += attn_params

            # Norm 2
            layers.append(Layer(f"Block {i} - Norm 2", "normalization", ln_params, (hidden_size,)))
            total_params += ln_params

            # MoE FFN
            # Gate
            gate_params = hidden_size * n_local_experts
            # Experts (SwiGLU)
            intermediate_size = getattr(config, 'intermediate_size', 0)
            expert_params = n_local_experts * (3 * hidden_size * intermediate_size)

            ffn_params = gate_params + expert_params
            desc = f"MoE (Experts: {n_local_experts}, Active: {active_experts})"

            layers.append(Layer(f"Block {i} - FFN", "feedforward", ffn_params, (hidden_size, intermediate_size), desc))
            total_params += ffn_params

            layers.append(Layer(f"Block {i}", "block_end", 0, (), ""))

        # 3. Final Norm
        layers.append(Layer("Final Norm", "normalization", hidden_size, (hidden_size,)))
        total_params += hidden_size

        # 4. Head
        if getattr(config, 'tie_word_embeddings', False):
            head_params = 0
            desc = "Tied"
        else:
            head_params = vocab_size * hidden_size
            desc = "Untied"
        layers.append(Layer("LM Head", "output", head_params, (hidden_size, vocab_size), desc))
        total_params += head_params

        # 5. MTP Modules (Multi-Token Prediction)
        # Assuming MTP modules are extra small decoder blocks or heads
        if getattr(config, 'use_mtp', False):
            num_mtp = getattr(config, 'num_mtp_modules', 0)

            # MTP usually shares embedding/head but has own transformer layers?
            # Or just extra heads?
            # MiniMax M2 paper/docs might specify. Usually it's like speculative decoding heads.
            # Let's approximate as extra lightweight blocks.

            # For visualization, we list them as special layers
            for m in range(num_mtp):
                 # MTP Block
                 # Assume similar structure but maybe smaller or fewer layers
                 # We'll just add a placeholder block for now
                 layers.append(Layer(f"MTP Module {m}", "adapter", 0, (), f"Speculative Head {m}"))

        return _finalize_architecture(
            Architecture(
                model_type=getattr(config, 'model_type', 'minimax_m2'),
                arch_type="decoder-only",
                total_layers=num_layers,
                total_params=total_params,
                memory_fp16_gb=total_params * 2 / (1024**3),
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

class InternS1Analyzer(TransformerAnalyzer):
    """Intern-S1-Pro Analyzer (Multimodal: Text + Vision + TimeSeries)."""

    def analyze(self, config: Any) -> Architecture:
        # Intern-S1-Pro is a composite model
        # Main config has "vision_config", "ts_config", "text_config"

        # We need to extract the "text_config" (LLM) as the base
        # And attach Vision/TS encoders

        text_config = getattr(config, 'text_config', config) # Fallback if direct
        vision_config = getattr(config, 'vision_config', None)
        ts_config = getattr(config, 'ts_config', None)

        # Base LLM Analysis (using text_config)
        vocab_size = getattr(text_config, 'vocab_size', 0)
        hidden_size = getattr(text_config, 'hidden_size', 0)
        num_layers = getattr(text_config, 'num_hidden_layers', 0)
        num_heads = getattr(text_config, 'num_attention_heads', 0)
        num_kv_heads = getattr(text_config, 'num_key_value_heads', num_heads)

        # MoE?
        # "num_experts": 512, "num_experts_per_tok": 8
        n_experts = getattr(text_config, 'num_experts', 0)
        active_experts = getattr(text_config, 'num_experts_per_tok', 0)

        features = ["Intern-S1-Pro"]
        if n_experts > 0:
            features.append("MoE")
        if vision_config:
            features.append("Vision")
        if ts_config:
            features.append("TimeSeries")

        layers = []
        total_params = 0

        # --- Vision Encoder ---
        if vision_config:
            v_hidden = getattr(vision_config, 'hidden_size', 1024)
            v_layers = getattr(vision_config, 'depth', 24)
            # ViT-Large/24 approx
            v_params = 12 * v_layers * v_hidden * v_hidden
            layers.append(Layer("Vision Encoder", "encoder", v_params, (v_layers, v_hidden), f"ViT-L/{v_layers}"))
            total_params += v_params

            # Projector? Usually a linear or MLP
            mm_hidden = getattr(vision_config, 'out_hidden_size', v_hidden)
            proj_params = v_hidden * mm_hidden
            layers.append(Layer("Vision Projector", "adapter", proj_params, (v_hidden, mm_hidden), "Projection"))
            total_params += proj_params

        # --- Time Series Encoder ---
        if ts_config:
            ts_hidden = getattr(ts_config, 'd_model', 768)
            ts_layers = getattr(ts_config, 'encoder_layers', 17)
            # Encoder-Decoder structure in TS config?
            # "encoder_layers": 17, "decoder_layers": 4
            ts_dec_layers = getattr(ts_config, 'decoder_layers', 4)

            ts_enc_params = 12 * ts_layers * ts_hidden * ts_hidden
            ts_dec_params = 12 * ts_dec_layers * ts_hidden * ts_hidden

            ts_params = ts_enc_params + ts_dec_params
            layers.append(Layer("TimeSeries Model", "encoder", ts_params, (ts_layers + ts_dec_layers, ts_hidden), "TS-Transformer"))
            total_params += ts_params

            # TS Adapter/Projector
            ts_out = getattr(ts_config, 'ts_adapt_out_dim', 1024)
            ts_proj = ts_hidden * ts_out
            layers.append(Layer("TS Projector", "adapter", ts_proj, (ts_hidden, ts_out), "Adapter"))
            total_params += ts_proj

        # --- Language Model ---

        # 1. Embedding
        emb_params = vocab_size * hidden_size
        layers.append(Layer("Token Embedding", "embedding", emb_params, (vocab_size, hidden_size), f"Vocab: {vocab_size}"))
        total_params += emb_params

        # 2. Layers
        for i in range(num_layers):
            layers.append(Layer(f"Block {i}", "block_start", 0, (), ""))

            # Norm 1
            ln_params = hidden_size
            layers.append(Layer(f"Block {i} - Norm 1", "normalization", ln_params, (hidden_size,)))
            total_params += ln_params

            # Attention
            head_dim = getattr(text_config, 'head_dim', hidden_size // num_heads)
            q_params = hidden_size * (num_heads * head_dim)
            k_params = hidden_size * (num_kv_heads * head_dim)
            v_params = hidden_size * (num_kv_heads * head_dim)
            o_params = (num_heads * head_dim) * hidden_size

            attn_params = q_params + k_params + v_params + o_params
            layers.append(Layer(f"Block {i} - Attention", "attention", attn_params, (hidden_size, hidden_size), "GQA"))
            total_params += attn_params

            # Norm 2
            layers.append(Layer(f"Block {i} - Norm 2", "normalization", ln_params, (hidden_size,)))
            total_params += ln_params

            # FFN / MoE
            # InternLM MoE: Gate + Experts
            intermediate_size = getattr(text_config, 'intermediate_size', 0)
            moe_inter_size = getattr(text_config, 'moe_intermediate_size', intermediate_size)

            # Check if this layer is MoE?
            # "mlp_only_layers": [] -> all MoE? or implied by num_experts > 0
            # Assuming all layers MoE if num_experts > 0

            if n_experts > 0:
                gate_params = hidden_size * n_experts
                expert_params = n_experts * (3 * hidden_size * moe_inter_size)
                ffn_params = gate_params + expert_params
                desc = f"MoE (Experts: {n_experts}, Active: {active_experts})"
            else:
                ffn_params = 3 * hidden_size * intermediate_size
                desc = f"Dense (Inter: {intermediate_size})"

            layers.append(Layer(f"Block {i} - FFN", "feedforward", ffn_params, (hidden_size, moe_inter_size), desc))
            total_params += ffn_params

            layers.append(Layer(f"Block {i}", "block_end", 0, (), ""))

        # 3. Final Norm
        layers.append(Layer("Final Norm", "normalization", hidden_size, (hidden_size,)))
        total_params += hidden_size

        # 4. Head
        if getattr(text_config, 'tie_word_embeddings', False):
            head_params = 0
            desc = "Tied"
        else:
            head_params = vocab_size * hidden_size
            desc = "Untied"
        layers.append(Layer("LM Head", "output", head_params, (hidden_size, vocab_size), desc))
        total_params += head_params

        return _finalize_architecture(
            Architecture(
                model_type="interns1_pro", # Composite
                arch_type="encoder-decoder", # Multimodal
                total_layers=num_layers,
                total_params=total_params,
                memory_fp16_gb=total_params * 2 / (1024**3),
                parameters={
                    "vocab_size": vocab_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "vision_layers": int(getattr(vision_config, "depth", 0) or 0),
                    "ts_encoder_layers": int(getattr(ts_config, "encoder_layers", 0) or 0),
                    "ts_decoder_layers": int(getattr(ts_config, "decoder_layers", 0) or 0),
                },
                features=features,
                layers=layers,
            ),
            total_layers=num_layers,
        )

class Qwen35Analyzer(TransformerAnalyzer):
    """Qwen3.5 MoE (A3B) Analyzer."""

    def analyze(self, config: Any) -> Architecture:
        def _get(obj: Any, key: str, default: Any = 0) -> Any:
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        # Qwen3.5 MoE has "text_config" and "vision_config"
        text_config = getattr(config, 'text_config', config)
        vision_config = getattr(config, 'vision_config', None)

        # Base
        vocab_size = int(_get(text_config, 'vocab_size', 0) or 0)
        hidden_size = int(_get(text_config, 'hidden_size', 0) or 0)
        num_layers = int(_get(text_config, 'num_hidden_layers', 0) or 0)
        num_heads = int(_get(text_config, 'num_attention_heads', 0) or 0)
        if num_heads <= 0 and hidden_size:
            num_heads = 1
        num_kv_heads = int(_get(text_config, 'num_key_value_heads', num_heads) or 0) or num_heads

        # MoE
        n_experts = int(_get(text_config, 'num_experts', 0) or 0)
        active_experts = int(_get(text_config, 'num_experts_per_tok', 0) or 0)

        # Layer Types (Linear vs Full Attention)
        layer_types = _get(text_config, 'layer_types', []) or []

        features = ["Qwen3.5", "MoE"]
        if vision_config:
            features.append("Vision")
        if "linear_attention" in layer_types:
            features.append("Linear Attention (Mamba/SSM?)")

        layers = []
        total_params = 0

        # --- Vision ---
        if vision_config:
            v_hidden = getattr(vision_config, 'hidden_size', 0)
            v_layers = getattr(vision_config, 'depth', 0)
            v_params = 12 * v_layers * v_hidden * v_hidden # approx
            layers.append(Layer("Vision Encoder", "encoder", v_params, (v_layers, v_hidden), "Qwen-Vision"))
            total_params += v_params

        # --- Text ---
        # 1. Embedding
        emb_params = vocab_size * hidden_size
        layers.append(Layer("Token Embedding", "embedding", emb_params, (vocab_size, hidden_size), f"Vocab: {vocab_size}"))
        total_params += emb_params

        # 2. Layers
        for i in range(num_layers):
            layers.append(Layer(f"Block {i}", "block_start", 0, (), ""))

            # Norm 1
            ln_params = hidden_size
            layers.append(Layer(f"Block {i} - Norm 1", "normalization", ln_params, (hidden_size,)))
            total_params += ln_params

            # Attention (Linear or Full)
            l_type = layer_types[i] if i < len(layer_types) else "full_attention"

            head_dim = _get(text_config, 'head_dim', None)
            head_dim = int(head_dim or 0)
            if head_dim <= 0:
                head_dim = (hidden_size // num_heads) if num_heads else 0
            q_params = hidden_size * (num_heads * head_dim)
            k_params = hidden_size * (num_kv_heads * head_dim)
            v_params = hidden_size * (num_kv_heads * head_dim)
            o_params = (num_heads * head_dim) * hidden_size
            attn_params = q_params + k_params + v_params + o_params

            attn_desc = "Full Attn" if l_type == "full_attention" else "Linear Attn"
            layers.append(Layer(f"Block {i} - Attention", "attention", attn_params, (hidden_size, hidden_size), attn_desc))
            total_params += attn_params

            # Norm 2
            layers.append(Layer(f"Block {i} - Norm 2", "normalization", ln_params, (hidden_size,)))
            total_params += ln_params

            # MoE FFN
            moe_inter = int(_get(text_config, 'moe_intermediate_size', 0) or 0)
            shared_inter = int(_get(text_config, 'shared_expert_intermediate_size', 0) or 0)

            # Gate
            gate_params = hidden_size * n_experts
            # Routed (SwiGLU)
            routed_params = n_experts * (3 * hidden_size * moe_inter)
            # Shared (SwiGLU)
            # Is shared expert enabled? usually yes if size > 0
            shared_params = 3 * hidden_size * shared_inter

            ffn_params = gate_params + routed_params + shared_params
            desc = f"MoE (Routed: {n_experts}, Active: {active_experts}, Shared Inter: {shared_inter})"

            layers.append(Layer(f"Block {i} - FFN", "feedforward", ffn_params, (hidden_size, moe_inter), desc))
            total_params += ffn_params

            layers.append(Layer(f"Block {i}", "block_end", 0, (), ""))

        # 3. Final Norm
        layers.append(Layer("Final Norm", "normalization", hidden_size, (hidden_size,)))
        total_params += hidden_size

        # 4. Head
        if getattr(text_config, 'tie_word_embeddings', False):
            head_params = 0
            desc = "Tied"
        else:
            head_params = vocab_size * hidden_size
            desc = "Untied"
        layers.append(Layer("LM Head", "output", head_params, (hidden_size, vocab_size), desc))
        total_params += head_params

        return _finalize_architecture(
            Architecture(
                model_type="qwen3_5_moe",
                arch_type="decoder-only",
                total_layers=num_layers,
                total_params=total_params,
                memory_fp16_gb=total_params * 2 / (1024**3),
                parameters={
                    "vocab_size": vocab_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "vision_layers": int(getattr(vision_config, "depth", 0) or 0),
                },
                features=features,
                layers=layers,
            ),
            total_layers=num_layers,
        )


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

class AnalyzerRegistry:
    _analyzers = {
        "default": TransformerAnalyzer(),
        "bert": BertAnalyzer(),
        "roberta": BertAnalyzer(),
        "t5": T5Analyzer(),
        "bart": T5Analyzer(),
        "bloom": BloomAnalyzer(),
        "gptneox": GPTNeoXAnalyzer(),
        "gpt_neox": GPTNeoXAnalyzer(),
        "gpt_bigcode": StarCoderAnalyzer(),
        "starcoder2": StarCoder2Analyzer(),
        "falcon": FalconAnalyzer(),
        "opt": OPTAnalyzer(),
        "qwen": QwenAnalyzer(),
        "qwen2": QwenAnalyzer(),
        "qwen2_moe": Qwen2MoeAnalyzer(),
        "deepseek": DeepSeekAnalyzer(),
        "deepseek_v3": DeepSeekAnalyzer(),
        "deepseek_v4": DeepSeekAnalyzer(),
        "llama": LlamaAnalyzer(),
        "mistral": MistralAnalyzer(),
        "mixtral": MistralAnalyzer(),
        "gemma": GemmaAnalyzer(),
        "gemma2": GemmaAnalyzer(),
        "gemma3": GemmaAnalyzer(),
        "gemma3_text": GemmaAnalyzer(),
        "gemma4": GemmaAnalyzer(),
        "gemma4_text": GemmaAnalyzer(),
        "phi": PhiAnalyzer(),
        "phi1": PhiAnalyzer(),
        "phi2": PhiAnalyzer(),
        "phi3": PhiAnalyzer(),
        "phi4": PhiAnalyzer(),
        "cohere": CohereAnalyzer(),
        "cohere2": CohereAnalyzer(),
        "stablelm": StableLMAnalyzer(),
        "stablelm_epoch": StableLMAnalyzer(),
        "stableplankton": StableLMAnalyzer(),
        "yi": YiAnalyzer(),
        "internlm": InternLMAnalyzer(),
        "internlm2": InternLMAnalyzer(),
        "internlm3": InternLMAnalyzer(),
        "baichuan": BaichuanAnalyzer(),
        "glm": GLMAnalyzer(),
        "glm4": GLMAnalyzer(),
        "glm5": GLMAnalyzer(),
        "glm_moe_dsa": GLMAnalyzer(),
        "chatglm": GLMAnalyzer(),
        "kimi_k25": KimiAnalyzer(),
        "ernie4_5_moe_vl": ErnieAnalyzer(),
        "gpt2": GPT2Analyzer(),
        "minimax_m2": MiniMaxAnalyzer(),
        "interns1_pro": InternS1Analyzer(),
        "qwen3_5_moe": Qwen35Analyzer(), # Add Qwen3.5 support
        "qwen3_5_moe_text": Qwen35Analyzer(),
        "hy_v3": Hy3Analyzer(),
        "mamba": SequenceMixerAnalyzer(),
        "mamba2": SequenceMixerAnalyzer(),
        "rwkv": SequenceMixerAnalyzer(),
        "retnet": SequenceMixerAnalyzer(),
        "hyena": SequenceMixerAnalyzer(),
    }

    _architecture_aliases = [
        ("llama", "llama"),
        ("mistral", "mistral"),
        ("mixtral", "mixtral"),
        ("gemma", "gemma"),
        ("phi", "phi"),
        ("cohere", "cohere"),
        ("bloom", "bloom"),
        ("gptneox", "gpt_neox"),
        ("gpt_neox", "gpt_neox"),
        ("starcoder2", "starcoder2"),
        ("starcoder", "gpt_bigcode"),
        ("bigcode", "gpt_bigcode"),
        ("falcon", "falcon"),
        ("opt", "opt"),
        ("yi", "yi"),
        ("internlm", "internlm2"),
        ("baichuan", "baichuan"),
        ("bert", "bert"),
        ("roberta", "roberta"),
        ("t5", "t5"),
        ("bart", "bart"),
        ("glm", "glm"),
        ("chatglm", "chatglm"),
        ("qwen2moe", "qwen2_moe"),
        ("qwen3_5moe", "qwen3_5_moe"),
        ("qwen3_5", "qwen3_5_moe"),
        ("qwen", "qwen"),
        ("deepseek", "deepseek"),
        ("kimi", "kimi_k25"),
        ("ernie", "ernie4_5_moe_vl"),
        ("intern", "interns1_pro"),
        ("minimax", "minimax_m2"),
        ("hyv3", "hy_v3"),
        ("hy_v3", "hy_v3"),
        ("mamba", "mamba"),
        ("rwkv", "rwkv"),
        ("retnet", "retnet"),
        ("hyena", "hyena"),
    ]

    @classmethod
    def _resolve_key(cls, model_type: str) -> str:
        normalized = str(model_type or "").lower()
        if normalized in cls._analyzers:
            return normalized

        family_prefixes = [
            ("qwen3_5", "qwen3_5_moe"),
            ("qwen2_moe", "qwen2_moe"),
            ("qwen2", "qwen2"),
            ("qwen", "qwen"),
            ("gemma", "gemma"),
            ("phi", "phi"),
            ("cohere", "cohere"),
            ("bloom", "bloom"),
            ("gptneox", "gpt_neox"),
            ("gpt_neox", "gpt_neox"),
            ("starcoder2", "starcoder2"),
            ("starcoder", "gpt_bigcode"),
            ("stablelm", "stablelm"),
            ("yi", "yi"),
            ("internlm", "internlm2"),
            ("baichuan", "baichuan"),
            ("glm", "glm"),
            ("mamba2", "mamba2"),
            ("mamba", "mamba"),
            ("rwkv", "rwkv"),
            ("retnet", "retnet"),
            ("hyena", "hyena"),
        ]
        for prefix, key in family_prefixes:
            if normalized.startswith(prefix):
                return key

        family_contains = [
            ("llama", "llama"),
            ("mistral", "mistral"),
            ("mixtral", "mixtral"),
            ("deepseek", "deepseek"),
            ("kimi", "kimi_k25"),
            ("chatglm", "chatglm"),
            ("bloom", "bloom"),
            ("neox", "gpt_neox"),
            ("starcoder2", "starcoder2"),
            ("starcoder", "gpt_bigcode"),
            ("bigcode", "gpt_bigcode"),
            ("yi", "yi"),
            ("internlm", "internlm2"),
            ("baichuan", "baichuan"),
            ("minimax", "minimax_m2"),
            ("intern", "interns1_pro"),
            ("ernie", "ernie4_5_moe_vl"),
            ("hy3", "hy_v3"),
            ("hunyuan", "hy_v3"),
            ("mamba", "mamba"),
            ("rwkv", "rwkv"),
            ("retnet", "retnet"),
            ("hyena", "hyena"),
        ]
        for token, key in family_contains:
            if token in normalized:
                return key

        return "default"

    @classmethod
    def get(cls, model_type: str) -> ModelAnalyzer:
        return cls._analyzers[cls._resolve_key(model_type)]

    @classmethod
    def _resolve_architecture_key(cls, architecture_name: str) -> str:
        normalized = str(architecture_name or "").replace("-", "_").lower()
        for token, key in cls._architecture_aliases:
            if token in normalized:
                return key
        return "default"

    @classmethod
    def resolve(cls, config: Any) -> ModelAnalyzer:
        candidates = [
            _cfg_get(config, "model_type", ""),
            _cfg_get(getattr(config, "text_config", None), "model_type", ""),
        ]
        for candidate in candidates:
            key = cls._resolve_key(str(candidate or ""))
            if key != "default":
                return cls._analyzers[key]

        for architecture_name in _architectures(config):
            key = cls._resolve_architecture_key(architecture_name)
            if key != "default":
                return cls._analyzers[key]

        return cls._analyzers["default"]
