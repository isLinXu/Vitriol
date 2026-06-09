"""Shared config-introspection helpers for architecture analyzers."""
import logging
from typing import Any, Optional

from ..core import Architecture

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
        except Exception as exc:
            logger.debug("to_dict() failed on %s: %s", type(obj).__name__, exc)
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
