"""
KV Cache Policy Definitions.

Provides policy classes for configuring KV cache behavior including:
- SafeExactPolicy: Exact attention without approximation
- Turbo3ExactKApproxVPolicy: Turbo3-style K approximation with exact V
- Various preset policies for different quality/performance trade-offs

Policies control:
- Quantization modes (exact vs approximate)
- Layer-specific handling (sliding window, MLA, full attention)
- Sparse V and compute skip options
- Turbo residual QJL settings
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .cache_store import KVCacheStoreConfig


class ApproxMode(str, Enum):
    EXACT = "exact"
    APPROX = "approx"


@dataclass(frozen=True)
class KVPolicy:
    mode: ApproxMode


class KVLayerType(str, Enum):
    FULL_ATTENTION = "full_attention"
    SLIDING_WINDOW = "sliding_window"
    MLA = "mla"
    COMPRESSED_ATTENTION = "compressed_attention"
    HASH_ATTENTION = "hash_attention"
    LINEAR = "linear_attention"
    OTHER = "other"


@dataclass(frozen=True)
class SafeExactPolicy(KVPolicy):
    def __init__(self) -> None:
        super().__init__(mode=ApproxMode.EXACT)


@dataclass(frozen=True)
class Turbo3ExactKApproxVPolicy(KVPolicy):
    v_quantize_only_first_n_full_attention_layers: int = 1
    v_protect_last_n_full_attention_layers: int = 0
    turbo_k_format: str = "turbo3"
    turbo_v_format: str = "turbo3"
    turbo_block_size: int = 32
    quantized_kv_start: int = 0
    enable_turbo_residual_qjl: bool = True
    turbo_residual_strength: float = 0.5
    enable_sparse_v: bool = False
    sparse_v_only_first_n_full_attention_layers: int = 0
    enable_compute_skip: bool = False
    compute_skip_only_first_n_full_attention_layers: int = 0

    def __init__(
        self,
        v_quantize_only_first_n_full_attention_layers: int = 1,
        v_protect_last_n_full_attention_layers: int = 0,
        turbo_k_format: str = "turbo3",
        turbo_v_format: str = "turbo3",
        turbo_block_size: int = 32,
        quantized_kv_start: int = 0,
        enable_turbo_residual_qjl: bool = True,
        turbo_residual_strength: float = 0.5,
        enable_sparse_v: bool = False,
        sparse_v_only_first_n_full_attention_layers: int = 0,
        enable_compute_skip: bool = False,
        compute_skip_only_first_n_full_attention_layers: int = 0,
    ) -> None:
        super().__init__(mode=ApproxMode.APPROX)
        object.__setattr__(self, "v_quantize_only_first_n_full_attention_layers", int(v_quantize_only_first_n_full_attention_layers))
        object.__setattr__(self, "v_protect_last_n_full_attention_layers", int(v_protect_last_n_full_attention_layers))
        object.__setattr__(self, "turbo_k_format", str(turbo_k_format))
        object.__setattr__(self, "turbo_v_format", str(turbo_v_format))
        object.__setattr__(self, "turbo_block_size", int(turbo_block_size))
        object.__setattr__(self, "quantized_kv_start", int(quantized_kv_start))
        object.__setattr__(self, "enable_turbo_residual_qjl", bool(enable_turbo_residual_qjl))
        object.__setattr__(self, "turbo_residual_strength", float(turbo_residual_strength))
        object.__setattr__(self, "enable_sparse_v", bool(enable_sparse_v))
        object.__setattr__(self, "sparse_v_only_first_n_full_attention_layers", int(sparse_v_only_first_n_full_attention_layers))
        object.__setattr__(self, "enable_compute_skip", bool(enable_compute_skip))
        object.__setattr__(self, "compute_skip_only_first_n_full_attention_layers", int(compute_skip_only_first_n_full_attention_layers))


@dataclass(frozen=True)
class KVLayerStrategy:
    layer_type: KVLayerType
    turbo_quantize_k: bool
    turbo_quantize_v: bool
    enable_sparse_v: bool
    enable_compute_skip: bool


def _classify_kv_layer_name(name: Any) -> KVLayerType:
    normalized = str(name or "").lower().replace("-", "_")
    if normalized in {"", "none"}:
        return KVLayerType.FULL_ATTENTION
    if "sliding" in normalized or "local_attention" in normalized or normalized in {"local", "window_attention"}:
        return KVLayerType.SLIDING_WINDOW
    if "mla" in normalized or "array" in normalized or "latent" in normalized:
        return KVLayerType.MLA
    if "hash" in normalized:
        return KVLayerType.HASH_ATTENTION
    if "compressed" in normalized or "csa" in normalized or "hca" in normalized:
        return KVLayerType.COMPRESSED_ATTENTION
    if any(token in normalized for token in ("linear", "mamba", "ssm", "state_space", "rwkv", "retnet", "hyena")):
        return KVLayerType.LINEAR
    if normalized in {
        "full",
        "global",
        "global_attention",
        "full_attention",
        "self_attention",
        "attention",
        "mha",
        "gqa",
        "mqa",
    }:
        return KVLayerType.FULL_ATTENTION
    if "full" in normalized or "global" in normalized:
        return KVLayerType.FULL_ATTENTION
    return KVLayerType.OTHER


def classify_kv_layer(handle: Any, layer_idx: int) -> KVLayerType:
    layer_types = getattr(handle, "layer_types", None)
    idx = int(layer_idx)
    if layer_types is None or idx < 0 or idx >= len(layer_types):
        return KVLayerType.FULL_ATTENTION

    return _classify_kv_layer_name(layer_types[idx])


def _full_attention_layers(handle: Any) -> list[int]:
    layer_types = getattr(handle, "layer_types", None)
    if layer_types is None:
        count = len(handle) if hasattr(handle, "__len__") else 0
        return list(range(count))
    return [i for i, name in enumerate(layer_types) if _classify_kv_layer_name(name) is KVLayerType.FULL_ATTENTION]


def _full_attention_pos(handle: Any, layer_idx: int) -> Optional[int]:
    full_layers = _full_attention_layers(handle)
    try:
        return full_layers.index(int(layer_idx))
    except ValueError:
        return None


def resolve_layer_strategy(policy: KVPolicy, handle: Any, layer_idx: int) -> KVLayerStrategy:
    layer_type = classify_kv_layer(handle, layer_idx)
    if policy.mode == ApproxMode.EXACT:
        return KVLayerStrategy(layer_type, False, False, False, False)

    if not isinstance(policy, Turbo3ExactKApproxVPolicy):
        return KVLayerStrategy(layer_type, False, False, False, False)

    if layer_type is not KVLayerType.FULL_ATTENTION:
        return KVLayerStrategy(layer_type, False, False, False, False)

    pos = _full_attention_pos(handle, layer_idx)
    full_layers = _full_attention_layers(handle)
    protect_from = max(0, len(full_layers) - int(policy.v_protect_last_n_full_attention_layers))

    turbo_quantize_v = True
    if pos is None:
        turbo_quantize_v = False
    elif int(policy.v_quantize_only_first_n_full_attention_layers) > 0:
        turbo_quantize_v = pos < int(policy.v_quantize_only_first_n_full_attention_layers)
    if pos is not None and pos >= protect_from:
        turbo_quantize_v = False

    enable_sparse_v = bool(policy.enable_sparse_v)
    if enable_sparse_v and int(policy.sparse_v_only_first_n_full_attention_layers) > 0:
        enable_sparse_v = pos is not None and pos < int(policy.sparse_v_only_first_n_full_attention_layers)

    enable_compute_skip = bool(policy.enable_compute_skip)
    if enable_compute_skip and int(policy.compute_skip_only_first_n_full_attention_layers) > 0:
        enable_compute_skip = pos is not None and pos < int(policy.compute_skip_only_first_n_full_attention_layers)

    return KVLayerStrategy(layer_type, True, turbo_quantize_v, enable_sparse_v, enable_compute_skip)


def apply_policy_to_store_cfg(base_cfg: "KVCacheStoreConfig", policy: KVPolicy, handle: Any, layer_idx: int) -> "KVCacheStoreConfig":
    strategy = resolve_layer_strategy(policy, handle, layer_idx)
    overrides = {
        "turbo_quantize_k": strategy.turbo_quantize_k,
        "turbo_quantize_v": strategy.turbo_quantize_v,
        "enable_sparse_v": strategy.enable_sparse_v,
        "enable_compute_skip": strategy.enable_compute_skip,
        "quantized_kv_start": getattr(policy, "quantized_kv_start", base_cfg.quantized_kv_start),
        "enable_turbo_residual_qjl": getattr(policy, "enable_turbo_residual_qjl", base_cfg.enable_turbo_residual_qjl),
        "turbo_residual_strength": getattr(policy, "turbo_residual_strength", base_cfg.turbo_residual_strength),
    }
    # ── New: Propagate temporal pooling settings from policy if present ──
    if hasattr(policy, "enable_temporal_pooling"):
        overrides["enable_temporal_pooling"] = policy.enable_temporal_pooling
    if hasattr(policy, "enable_sliding_window_eviction"):
        overrides["enable_sliding_window_eviction"] = policy.enable_sliding_window_eviction
    if hasattr(policy, "enable_zero_copy_decode"):
        overrides["enable_zero_copy_decode"] = policy.enable_zero_copy_decode
    if hasattr(policy, "enable_layer_adaptive"):
        overrides["enable_layer_adaptive"] = policy.enable_layer_adaptive
    if hasattr(policy, "enable_spectral_kv"):
        overrides["enable_spectral_kv"] = policy.enable_spectral_kv
    if hasattr(policy, "spectral_target_bpv"):
        overrides["spectral_target_bpv"] = policy.spectral_target_bpv
    if hasattr(policy, "enable_predictive_kv"):
        overrides["enable_predictive_kv"] = policy.enable_predictive_kv
    if hasattr(policy, "predictive_target_bpv"):
        overrides["predictive_target_bpv"] = policy.predictive_target_bpv
    if hasattr(policy, "enable_cross_layer_kv"):
        overrides["enable_cross_layer_kv"] = policy.enable_cross_layer_kv
    if hasattr(policy, "cross_layer_target_bpv"):
        overrides["cross_layer_target_bpv"] = policy.cross_layer_target_bpv
    if hasattr(policy, "cross_layer_iframe_interval"):
        overrides["cross_layer_iframe_interval"] = policy.cross_layer_iframe_interval
    if hasattr(policy, "enable_attention_gated_kv"):
        overrides["enable_attention_gated_kv"] = policy.enable_attention_gated_kv
    if hasattr(policy, "attention_gated_target_bpv"):
        overrides["attention_gated_target_bpv"] = policy.attention_gated_target_bpv
    if hasattr(policy, "enable_dict_kv"):
        overrides["enable_dict_kv"] = policy.enable_dict_kv
    if hasattr(policy, "dict_kv_n_atoms"):
        overrides["dict_kv_n_atoms"] = policy.dict_kv_n_atoms
    if hasattr(policy, "dict_kv_sparsity"):
        overrides["dict_kv_sparsity"] = policy.dict_kv_sparsity
    return replace(base_cfg, **overrides)


@dataclass(frozen=True)
class KVPolicyPreset:
    name: str
    policy_type: str
    params: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "policy_type": self.policy_type, "params": dict(self.params)}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "KVPolicyPreset":
        return KVPolicyPreset(name=str(d["name"]), policy_type=str(d["policy_type"]), params=dict(d.get("params", {})))

    @staticmethod
    def safe_default() -> "KVPolicyPreset":
        return KVPolicyPreset(name="safe", policy_type="SafeExactPolicy", params={})

    @staticmethod
    def balanced_default() -> "KVPolicyPreset":
        return KVPolicyPreset(
            name="balanced",
            policy_type="Turbo3ExactKApproxVPolicy",
            params={"v_quantize_only_first_n_full_attention_layers": 1, "quantized_kv_start": 2048},
        )

    @staticmethod
    def fast_balanced_default() -> "KVPolicyPreset":
        return KVPolicyPreset(
            name="fast-balanced",
            policy_type="Turbo3ExactKApproxVPolicy",
            params={
                "v_quantize_only_first_n_full_attention_layers": 1,
                "quantized_kv_start": 2048,
                "enable_turbo_residual_qjl": False,
            },
        )

    @staticmethod
    def aggressive_default() -> "KVPolicyPreset":
        return KVPolicyPreset(
            name="aggressive",
            policy_type="Turbo3ExactKApproxVPolicy",
            params={
                "v_quantize_only_first_n_full_attention_layers": 6,
                "quantized_kv_start": 1024,
                "enable_sparse_v": True,
                "sparse_v_only_first_n_full_attention_layers": 4,
            },
        )

    @staticmethod
    def ultra_long_default() -> "KVPolicyPreset":
        return KVPolicyPreset(
            name="ultra-long",
            policy_type="Turbo3ExactKApproxVPolicy",
            params={
                "v_quantize_only_first_n_full_attention_layers": 8,
                "v_protect_last_n_full_attention_layers": 1,
                "quantized_kv_start": 512,
                "enable_sparse_v": True,
                "sparse_v_only_first_n_full_attention_layers": 6,
                "enable_compute_skip": True,
                "compute_skip_only_first_n_full_attention_layers": 4,
            },
        )

    @staticmethod
    def deepseek_v4_default() -> "KVPolicyPreset":
        """DeepSeek-V4 preset: compression-aware and conservative for CSA/HCA/hash layers."""
        return KVPolicyPreset(
            name="deepseek-v4",
            policy_type="Turbo3ExactKApproxVPolicy",
            params={
                "v_quantize_only_first_n_full_attention_layers": 1,
                "v_protect_last_n_full_attention_layers": 1,
                "quantized_kv_start": 0,
                "enable_sparse_v": False,
                "enable_compute_skip": False,
            },
        )

    @staticmethod
    def hy3_default() -> "KVPolicyPreset":
        """Hy3 preset: long-context GQA/MoE-friendly KV compression with protected tail layers."""
        return KVPolicyPreset(
            name="hy3",
            policy_type="Turbo3ExactKApproxVPolicy",
            params={
                "v_quantize_only_first_n_full_attention_layers": 4,
                "v_protect_last_n_full_attention_layers": 2,
                "quantized_kv_start": 1024,
                "enable_sparse_v": True,
                "sparse_v_only_first_n_full_attention_layers": 2,
                "enable_compute_skip": False,
            },
        )

    @staticmethod
    def smart_default() -> "KVPolicyPreset":
        """Smart preset: uses Temporal Pooling + Zero-Copy Decode + Sliding Window."""
        return KVPolicyPreset(
            name="smart",
            policy_type="Turbo3ExactKApproxVPolicy",
            params={
                "v_quantize_only_first_n_full_attention_layers": 4,
                "quantized_kv_start": 1024,
                "enable_temporal_pooling": True,
                "enable_zero_copy_decode": True,
                "enable_sliding_window_eviction": True,
            },
        )

    @staticmethod
    def spectral_default() -> "KVPolicyPreset":
        """SpectralKV preset: frequency-aware compression for better quality at same bpv."""
        return KVPolicyPreset(
            name="spectral",
            policy_type="Turbo3ExactKApproxVPolicy",
            params={
                "v_quantize_only_first_n_full_attention_layers": 4,
                "quantized_kv_start": 1024,
                "enable_spectral_kv": True,
                "spectral_target_bpv": 3.0,
            },
        )

    @staticmethod
    def predictive_default() -> "KVPolicyPreset":
        """PredictiveKV preset: linear-prediction residual coding for temporal correlation."""
        return KVPolicyPreset(
            name="predictive",
            policy_type="Turbo3ExactKApproxVPolicy",
            params={
                "v_quantize_only_first_n_full_attention_layers": 4,
                "quantized_kv_start": 1024,
                "enable_predictive_kv": True,
                "predictive_target_bpv": 3.0,
            },
        )

    @staticmethod
    def spectral_predictive_default() -> "KVPolicyPreset":
        """Combined SpectralKV + PredictiveKV: maximum compression quality."""
        return KVPolicyPreset(
            name="spectral-predictive",
            policy_type="Turbo3ExactKApproxVPolicy",
            params={
                "v_quantize_only_first_n_full_attention_layers": 4,
                "quantized_kv_start": 1024,
                "enable_spectral_kv": True,
                "spectral_target_bpv": 3.0,
                "enable_predictive_kv": True,
                "predictive_target_bpv": 3.0,
            },
        )

    @staticmethod
    def cross_layer_default() -> "KVPolicyPreset":
        """CrossLayerKV: cross-layer differential compression for depth correlation."""
        return KVPolicyPreset(
            name="cross-layer",
            policy_type="Turbo3ExactKApproxVPolicy",
            params={
                "v_quantize_only_first_n_full_attention_layers": 4,
                "quantized_kv_start": 1024,
                "enable_cross_layer_kv": True,
                "cross_layer_target_bpv": 2.4,
                "cross_layer_iframe_interval": 4,
            },
        )

    @staticmethod
    def cross_layer_spectral_default() -> "KVPolicyPreset":
        """CrossLayerKV + SpectralKV: depth correlation + frequency compression."""
        return KVPolicyPreset(
            name="cross-layer-spectral",
            policy_type="Turbo3ExactKApproxVPolicy",
            params={
                "v_quantize_only_first_n_full_attention_layers": 4,
                "quantized_kv_start": 1024,
                "enable_cross_layer_kv": True,
                "cross_layer_target_bpv": 2.4,
                "enable_spectral_kv": True,
                "spectral_target_bpv": 3.0,
            },
        )

    @staticmethod
    def ultimate_default() -> "KVPolicyPreset":
        """Ultimate: CrossLayer + Predictive + Spectral — maximum compression."""
        return KVPolicyPreset(
            name="ultimate",
            policy_type="Turbo3ExactKApproxVPolicy",
            params={
                "v_quantize_only_first_n_full_attention_layers": 4,
                "quantized_kv_start": 1024,
                "enable_cross_layer_kv": True,
                "cross_layer_target_bpv": 2.0,
                "cross_layer_iframe_interval": 4,
                "enable_predictive_kv": True,
                "predictive_target_bpv": 3.0,
                "enable_spectral_kv": True,
                "spectral_target_bpv": 3.0,
            },
        )

    @staticmethod
    def attention_gated_default() -> "KVPolicyPreset":
        """AttentionGatedKV: attention-importance-driven variable precision."""
        return KVPolicyPreset(
            name="attention-gated",
            policy_type="Turbo3ExactKApproxVPolicy",
            params={
                "v_quantize_only_first_n_full_attention_layers": 4,
                "quantized_kv_start": 1024,
                "enable_attention_gated_kv": True,
                "attention_gated_target_bpv": 2.4,
            },
        )


def list_policy_presets() -> list[KVPolicyPreset]:
    presets = [
        KVPolicyPreset.safe_default(),
        KVPolicyPreset.balanced_default(),
        KVPolicyPreset.fast_balanced_default(),
        KVPolicyPreset.aggressive_default(),
        KVPolicyPreset.ultra_long_default(),
        KVPolicyPreset.deepseek_v4_default(),
        KVPolicyPreset.hy3_default(),
        KVPolicyPreset.smart_default(),
        KVPolicyPreset.spectral_default(),
        KVPolicyPreset.predictive_default(),
        KVPolicyPreset.spectral_predictive_default(),
        KVPolicyPreset.cross_layer_default(),
        KVPolicyPreset.cross_layer_spectral_default(),
        KVPolicyPreset.ultimate_default(),
        KVPolicyPreset.attention_gated_default(),
    ]
    # Add TurboQuantum presets if available
    try:
        from .turboquantum import get_turboquantum_presets
        for tq_preset in get_turboquantum_presets():
            presets.append(KVPolicyPreset(
                name=tq_preset["name"],
                policy_type="TurboQuantum",
                params={"mode_config": None},  # Resolved at build time
            ))
    except ImportError:
        pass
    return presets


def build_policy(preset: KVPolicyPreset) -> KVPolicy:
    if preset.policy_type == "SafeExactPolicy":
        return SafeExactPolicy()
    if preset.policy_type == "Turbo3ExactKApproxVPolicy":
        # Filter params that Turbo3ExactKApproxVPolicy accepts
        # Extra params (like enable_spectral_kv) are stored in the policy
        # and propagated via apply_policy_to_store_cfg using hasattr/getattr
        known_keys = {
            "v_quantize_only_first_n_full_attention_layers",
            "v_protect_last_n_full_attention_layers",
            "turbo_k_format", "turbo_v_format", "turbo_block_size",
            "quantized_kv_start", "enable_turbo_residual_qjl",
            "turbo_residual_strength", "enable_sparse_v",
            "sparse_v_only_first_n_full_attention_layers",
            "enable_compute_skip",
            "compute_skip_only_first_n_full_attention_layers",
        }
        turbo_params = {k: v for k, v in preset.params.items() if k in known_keys}
        policy = Turbo3ExactKApproxVPolicy(**turbo_params)
        # Attach extra params as dynamic attributes for apply_policy_to_store_cfg
        extra_params = {k: v for k, v in preset.params.items() if k not in known_keys}
        for k, v in extra_params.items():
            object.__setattr__(policy, k, v)
        return policy
    if preset.policy_type == "TurboQuantum":
        # TurboQuantum: store config name, resolve in codec layer
        return preset  # Return as-is; codec layer handles it
    raise ValueError(f"Unknown policy_type={preset.policy_type}")
