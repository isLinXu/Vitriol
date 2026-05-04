from .codec import (
    AdaptiveKVCodec,
    ComputeSkipConfig,
    ComputeSkipResult,
    _vectorized_blockwise_qdq as _vectorized_blockwise_qdq,
    adaptive_kv_bits,
    clear_projection_cache as clear_projection_cache,
    compute_skip_attention,
    kv_bytes_per_value,
)
try:
    from .triton_kernels import get_backend_name, triton_fwht, _HAS_TRITON
except ImportError:
    def get_backend_name():
        return "python"
    triton_fwht = None
    _HAS_TRITON = False
from .cache_store import KVCacheStore, KVCacheStoreConfig
from .backend import KVMeta, KVStoreBackend
from .policy import (
    ApproxMode,
    KVLayerStrategy,
    KVLayerType,
    KVPolicyPreset,
    SafeExactPolicy,
    Turbo3ExactKApproxVPolicy,
    apply_policy_to_store_cfg,
    build_policy,
    classify_kv_layer,
    list_policy_presets,
    resolve_layer_strategy,
)
try:
    from .turboquantum import (
        TurboQuantumConfig,
        TurboQuantumResult,
        TurboQuantumCodec,
        turboquantum_compress,
        create_turboquantum_codec,
        compute_attention_entropy as turboquantum_entropy,
        get_turboquantum_presets,
    )
except ImportError:
    TurboQuantumConfig = None
    TurboQuantumResult = None
    TurboQuantumCodec = None
    turboquantum_compress = None
    create_turboquantum_codec = None
    turboquantum_entropy = None
    get_turboquantum_presets = None

# ── Layer-Aware Adaptive Bit Allocation ──
from .layer_adaptive import (
    LayerAdaptiveBitAllocator,
    LayerAdaptiveConfig,
    apply_layer_adaptive_to_config,
)

# ── Temporal Importance Pooling ──
from .temporal_pooling import (
    TemporalPoolingConfig,
    temporal_importance_attention,
    temporal_importance_attention_with_residual_proxy,
    create_temporal_pooling_config_from_preset,
)

# ── Hybrid Pipeline + Sliding Window + Zero-Copy Decode ──
from .hybrid_pipeline import (
    HybridKVCacheStore,
    HybridPipelineConfig,
    SlidingWindowConfig,
    SlidingWindowEvictor,
    ZeroCopyDecodeCache,
)

# ── SpectralKV: Frequency-Aware KV Compression ──
from .spectral import (
    SpectralKVCodec,
    SpectralKVCompressed,
    SpectralKVConfig,
    spectral_qdq,
)

# ── PredictiveKV: Linear-Prediction-Based KV Compression ──
from .predictive import (
    PredictiveKVCodec,
    PredictiveKVCompressed,
    PredictiveKVConfig,
    predictive_qdq,
)

# ── CrossLayerKV: Cross-Layer Differential KV Compression ──
from .cross_layer import (
    CrossLayerKVCodec,
    CrossLayerKVCompressed,
    CrossLayerKVConfig,
    cross_layer_qdq,
    compress_multilayer_kv,
    decompress_multilayer_kv,
    estimate_layer_correlation,
    compute_layer_delta_stats,
)

# ── AttentionGatedKV: Attention-Gated Variable-Precision KV Compression ──
from .attention_gated import (
    AttentionGatedKVCodec,
    AttentionGatedKVCompressed,
    AttentionGatedKVConfig,
    attention_gated_qdq,
    compute_attention_importance,
    attention_gated_sdpa,
)

# ── DictKV: Dictionary-Based Sparse Coding KV Compression ──
from .dict_kv import (
    DictKVCodec,
    DictKVCompressed,
    DictKVConfig,
    dict_kv_qdq,
    orthogonal_matching_pursuit,
    learn_dictionary_online,
    learn_dictionary_ksvd,
)

# ── ExoBrain: External Brain System for Ultra Shell Inference ──
from .exobrain import (
    ExoBrainBackend,
    ExoBrainBus,
    ExoBrainConfig,
    ExoBrainAttentionPatcher,
    ShellProjection,
    VectorDBSource,
    APIKnowledgeSource,
    LocalWeightSource,
    KnowledgeSource,
    cross_attention_fusion,
    compute_gate,
    AdaptiveLayerSelector,
    compute_attention_entropy,
    MultiTeacherRouter,
)

# ── ExoBrain Inference & Distillation ──
from .exobrain_inference import (
    ExoBrainInferencePipeline,
    KnowledgeDistiller,
    TeacherKVExtractor,
    TeacherKVCache,
    InferenceResult,
    DistillResult,
    quick_exobrain_infer,
    HeadDimProjection,
    KVPrefetcher,
    ExoBrainEvaluator,
    AdaptiveInjectionScheduler,
    compute_perplexity_from_logits,
    BrainKVCompressor,
    ProgressiveDistiller,
    ExoBrainProfiler,
)

__all__ = [
    "AdaptiveKVCodec",
    "ComputeSkipConfig",
    "ComputeSkipResult",
    "adaptive_kv_bits",
    "compute_skip_attention",
    "kv_bytes_per_value",
    "KVCacheStore",
    "KVCacheStoreConfig",
    "KVMeta",
    "KVStoreBackend",
    "ApproxMode",
    "KVLayerStrategy",
    "KVLayerType",
    "KVPolicyPreset",
    "SafeExactPolicy",
    "Turbo3ExactKApproxVPolicy",
    "apply_policy_to_store_cfg",
    "build_policy",
    "classify_kv_layer",
    "list_policy_presets",
    "resolve_layer_strategy",
    # TurboQuantum exports
    "TurboQuantumConfig",
    "TurboQuantumResult",
    "TurboQuantumCodec",
    "turboquantum_compress",
    "create_turboquantum_codec",
    "turboquantum_entropy",
    "get_turboquantum_presets",
    # Layer-Adaptive
    "LayerAdaptiveBitAllocator",
    "LayerAdaptiveConfig",
    "apply_layer_adaptive_to_config",
    # Temporal Importance Pooling
    "TemporalPoolingConfig",
    "temporal_importance_attention",
    "temporal_importance_attention_with_residual_proxy",
    "create_temporal_pooling_config_from_preset",
    # Hybrid Pipeline
    "HybridKVCacheStore",
    "HybridPipelineConfig",
    "SlidingWindowConfig",
    "SlidingWindowEvictor",
    "ZeroCopyDecodeCache",
    # SpectralKV
    "SpectralKVCodec",
    "SpectralKVCompressed",
    "SpectralKVConfig",
    "spectral_qdq",
    # PredictiveKV
    "PredictiveKVCodec",
    "PredictiveKVCompressed",
    "PredictiveKVConfig",
    "predictive_qdq",
    # CrossLayerKV
    "CrossLayerKVCodec",
    "CrossLayerKVCompressed",
    "CrossLayerKVConfig",
    "cross_layer_qdq",
    "compress_multilayer_kv",
    "decompress_multilayer_kv",
    "estimate_layer_correlation",
    "compute_layer_delta_stats",
    # AttentionGatedKV
    "AttentionGatedKVCodec",
    "AttentionGatedKVCompressed",
    "AttentionGatedKVConfig",
    "attention_gated_qdq",
    "compute_attention_importance",
    "attention_gated_sdpa",
    # DictKV
    "DictKVCodec",
    "DictKVCompressed",
    "DictKVConfig",
    "dict_kv_qdq",
    "orthogonal_matching_pursuit",
    "learn_dictionary_online",
    "learn_dictionary_ksvd",
    # ExoBrain
    "ExoBrainBackend",
    "ExoBrainBus",
    "ExoBrainConfig",
    "ExoBrainAttentionPatcher",
    "ShellProjection",
    "VectorDBSource",
    "APIKnowledgeSource",
    "LocalWeightSource",
    "KnowledgeSource",
    "cross_attention_fusion",
    "compute_gate",
    "AdaptiveLayerSelector",
    "compute_attention_entropy",
    "MultiTeacherRouter",
    # ExoBrain Inference & Distillation
    "ExoBrainInferencePipeline",
    "KnowledgeDistiller",
    "TeacherKVExtractor",
    "TeacherKVCache",
    "InferenceResult",
    "DistillResult",
    "quick_exobrain_infer",
    "HeadDimProjection",
    "KVPrefetcher",
    "ExoBrainEvaluator",
    "AdaptiveInjectionScheduler",
    "compute_perplexity_from_logits",
    "BrainKVCompressor",
    "ProgressiveDistiller",
    "ExoBrainProfiler",
]
