from .codec import (
    AdaptiveKVCodec,
    ComputeSkipConfig,
    ComputeSkipResult,
    adaptive_kv_bits,
    compute_skip_attention,
    kv_bytes_per_value,
)
from .codec import (
    _vectorized_blockwise_qdq as _vectorized_blockwise_qdq,
)
from .codec import (
    clear_projection_cache as clear_projection_cache,
)

try:
    from .triton_kernels import _HAS_TRITON, get_backend_name, triton_fwht
except ImportError:
    def get_backend_name():
        return "python"
    triton_fwht = None
    _HAS_TRITON = False
from .backend import KVMeta, KVStoreBackend
from .cache_store import KVCacheStore, KVCacheStoreConfig
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
from .utils import clear_vitriol_kv

try:
    from .turboquantum import (
        TurboQuantumCodec,
        TurboQuantumConfig,
        TurboQuantumResult,
        create_turboquantum_codec,
        get_turboquantum_presets,
        turboquantum_compress,
    )
    from .turboquantum import (
        compute_attention_entropy as turboquantum_entropy,
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
# ── AttentionGatedKV: Attention-Gated Variable-Precision KV Compression ──
from .attention_gated import (
    AttentionGatedKVCodec,
    AttentionGatedKVCompressed,
    AttentionGatedKVConfig,
    attention_gated_qdq,
    attention_gated_sdpa,
    compute_attention_importance,
)

# ── CrossLayerKV: Cross-Layer Differential KV Compression ──
from .cross_layer import (
    CrossLayerKVCodec,
    CrossLayerKVCompressed,
    CrossLayerKVConfig,
    compress_multilayer_kv,
    compute_layer_delta_stats,
    cross_layer_qdq,
    decompress_multilayer_kv,
    estimate_layer_correlation,
)

# ── DictKV: Dictionary-Based Sparse Coding KV Compression ──
from .dict_kv import (
    DictKVCodec,
    DictKVCompressed,
    DictKVConfig,
    dict_kv_qdq,
    learn_dictionary_ksvd,
    learn_dictionary_online,
    orthogonal_matching_pursuit,
)

# ── ExoBrain: External Brain System for Ultra Shell Inference ──
from .exobrain import (
    AdaptiveLayerSelector,
    APIKnowledgeSource,
    ExoBrainAttentionPatcher,
    ExoBrainBackend,
    ExoBrainBus,
    ExoBrainConfig,
    KnowledgeSource,
    LocalWeightSource,
    MultiTeacherRouter,
    ShellProjection,
    VectorDBSource,
    compute_attention_entropy,
    compute_gate,
    cross_attention_fusion,
)

# ── ExoBrain Inference & Distillation ──
from .exobrain_inference import (
    AdaptiveInjectionScheduler,
    BrainKVCompressor,
    DistillResult,
    ExoBrainEvaluator,
    ExoBrainInferencePipeline,
    ExoBrainProfiler,
    HeadDimProjection,
    InferenceResult,
    KnowledgeDistiller,
    KVPrefetcher,
    ProgressiveDistiller,
    TeacherKVCache,
    TeacherKVExtractor,
    compute_perplexity_from_logits,
    quick_exobrain_infer,
)

# ── Hybrid Pipeline + Sliding Window + Zero-Copy Decode ──
from .hybrid_pipeline import (
    HybridKVCacheStore,
    HybridPipelineConfig,
    SlidingWindowConfig,
    SlidingWindowEvictor,
    ZeroCopyDecodeCache,
)
from .layer_adaptive import (
    LayerAdaptiveBitAllocator,
    LayerAdaptiveConfig,
    apply_layer_adaptive_to_config,
)

# ── PredictiveKV: Linear-Prediction-Based KV Compression ──
from .predictive import (
    PredictiveKVCodec,
    PredictiveKVCompressed,
    PredictiveKVConfig,
    predictive_qdq,
)

# ── SpectralKV: Frequency-Aware KV Compression ──
from .spectral import (
    SpectralKVCodec,
    SpectralKVCompressed,
    SpectralKVConfig,
    spectral_qdq,
)

# ── Temporal Importance Pooling ──
from .temporal_pooling import (
    TemporalPoolingConfig,
    create_temporal_pooling_config_from_preset,
    temporal_importance_attention,
    temporal_importance_attention_with_residual_proxy,
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
    "clear_vitriol_kv",
]
