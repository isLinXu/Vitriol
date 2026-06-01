"""ExoBrain inference pipeline & knowledge distiller (compatibility facade).

The implementation now lives in the :mod:`vitriol.kv.exobrain` package
(``teacher``, ``scheduler``, ``pipeline``, ``distill``, ``profiler`` submodules).
This module re-exports the public surface so the historical import path
``vitriol.kv.exobrain_inference`` keeps working unchanged.
"""

from .exobrain.distill import KnowledgeDistiller, ProgressiveDistiller
from .exobrain.pipeline import ExoBrainInferencePipeline
from .exobrain.profiler import (
    ExoBrainEvaluator,
    ExoBrainProfiler,
    quick_exobrain_infer,
)
from .exobrain.scheduler import (
    AdaptiveInjectionScheduler,
    BrainKVCompressor,
    KVPrefetcher,
    compute_perplexity_from_logits,
)
from .exobrain.teacher import (
    DistillResult,
    HeadDimProjection,
    InferenceResult,
    TeacherKVCache,
    TeacherKVExtractor,
)

__all__ = [
    "HeadDimProjection",
    "InferenceResult",
    "DistillResult",
    "TeacherKVCache",
    "TeacherKVExtractor",
    "AdaptiveInjectionScheduler",
    "compute_perplexity_from_logits",
    "BrainKVCompressor",
    "KVPrefetcher",
    "ExoBrainInferencePipeline",
    "KnowledgeDistiller",
    "ProgressiveDistiller",
    "ExoBrainProfiler",
    "ExoBrainEvaluator",
    "quick_exobrain_infer",
]
