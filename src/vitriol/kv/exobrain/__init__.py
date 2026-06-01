"""ExoBrain: External Brain System for Heterogeneous Reasoning.

A lightweight "shell model" (real ~0.1B weights) reasons using KV pairs from an
external "brain" (7B+ model), aligned through a thin ``ShellProjection`` layer.
See the family modules for details:

- ``projection`` — ShellProjection cognitive-alignment layer
- ``sources``    — knowledge sources (Vector DB / API / local weights)
- ``config``     — configuration + adaptive layer selection
- ``fusion``     — knowledge bus, multi-teacher routing, cross-attention fusion
- ``backend``    — KV store backend + attention patcher

Inference / distillation pipelines live in :mod:`vitriol.kv.exobrain_inference`.
"""

from .backend import ExoBrainAttentionPatcher, ExoBrainBackend
from .config import AdaptiveLayerSelector, ExoBrainConfig, compute_attention_entropy
from .fusion import (
    ExoBrainBus,
    MultiTeacherRouter,
    compute_gate,
    cross_attention_fusion,
)
from .projection import ShellProjection
from .sources import (
    APIKnowledgeSource,
    KnowledgeSource,
    LocalWeightSource,
    VectorDBSource,
)

__all__ = [
    "ShellProjection",
    "KnowledgeSource",
    "VectorDBSource",
    "APIKnowledgeSource",
    "LocalWeightSource",
    "ExoBrainConfig",
    "AdaptiveLayerSelector",
    "compute_attention_entropy",
    "MultiTeacherRouter",
    "cross_attention_fusion",
    "compute_gate",
    "ExoBrainBus",
    "ExoBrainBackend",
    "ExoBrainAttentionPatcher",
]
