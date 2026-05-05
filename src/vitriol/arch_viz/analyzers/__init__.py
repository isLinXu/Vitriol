"""
Architectural Analyzers Sub-package.

Legacy single-file had 32 analyzer classes tightly coupled.
New sub-package structure: one class per file (or related group).
"""

# New sub-package analyzers
from .gqa import GQAAnalyzer
from .mla import MLAAnalyzer
from .moe import MoEAnalyzer
from .mamba import MambaAnalyzer
from .swa import SWAAnalyzer

# Re-export legacy analyzers and registry for backward compatibility
from .._analyzers_legacy import (  # noqa: F401
    AnalyzerRegistry,
    ModelAnalyzer,
    TransformerAnalyzer,
    QwenAnalyzer,
    SequenceMixerAnalyzer,
    Qwen2MoeAnalyzer,
    LlamaAnalyzer,
    MistralAnalyzer,
    GemmaAnalyzer,
    PhiAnalyzer,
    CohereAnalyzer,
    StableLMAnalyzer,
    YiAnalyzer,
    InternLMAnalyzer,
    BaichuanAnalyzer,
    BloomAnalyzer,
    GPTNeoXAnalyzer,
    StarCoderAnalyzer,
    StarCoder2Analyzer,
    DeepSeekAnalyzer,
    KimiAnalyzer,
    GLMAnalyzer,
    ErnieAnalyzer,
    GPT2Analyzer,
    BertAnalyzer,
    T5Analyzer,
    FalconAnalyzer,
    OPTAnalyzer,
    MiniMaxAnalyzer,
    InternS1Analyzer,
    Qwen35Analyzer,
    Hy3Analyzer,
)

__all__ = [
    # New sub-package analyzers
    "GQAAnalyzer",
    "MLAAnalyzer",
    "MoEAnalyzer",
    "MambaAnalyzer",
    "SWAAnalyzer",
    # Legacy re-exports
    "AnalyzerRegistry",
    "ModelAnalyzer",
    "TransformerAnalyzer",
    "QwenAnalyzer",
    "SequenceMixerAnalyzer",
    "Qwen2MoeAnalyzer",
    "LlamaAnalyzer",
    "MistralAnalyzer",
    "GemmaAnalyzer",
    "PhiAnalyzer",
    "CohereAnalyzer",
    "StableLMAnalyzer",
    "YiAnalyzer",
    "InternLMAnalyzer",
    "BaichuanAnalyzer",
    "BloomAnalyzer",
    "GPTNeoXAnalyzer",
    "StarCoderAnalyzer",
    "StarCoder2Analyzer",
    "DeepSeekAnalyzer",
    "KimiAnalyzer",
    "GLMAnalyzer",
    "ErnieAnalyzer",
    "GPT2Analyzer",
    "BertAnalyzer",
    "T5Analyzer",
    "FalconAnalyzer",
    "OPTAnalyzer",
    "MiniMaxAnalyzer",
    "InternS1Analyzer",
    "Qwen35Analyzer",
    "Hy3Analyzer",
]
