"""
Architectural Analyzers Sub-package.

Legacy single-file had 32 analyzer classes tightly coupled.
New sub-package structure: one class per file (or related group).
"""

# New sub-package analyzers
# Re-export legacy analyzers and registry for backward compatibility
from .._analyzers_legacy import (  # noqa: F401
    AnalyzerRegistry,
    BaichuanAnalyzer,
    BertAnalyzer,
    BloomAnalyzer,
    CohereAnalyzer,
    DeepSeekAnalyzer,
    ErnieAnalyzer,
    FalconAnalyzer,
    GemmaAnalyzer,
    GLMAnalyzer,
    GPT2Analyzer,
    GPTNeoXAnalyzer,
    Hy3Analyzer,
    InternLMAnalyzer,
    InternS1Analyzer,
    KimiAnalyzer,
    LlamaAnalyzer,
    MiniMaxAnalyzer,
    MistralAnalyzer,
    ModelAnalyzer,
    OPTAnalyzer,
    PhiAnalyzer,
    Qwen2MoeAnalyzer,
    Qwen35Analyzer,
    QwenAnalyzer,
    SequenceMixerAnalyzer,
    StableLMAnalyzer,
    StarCoder2Analyzer,
    StarCoderAnalyzer,
    T5Analyzer,
    TransformerAnalyzer,
    YiAnalyzer,
)
from .gqa import GQAAnalyzer
from .mamba import MambaAnalyzer
from .mla import MLAAnalyzer
from .moe import MoEAnalyzer
from .swa import SWAAnalyzer

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
