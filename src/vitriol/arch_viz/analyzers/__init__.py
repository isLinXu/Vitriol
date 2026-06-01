"""
Architectural Analyzers Sub-package.

Legacy single-file had 32 analyzer classes tightly coupled.
New sub-package structure: one class per file (or related group).
"""

# Model-family analyzers + registry, organised by family module.
from .base import ModelAnalyzer, TransformerAnalyzer  # noqa: F401
from .deepseek import DeepSeekAnalyzer, KimiAnalyzer  # noqa: F401
from .dense import (  # noqa: F401
    BaichuanAnalyzer,
    CohereAnalyzer,
    GemmaAnalyzer,
    InternLMAnalyzer,
    LlamaAnalyzer,
    MistralAnalyzer,
    PhiAnalyzer,
    Qwen2MoeAnalyzer,
    QwenAnalyzer,
    StableLMAnalyzer,
    YiAnalyzer,
)
from .encoder import BertAnalyzer, T5Analyzer  # noqa: F401
from .ernie import ErnieAnalyzer  # noqa: F401
from .glm import GLMAnalyzer  # noqa: F401
from .gpt import (  # noqa: F401
    BloomAnalyzer,
    FalconAnalyzer,
    GPT2Analyzer,
    GPTNeoXAnalyzer,
    OPTAnalyzer,
    StarCoder2Analyzer,
    StarCoderAnalyzer,
)
from .gqa import GQAAnalyzer
from .hy3 import Hy3Analyzer  # noqa: F401
from .intern_s1 import InternS1Analyzer  # noqa: F401
from .mamba import MambaAnalyzer
from .minimax import MiniMaxAnalyzer  # noqa: F401
from .mla import MLAAnalyzer
from .moe import MoEAnalyzer
from .qwen35 import Qwen35Analyzer  # noqa: F401
from .registry import AnalyzerRegistry  # noqa: F401
from .sequence_mixer import SequenceMixerAnalyzer  # noqa: F401
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
