"""
Architectural Analyzers Sub-package.

Legacy single-file had 32 analyzer classes tightly coupled.
New sub-package structure: one class per file (or related group).
"""

from .gqa import GqaAnalyzer
from .mla import MlaAnalyzer
from .moe import MoeAnalyzer
from .mamba import MambaAnalyzer
from .swa import SwaAnalyzer

__all__ = [
    "GqaAnalyzer",
    "MlaAnalyzer", 
    "MoeAnalyzer",
    "MambaAnalyzer",
    "SwaAnalyzer",
]
