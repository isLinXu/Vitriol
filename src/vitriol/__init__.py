"""Vitriol — Unified framework for model structure visualization, compression, pruning, quantization and efficient inference.

V.I.T.R.I.O.L.
Visita Interiora Terrae Rectificando Invenies Occultum Lapidem.
Explore the inner structure, purify redundant weights, and discover the hidden efficient core.
Venture into the model's depths, distill the essence of everything, and uncover the hidden core.
"""

from .version import __version__

__all__ = [
    "__version__",
    "MinimalWeightGenerator",
    "ModelValidator",
    "ModelAnalyzer",
    "GenerationConfig",
]


def __getattr__(name: str):
    """Lazy attribute loading to keep `import vitriol` lightweight.

    This avoids importing heavy optional dependencies (e.g., torch/transformers)
    unless the user actually accesses those symbols.
    """
    if name == "MinimalWeightGenerator":
        from .core.generator import MinimalWeightGenerator as _MinimalWeightGenerator

        return _MinimalWeightGenerator
    if name == "ModelValidator":
        from .core.validator import ModelValidator as _ModelValidator

        return _ModelValidator
    if name == "ModelAnalyzer":
        from .core.analyzer import ModelAnalyzer as _ModelAnalyzer

        return _ModelAnalyzer
    if name == "GenerationConfig":
        from .config.manager import GenerationConfig as _GenerationConfig

        return _GenerationConfig
    raise AttributeError(f"module 'vitriol' has no attribute '{name}'")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
