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
    """Lazy import to avoid importing heavy dependencies unless needed.

    This keeps `import vitriol` lightweight, deferring torch/transformers
    imports until the user actually accesses those symbols.
    """
    if name == "MinimalWeightGenerator":
        from .core.generator import MinimalWeightGenerator as _MinimalWeightGenerator
        return _MinimalWeightGenerator
    elif name == "ModelValidator":
        from .core.validator import ModelValidator as _ModelValidator
        return _ModelValidator
    elif name == "ModelAnalyzer":
        from .core.analyzer import ModelAnalyzer as _ModelAnalyzer
        return _ModelAnalyzer
    elif name == "GenerationConfig":
        from .config.manager import GenerationConfig as _GenerationConfig
        return _GenerationConfig
    else:
        raise AttributeError(f"module 'vitriol' has no attribute '{name}'")


def __dir__():
    """List all exported names."""
    return sorted(list(globals().keys()) + list(__all__))
