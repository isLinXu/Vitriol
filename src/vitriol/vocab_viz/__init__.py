"""Vocabulary Visualization — tokenizer size treemaps and comparison charts."""

__all__ = [
    "VocabVisualizer",
]


def __getattr__(name: str):
    """Lazy import to defer plotly dependency until actually needed."""
    if name == "VocabVisualizer":
        try:
            from .core import VocabVisualizer
            return VocabVisualizer
        except ImportError as exc:
            from ..utils.exceptions import MissingOptionalDependencyError
            raise MissingOptionalDependencyError(
                getattr(exc, "name", None) or "plotly",
                feature="vocabulary visualization",
                extra="viz",
            ) from exc
    # Allow submodule access (needed for unittest.mock.patch paths)
    import importlib
    try:
        return importlib.import_module(f".{name}", __name__)
    except ImportError:
        raise AttributeError(f"module 'vitriol.vocab_viz' has no attribute '{name}'")
