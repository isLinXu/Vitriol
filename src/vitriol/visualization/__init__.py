# ⚠️ DEPRECATED: Use vitriol.arch_viz instead.
# This module will be removed in a future version.

__all__ = ["WeightVisualizer"]


def __getattr__(name: str):
    """Lazy import to defer plotly dependency until actually needed."""
    if name == "WeightVisualizer":
        from vitriol.visualization.visualizer import WeightVisualizer
        return WeightVisualizer
    # Allow submodule access (needed for unittest.mock.patch paths)
    import importlib
    try:
        return importlib.import_module(f".{name}", __name__)
    except ImportError:
        raise AttributeError(f"module 'vitriol.visualization' has no attribute '{name}'")
