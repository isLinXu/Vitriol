"""
Vitriol-NAS: Neural Architecture Search for LLMs

This subpackage contains heavyweight modules such as evaluator/searcher (torch/transformers).
To reduce import-time overhead (and allow reusing pure algorithm modules like search_space /
targeted_nas in lightweight environments), we use lazy imports and only import a module when
its symbol is actually accessed.
"""

from __future__ import annotations

from .search_space import ArchitectureGene, LLMSearchSpace

_LAZY_EXPORTS = {
    # searcher
    "RandomSearcher": (".searcher", "RandomSearcher"),
    "EvolutionarySearcher": (".searcher", "EvolutionarySearcher"),
    # evaluator
    "HybridEvaluator": (".evaluator", "HybridEvaluator"),
    "ZeroCostProxy": (".evaluator", "ZeroCostProxy"),
    # controller
    "NASController": (".controller", "NASController"),
}


def __getattr__(name: str):  # pragma: no cover
    if name in _LAZY_EXPORTS:
        module_name, attr = _LAZY_EXPORTS[name]
        from importlib import import_module

        mod = import_module(module_name, package=__name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "LLMSearchSpace",
    "ArchitectureGene",
    *_LAZY_EXPORTS.keys(),
]
