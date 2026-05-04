"""
vitriol.core

This package contains heavyweight modules such as generator/validator (torch/transformers).
To allow importing lightweight helper modules (e.g., the manifest builder) in minimal
environments, we use lazy imports here.
"""

from __future__ import annotations

_LAZY_EXPORTS = {
    "MinimalWeightGenerator": (".generator", "MinimalWeightGenerator"),
    "ModelValidator": (".validator", "ModelValidator"),
    "ModelAnalyzer": (".analyzer", "ModelAnalyzer"),
}


def __getattr__(name: str):  # pragma: no cover
    if name in _LAZY_EXPORTS:
        module_name, attr = _LAZY_EXPORTS[name]
        from importlib import import_module

        mod = import_module(module_name, package=__name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_EXPORTS.keys())
