"""
Vitriol Tools Module
===================

Provides standalone utility tools for model comparison, architecture analysis,
and batch processing.

Example::

    from vitriol.tools import ModelComparator, format_params
"""

from .comparator import ModelComparator, format_params

__all__ = [
    "ModelComparator",
    "format_params",
]
