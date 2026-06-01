"""Helpers for optional dependencies with clear, actionable errors.

Vitriol degrades gracefully when optional dependencies (triton, fastapi,
gradio, ...) are missing. The historical pattern of binding a missing symbol to
``None`` works, but produces a confusing ``'NoneType' object is not callable``
the moment a user actually invokes the feature. The helpers here surface a
:class:`~vitriol.utils.exceptions.MissingOptionalDependencyError` with an
install hint instead.

Typical usage::

    from .utils.optional import require

    def run_fancy():
        fancy = require("fancylib", feature="the fancy pipeline", extra="fancy")
        return fancy.do_thing()

or, to keep a module-level symbol that fails loudly only when touched::

    try:
        import fancylib
    except ImportError:
        fancylib = MissingDependencyStub("fancylib", feature="...", extra="fancy")
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Optional

from .exceptions import MissingOptionalDependencyError

__all__ = ["require", "has", "MissingDependencyStub"]


def require(
    package: str,
    *,
    feature: Optional[str] = None,
    extra: Optional[str] = None,
    import_name: Optional[str] = None,
) -> ModuleType:
    """Import an optional dependency or raise a clear, actionable error.

    Args:
        package: Distribution / pip name shown in the install hint.
        feature: Human-readable feature name for the error message.
        extra: ``pip install 'vitriol[<extra>]'`` extra that bundles *package*.
        import_name: Module to import, if it differs from *package*.

    Returns:
        The imported module.

    Raises:
        MissingOptionalDependencyError: If the dependency cannot be imported.
    """
    name = import_name or package
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise MissingOptionalDependencyError(package, feature=feature, extra=extra) from exc


def has(import_name: str) -> bool:
    """Return True if *import_name* can be imported, without raising."""
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


class MissingDependencyStub:
    """Placeholder for an optional symbol that is not available.

    Any attribute access or call raises
    :class:`~vitriol.utils.exceptions.MissingOptionalDependencyError`, replacing
    the opaque ``'NoneType' object is not callable`` with an actionable message.
    Evaluates falsy so existing ``if symbol:`` checks keep behaving like the old
    ``symbol = None`` pattern.
    """

    __slots__ = ("_package", "_feature", "_extra")

    def __init__(self, package: str, *, feature: Optional[str] = None, extra: Optional[str] = None):
        self._package = package
        self._feature = feature
        self._extra = extra

    def _raise(self):
        raise MissingOptionalDependencyError(self._package, feature=self._feature, extra=self._extra)

    def __call__(self, *args, **kwargs):
        self._raise()

    def __getattr__(self, item):
        # Dunder/internal lookups should not trigger the loud error.
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        self._raise()

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"<MissingDependencyStub package={self._package!r}>"
