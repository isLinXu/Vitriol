"""Runtime markers for experimental Vitriol features.

Many Vitriol features (the REST API, the RL-based architecture searcher,
TurboQuantum, ...) are documented as "experimental" in prose only. This module
turns that status into a programmatic signal: decorating a function or class
with :func:`experimental` emits an :class:`ExperimentalWarning` the first time
it is used, so downstream users get a clear heads-up that the API may change.

Set the ``VITRIOL_SILENCE_EXPERIMENTAL`` environment variable (to ``1``/``true``/
``yes``/``on``) to suppress these warnings entirely.
"""

from __future__ import annotations

import functools
import os
import warnings
from typing import Callable, Optional, TypeVar, Union

__all__ = ["ExperimentalWarning", "experimental", "is_experimental"]

_T = TypeVar("_T")
_ENV_SILENCE = "VITRIOL_SILENCE_EXPERIMENTAL"
_FLAG = "__vitriol_experimental__"


class ExperimentalWarning(UserWarning):
    """Emitted the first time an experimental Vitriol feature is used."""


def _silenced() -> bool:
    return os.environ.get(_ENV_SILENCE, "").strip().lower() in {"1", "true", "yes", "on"}


def _format(name: str, since: Optional[str], detail: Optional[str]) -> str:
    msg = f"'{name}' is experimental and may change or be removed without notice."
    if since:
        msg += f" (since {since})"
    if detail:
        msg += f" {detail}"
    msg += f" Set {_ENV_SILENCE}=1 to silence this warning."
    return msg


def experimental(
    feature: Union[str, Callable, type, None] = None,
    *,
    since: Optional[str] = None,
    detail: Optional[str] = None,
) -> Callable:
    """Mark a function or class as experimental.

    Usable either bare (``@experimental``) or parametrised
    (``@experimental("REST API", since="0.5", detail="...")``). The decorated
    object emits :class:`ExperimentalWarning` once — on first call for a
    function, on first instantiation for a class — and is tagged with the
    ``__vitriol_experimental__`` attribute (see :func:`is_experimental`).
    """

    def decorate(obj):
        name = feature if isinstance(feature, str) else getattr(
            obj, "__qualname__", getattr(obj, "__name__", repr(obj))
        )
        state = {"warned": False}

        def _warn() -> None:
            if state["warned"] or _silenced():
                return
            state["warned"] = True
            warnings.warn(_format(name, since, detail), ExperimentalWarning, stacklevel=3)

        if isinstance(obj, type):
            original_init = obj.__init__

            @functools.wraps(original_init)
            def __init__(self, *args, **kwargs):  # noqa: N807
                _warn()
                original_init(self, *args, **kwargs)

            obj.__init__ = __init__
            setattr(obj, _FLAG, True)
            return obj

        @functools.wraps(obj)
        def wrapper(*args, **kwargs):
            _warn()
            return obj(*args, **kwargs)

        setattr(wrapper, _FLAG, True)
        return wrapper

    # Bare usage: @experimental (feature is the decorated object itself).
    if callable(feature) and not isinstance(feature, str):
        return decorate(feature)
    return decorate


def is_experimental(obj) -> bool:
    """Return True if *obj* was marked with :func:`experimental`."""
    return bool(getattr(obj, _FLAG, False))
