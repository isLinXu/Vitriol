"""Helpers for parsing human-readable byte sizes."""

from __future__ import annotations

from typing import Final

_SIZE_UNITS: Final[tuple[tuple[str, int], ...]] = (
    ("TB", 1 << 40),
    ("GB", 1 << 30),
    ("MB", 1 << 20),
    ("KB", 1 << 10),
    ("B", 1),
)


def parse_size_to_bytes(size: str) -> int:
    """Parse strings such as ``5GB`` or ``512 MB`` into bytes."""
    if not size or not isinstance(size, str):
        raise ValueError(f"Invalid size string: {size!r}")

    normalized = size.upper().replace(" ", "")
    for unit, multiplier in _SIZE_UNITS:
        if normalized.endswith(unit):
            raw_number = normalized[: -len(unit)]
            try:
                value = float(raw_number)
            except ValueError as exc:
                raise ValueError(
                    f"Cannot parse numeric value from size string: {size!r}"
                ) from exc
            if value < 0:
                raise ValueError(f"Negative size not allowed: {size!r}")
            return int(value * multiplier)

    try:
        value = int(normalized)
    except ValueError as exc:
        raise ValueError(f"Cannot parse size string: {size!r}") from exc
    if value < 0:
        raise ValueError(f"Negative size not allowed: {size!r}")
    return value
