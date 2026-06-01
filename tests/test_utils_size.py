from __future__ import annotations

import pytest

from vitriol.utils.size import parse_size_to_bytes


def test_parse_size_to_bytes_supports_units_spaces_and_decimals() -> None:
    assert parse_size_to_bytes("100B") == 100
    assert parse_size_to_bytes("10KB") == 10 * 1024
    assert parse_size_to_bytes(" 1.5 GB ") == int(1.5 * (1 << 30))
    assert parse_size_to_bytes("2TB") == 2 * (1 << 40)


def test_parse_size_to_bytes_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="Invalid size"):
        parse_size_to_bytes("")
    with pytest.raises(ValueError, match="Negative size"):
        parse_size_to_bytes("-5GB")
    with pytest.raises(ValueError, match="Cannot parse"):
        parse_size_to_bytes("abc")
