"""Tests for optional-dependency helpers and the missing-dependency error."""

import pytest

from vitriol.utils.exceptions import MissingOptionalDependencyError
from vitriol.utils.optional import MissingDependencyStub, has, require


def test_require_returns_existing_module():
    mod = require("json")
    assert mod.dumps({"a": 1}) == '{"a": 1}'


def test_require_supports_distinct_import_name():
    # Distribution name differs from import name; both point at a stdlib module here.
    mod = require("python-dateutil-ish", import_name="json")
    assert mod is require("json")


def test_require_missing_raises_actionable_error():
    with pytest.raises(MissingOptionalDependencyError) as ei:
        require("definitely_not_installed_pkg_xyz", feature="the widget", extra="widget")

    msg = str(ei.value)
    assert "definitely_not_installed_pkg_xyz" in msg
    assert "the widget" in msg
    assert "pip install definitely_not_installed_pkg_xyz" in msg
    assert "vitriol[widget]" in msg


def test_missing_dependency_error_is_import_error():
    # Existing `except ImportError` handlers must keep catching it.
    assert issubclass(MissingOptionalDependencyError, ImportError)
    err = MissingOptionalDependencyError("pkg", feature="f", extra="e")
    assert err.package == "pkg"
    assert err.feature == "f"
    assert err.extra == "e"
    assert isinstance(err, ImportError)


def test_has_true_and_false():
    assert has("json") is True
    assert has("definitely_not_installed_pkg_xyz") is False


def test_stub_is_falsy_and_raises_on_use():
    stub = MissingDependencyStub("pkg", feature="the feature", extra="x")
    assert bool(stub) is False
    assert "pkg" in repr(stub)

    with pytest.raises(MissingOptionalDependencyError):
        stub()

    with pytest.raises(MissingOptionalDependencyError):
        _ = stub.some_attribute


def test_stub_dunder_access_raises_attribute_error():
    stub = MissingDependencyStub("pkg")
    with pytest.raises(AttributeError):
        _ = stub.__wrapped__
