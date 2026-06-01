"""Tests for the @experimental runtime marker."""

import warnings

import pytest

from vitriol.utils.experimental import ExperimentalWarning, experimental, is_experimental


def test_function_warns_once_and_returns_value():
    @experimental
    def add(a, b):
        return a + b

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert add(2, 3) == 5
        assert add(4, 5) == 9  # second call must not warn again

    exp = [w for w in caught if issubclass(w.category, ExperimentalWarning)]
    assert len(exp) == 1
    assert "experimental" in str(exp[0].message)
    assert is_experimental(add)


def test_function_preserves_metadata():
    @experimental("nice feature")
    def documented():
        """My docstring."""
        return 1

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "My docstring."


def test_parametrised_message_contains_feature_since_detail():
    @experimental("FancyThing", since="0.9", detail="May change.")
    def go():
        return None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        go()

    msg = str(caught[0].message)
    assert "FancyThing" in msg
    assert "0.9" in msg
    assert "May change." in msg


def test_class_warns_on_instantiation():
    @experimental("CoolClass")
    class Cool:
        def __init__(self, x):
            self.x = x

    assert is_experimental(Cool)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        inst = Cool(7)

    assert inst.x == 7
    exp = [w for w in caught if issubclass(w.category, ExperimentalWarning)]
    assert len(exp) == 1
    assert "CoolClass" in str(exp[0].message)


def test_env_var_silences_warning(monkeypatch):
    monkeypatch.setenv("VITRIOL_SILENCE_EXPERIMENTAL", "1")

    @experimental
    def quiet():
        return 42

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert quiet() == 42

    assert not [w for w in caught if issubclass(w.category, ExperimentalWarning)]


def test_is_experimental_false_for_plain_object():
    def plain():
        return 1

    assert is_experimental(plain) is False
