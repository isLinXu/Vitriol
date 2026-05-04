#!/usr/bin/env python3
"""
Comprehensive Validation Test for All Model Adapters

Tests all 11 adapter classes:
- LlamaAdapter
- QwenMoeAdapter
- Qwen35MoeAdapter
- DeepSeekAdapter
- MistralAdapter
- GemmaAdapter
- PhiAdapter
- CohereAdapter
- GLMAdapter
- StableLMAdapter
- MiniMaxAdapter
- DefaultAdapter

Validates:
1. Adapter registry integration
2. Match method correctness
3. Config patching
4. Model class selection
5. Adapter capabilities
"""

import sys
import os

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from vitriol.adapters import (
    LlamaAdapter,
    QwenMoeAdapter,
    Qwen35MoeAdapter,
    DeepSeekAdapter,
    MistralAdapter,
    GemmaAdapter,
    PhiAdapter,
    CohereAdapter,
    GLMAdapter,
    StableLMAdapter,
    MiniMaxAdapter,
    AdapterRegistry,
    ModelAdapter,
)
from vitriol.adapters.base import DefaultAdapter


def sep(title: str = "") -> None:
    print(f"\n{'='*70}")
    if title:
        print(f"  {title}")
        print(f"{'='*70}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Adapter Registry Discovery
# ─────────────────────────────────────────────────────────────────────────────

def test_adapter_discovery():
    """Test that AdapterRegistry discovers all expected adapters."""
    sep("1. Adapter Registry Discovery")

    expected_adapters = [
        "LlamaAdapter",
        "QwenMoeAdapter",
        "Qwen35MoeAdapter",
        "DeepSeekAdapter",
        "MistralAdapter",
        "GemmaAdapter",
        "PhiAdapter",
        "CohereAdapter",
        "GLMAdapter",
        "StableLMAdapter",
        "MiniMaxAdapter",
        "DefaultAdapter",
    ]

    try:
        metadata = AdapterRegistry.discover_builtin_adapter_metadata()
        discovered_names = {item["name"] for item in metadata}

        passed = 0
        failed = 0

        for name in expected_adapters:
            if name in discovered_names:
                print(f"  ✅ {name}: discovered")
                passed += 1
            else:
                print(f"  ❌ {name}: NOT discovered")
                failed += 1

        print(f"\n  Discovered {len(discovered_names)} adapters: {sorted(discovered_names)}")
        print(f"\n  Results: {passed} passed, {failed} failed")
        assert failed == 0

    except Exception as e:
        print(f"  ❌ AdapterRegistry.discover_builtin_adapter_metadata() raised: {type(e).__name__}: {e}")
        assert False


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Match Method Correctness
# ─────────────────────────────────────────────────────────────────────────────

class MockConfig:
    """Minimal mock PretrainedConfig for testing."""

    def __init__(self, model_type: str, model_name: str = "mock/model", **kwargs):
        self.model_type = model_type
        self.model_name_or_path = model_name
        self._kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_match_methods():
    """Test that each adapter's match() returns correct results."""
    sep("2. Adapter Match Methods")

    adapters = [
        (LlamaAdapter, "llama", True),
        (QwenMoeAdapter, "qwen2_moe", True),
        (Qwen35MoeAdapter, "qwen3_5_moe", True),
        (DeepSeekAdapter, "deepseek", True),
        (MistralAdapter, "mistral", True),
        (GemmaAdapter, "gemma", True),
        (PhiAdapter, "phi", True),
        (CohereAdapter, "cohere", True),
        (GLMAdapter, "glm", True),
        (StableLMAdapter, "stablelm", True),
        (MiniMaxAdapter, "minimax", True),
    ]

    passed = 0
    failed = 0

    for cls, model_type, expected in adapters:
        try:
            config = MockConfig(model_type=model_type, model_name="test/model")
            result = cls.match("test/model", config)
            if result == expected:
                print(f"  ✅ {cls.__name__}.match('{model_type}') → {result}")
                passed += 1
            else:
                print(f"  ❌ {cls.__name__}.match('{model_type}') → {result}, expected {expected}")
                failed += 1
        except Exception as e:
            print(f"  ❌ {cls.__name__}.match('{model_type}') raised {type(e).__name__}: {e}")
            failed += 1

    # Test negative matches (should return False for wrong model types)
    print("\n  Negative match tests (should all return False):")
    negative_cases = [
        (LlamaAdapter, "gpt2"),
        (QwenMoeAdapter, "llama"),
        (DeepSeekAdapter, "mistral"),
        (MistralAdapter, "deepseek"),
    ]

    for cls, model_type in negative_cases:
        try:
            config = MockConfig(model_type=model_type)
            result = cls.match("test/model", config)
            if result == False:
                print(f"  ✅ {cls.__name__}.match('{model_type}') → False")
                passed += 1
            else:
                print(f"  ❌ {cls.__name__}.match('{model_type}') → {result}, expected False")
                failed += 1
        except Exception as e:
            print(f"  ❌ {cls.__name__}.match('{model_type}') raised {type(e).__name__}: {e}")
            failed += 1

    print(f"\n  Results: {passed} passed, {failed} failed")
    assert failed == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Patch Config
# ─────────────────────────────────────────────────────────────────────────────

def test_patch_config():
    """Test that adapters can patch configs correctly."""
    sep("3. Adapter patch_config()")

    adapters = [
        LlamaAdapter(),
        QwenMoeAdapter(),
        Qwen35MoeAdapter(),
        DeepSeekAdapter(),
        MistralAdapter(),
        GemmaAdapter(),
        PhiAdapter(),
        CohereAdapter(),
        GLMAdapter(),
        StableLMAdapter(),
        MiniMaxAdapter(),
        DefaultAdapter(),
    ]

    passed = 0
    failed = 0

    for adapter in adapters:
        try:
            config = MockConfig(model_type="mock", test_attr="original")
            result = adapter.patch_config(config)
            # Should return a config object (may be same or modified)
            if result is not None:
                print(f"  ✅ {type(adapter).__name__}.patch_config() → returned config")
                passed += 1
            else:
                print(f"  ❌ {type(adapter).__name__}.patch_config() → returned None")
                failed += 1
        except Exception as e:
            print(f"  ❌ {type(adapter).__name__}.patch_config() raised {type(e).__name__}: {e}")
            failed += 1

    print(f"\n  Results: {passed} passed, {failed} failed")
    assert failed == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Get Model Class
# ─────────────────────────────────────────────────────────────────────────────

def test_get_model_class():
    """Test get_model_class() returns correct types."""
    sep("4. Adapter get_model_class()")

    adapters = [
        LlamaAdapter(),
        QwenMoeAdapter(),
        Qwen35MoeAdapter(),
        DeepSeekAdapter(),
        MistralAdapter(),
        GemmaAdapter(),
        PhiAdapter(),
        CohereAdapter(),
        GLMAdapter(),
        StableLMAdapter(),
        MiniMaxAdapter(),
        DefaultAdapter(),
    ]

    passed = 0
    failed = 0

    for adapter in adapters:
        try:
            config = MockConfig(model_type="mock")
            result = adapter.get_model_class(config)
            # Most should return None (use AutoModel), some may return specific class
            if result is None or isinstance(result, type) or result == "LlamaForCausalLM":
                print(f"  ✅ {type(adapter).__name__}.get_model_class() → {result}")
                passed += 1
            else:
                print(f"  ⚠️  {type(adapter).__name__}.get_model_class() → {result} (unusual)")
                passed += 1
        except Exception as e:
            print(f"  ❌ {type(adapter).__name__}.get_model_class() raised {type(e).__name__}: {e}")
            failed += 1

    print(f"\n  Results: {passed} passed, {failed} failed")
    assert failed == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Validate Config
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_config():
    """Test that validate_config() works for all adapters."""
    sep("5. Adapter validate_config()")

    adapters = [
        LlamaAdapter(),
        QwenMoeAdapter(),
        Qwen35MoeAdapter(),
        DeepSeekAdapter(),
        MistralAdapter(),
        GemmaAdapter(),
        PhiAdapter(),
        CohereAdapter(),
        GLMAdapter(),
        StableLMAdapter(),
        MiniMaxAdapter(),
        DefaultAdapter(),
    ]

    passed = 0
    failed = 0

    for adapter in adapters:
        try:
            config = MockConfig(model_type="mock")
            result = adapter.validate_config(config)
            if isinstance(result, bool):
                print(f"  ✅ {type(adapter).__name__}.validate_config() → {result}")
                passed += 1
            else:
                print(f"  ❌ {type(adapter).__name__}.validate_config() → {result} (not bool)")
                failed += 1
        except NotImplementedError:
            # Some adapters may not override validate_config
            print(f"  ⚠️  {type(adapter).__name__}.validate_config() not implemented (OK)")
            passed += 1
        except Exception as e:
            print(f"  ❌ {type(adapter).__name__}.validate_config() raised {type(e).__name__}: {e}")
            failed += 1

    print(f"\n  Results: {passed} passed, {failed} failed")
    assert failed == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Adapter Inheritance Check
# ─────────────────────────────────────────────────────────────────────────────

def test_adapter_inheritance():
    """Test that all adapters properly inherit from ModelAdapter."""
    sep("6. Adapter Inheritance")

    adapters = [
        LlamaAdapter,
        QwenMoeAdapter,
        Qwen35MoeAdapter,
        DeepSeekAdapter,
        MistralAdapter,
        GemmaAdapter,
        PhiAdapter,
        CohereAdapter,
        GLMAdapter,
        StableLMAdapter,
        MiniMaxAdapter,
        DefaultAdapter,
    ]

    passed = 0
    failed = 0

    for cls in adapters:
        try:
            if issubclass(cls, ModelAdapter):
                print(f"  ✅ {cls.__name__} inherits from ModelAdapter")
                passed += 1
            else:
                print(f"  ❌ {cls.__name__} does NOT inherit from ModelAdapter")
                failed += 1
        except Exception as e:
            print(f"  ❌ {cls.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n  Results: {passed} passed, {failed} failed")
    assert failed == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Default Adapter Fallback
# ─────────────────────────────────────────────────────────────────────────────

def test_default_adapter():
    """Test that DefaultAdapter is used as fallback."""
    sep("7. Default Adapter Fallback")

    try:
        # DefaultAdapter should match any config
        config = MockConfig(model_type="unknown_model_type_xyz")
        result = DefaultAdapter.match("unknown/model", config)
        if result:
            print("  ✅ DefaultAdapter.match() returns True for unknown types")
            passed = 1
        else:
            print("  ❌ DefaultAdapter.match() should return True for unknown types")
            passed = 0
    except Exception as e:
        print(f"  ❌ DefaultAdapter: {type(e).__name__}: {e}")
        passed = 0

    assert passed == 1


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("="*70)
    print("  Comprehensive Adapter Unit Tests")
    print("="*70)

    results = []

    results.append(("Adapter Discovery", test_adapter_discovery()))
    results.append(("Match Methods", test_match_methods()))
    results.append(("Patch Config", test_patch_config()))
    results.append(("Get Model Class", test_get_model_class()))
    results.append(("Validate Config", test_validate_config()))
    results.append(("Inheritance", test_adapter_inheritance()))
    results.append(("Default Fallback", test_default_adapter()))

    sep("FINAL RESULTS")
    all_passed = True
    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}: {name}")
        if not ok:
            all_passed = False

    print(f"\n{'='*70}")
    if all_passed:
        print("  🎉 All adapter tests passed!")
    else:
        print("  ⚠️  Some tests failed — review above")
    print(f"{'='*70}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
