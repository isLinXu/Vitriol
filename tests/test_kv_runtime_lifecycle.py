"""Runtime preprocessing must respect input changes, SDPA semantics and ownership."""

import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import wraps
from threading import Event

import pytest
import torch
import torch.nn.functional as F

from vitriol.kv.codec import ComputeSkipConfig
from vitriol.patches.kv_runtime_patches import KVRuntimePatchConfig, KVRuntimePatcher


@pytest.fixture
def runtime(monkeypatch):
    original = F.scaled_dot_product_attention
    patchers = []

    def make(**kwargs):
        patcher = KVRuntimePatcher(KVRuntimePatchConfig(decode_only=False, **kwargs))
        patchers.append(patcher)
        return patcher

    yield make, original
    for patcher in reversed(patchers):
        patcher.restore()
    monkeypatch.setattr(F, "scaled_dot_product_attention", original)


def _inputs(dtype=torch.float32):
    generator = torch.Generator().manual_seed(31)
    return [torch.randn(1, 2, length, 16, generator=generator).to(dtype) for length in (2, 5, 5)]


def test_runtime_config_does_not_share_mutable_adaptive_codec():
    first, second = KVRuntimePatchConfig(), KVRuntimePatchConfig()
    first.adaptive_bits.target_avg_bits = 7
    assert second.adaptive_bits.target_avg_bits == 3.5


def _uncached(patcher, original, query, key, value):
    fresh = KVRuntimePatcher(replace(patcher.cfg, preprocess_cache_max_entries=0))
    current = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = original
    try:
        fresh.apply()
        return F.scaled_dot_product_attention(query, key, value)
    finally:
        fresh.restore()
        F.scaled_dot_product_attention = current


@pytest.mark.parametrize("mode", ["turbo", "adaptive"])
@pytest.mark.parametrize("changed", [0, 1, 2])
def test_cache_revalidates_inplace_query_key_and_value_changes(runtime, mode, changed):
    make, original = runtime
    patcher = make(enable_turbo_quant=mode == "turbo", enable_adaptive_bits=mode == "adaptive")
    tensors = _inputs()
    patcher.apply()
    F.scaled_dot_product_attention(*tensors)
    F.scaled_dot_product_attention(*tensors)
    assert patcher.stats()["preprocess_cache_hits"] == 1
    tensors[changed].mul_(7).add_(0.5)
    actual = F.scaled_dot_product_attention(*tensors)
    expected = _uncached(patcher, original, *tensors)
    torch.testing.assert_close(actual, expected)


def test_cache_identity_collision_does_not_reuse_another_tensor(runtime, monkeypatch):
    make, original = runtime
    # Deterministically exercise object-ID reuse without relying on allocator timing.
    monkeypatch.setattr("vitriol.patches.kv_runtime_patches.id", lambda obj: 17, raising=False)
    patcher = make(enable_turbo_quant=True)
    patcher.apply()
    tensors = _inputs()
    F.scaled_dot_product_attention(*tensors)
    different = [x + 10 for x in tensors]
    torch.testing.assert_close(F.scaled_dot_product_attention(*different), _uncached(patcher, original, *different))


def test_inference_tensors_without_version_counters_are_not_cached(runtime):
    make, original = runtime
    patcher = make(enable_turbo_quant=True)
    patcher.apply()
    with torch.inference_mode():
        query, key, value = _inputs()
        F.scaled_dot_product_attention(query, key, value)
        value.add_(3)
        actual = F.scaled_dot_product_attention(query, key, value)
        torch.testing.assert_close(actual, _uncached(patcher, original, query, key, value))
    assert patcher.stats()["preprocess_cache_entries"] == 0


def test_preprocessing_graph_is_not_reused_across_backward_passes(runtime):
    make, _ = runtime
    patcher = make(enable_adaptive_bits=True)
    patcher.apply()
    tensors = [x.requires_grad_() for x in _inputs()]
    for _ in range(2):
        output = F.scaled_dot_product_attention(*tensors)
        output.square().sum().backward()
        assert all(torch.isfinite(x.grad).all() for x in tensors)
        for x in tensors:
            x.grad = None
    assert patcher.stats()["preprocess_cache_entries"] == 0


def test_cache_does_not_reuse_values_from_another_autocast_context(runtime):
    make, original = runtime
    patcher = make(enable_turbo_quant=True)
    patcher.apply()
    tensors = _inputs()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        F.scaled_dot_product_attention(*tensors)
        assert patcher.stats()["preprocess_cache_entries"] == 0
    torch.testing.assert_close(F.scaled_dot_product_attention(*tensors), _uncached(patcher, original, *tensors))


def test_restore_on_unapplied_instance_cannot_remove_active_patch(runtime):
    make, _ = runtime
    owner, other = make(), make()
    owner.apply()
    installed = F.scaled_dot_product_attention
    other.restore()
    assert F.scaled_dot_product_attention is installed


def test_conflicting_patch_is_explicit_and_original_owner_remains_active(runtime):
    make, _ = runtime
    owner, other = make(), make(enable_turbo_quant=True)
    owner.apply()
    installed = F.scaled_dot_product_attention
    owner.apply()
    assert F.scaled_dot_product_attention is installed
    with pytest.raises(RuntimeError, match="already active"):
        other.apply()
    other.restore()
    assert F.scaled_dot_product_attention is installed


def test_restore_releases_cached_tensors_and_reapply_is_fresh(runtime):
    make, original = runtime
    patcher = make(enable_turbo_quant=True)
    tensors = _inputs()
    patcher.apply()
    F.scaled_dot_product_attention(*tensors)
    assert patcher.stats()["preprocess_cache_entries"] == 1
    patcher.restore()
    assert patcher.stats()["preprocess_cache_entries"] == 0
    assert F.scaled_dot_product_attention is original
    tensors[2].add_(8)
    patcher.apply()
    torch.testing.assert_close(F.scaled_dot_product_attention(*tensors), _uncached(patcher, original, *tensors))


@pytest.mark.parametrize("preserve_metadata", [False, True])
def test_restored_patch_delegates_through_foreign_wrapper_and_can_be_reapplied(runtime, preserve_metadata):
    make, original = runtime
    patcher = make(enable_turbo_quant=True)
    tensors = _inputs()
    patcher.apply()
    installed = F.scaled_dot_product_attention

    def foreign(*args, **kwargs):
        return installed(*args, **kwargs)

    if preserve_metadata:
        foreign = wraps(installed)(foreign)
    F.scaled_dot_product_attention = foreign
    patcher.restore()
    assert F.scaled_dot_product_attention is foreign
    torch.testing.assert_close(foreign(*tensors), original(*tensors))
    patcher.apply()
    torch.testing.assert_close(F.scaled_dot_product_attention(*tensors), _uncached(patcher, original, *tensors))
    patcher.restore()
    assert F.scaled_dot_product_attention is foreign


def test_context_manager_restores_after_attention_failure(runtime):
    make, original = runtime
    patcher = make(enable_turbo_quant=True)
    with pytest.raises(RuntimeError, match="intentional"):
        with patcher:
            F.scaled_dot_product_attention(*_inputs())
            raise RuntimeError("intentional")
    assert F.scaled_dot_product_attention is original
    assert patcher.stats()["preprocess_cache_entries"] == 0


@pytest.mark.parametrize("mode", ["skip", "sparse"])
@pytest.mark.parametrize("positional", [False, True])
def test_runtime_preserves_mask_dropout_causality_and_custom_scale(runtime, mode, positional):
    make, original = runtime
    patcher = make(enable_compute_skip=mode == "skip", compute_skip=ComputeSkipConfig(block_size=4, epsilon=0),
                   enable_sparse_v=mode == "sparse", sparse_v_threshold=0)
    query, key, value = _inputs()
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[:, 2:] = False
    expected = original(query, key, value, mask, 0, False, scale=0.125)
    patcher.apply()
    actual = (F.scaled_dot_product_attention(query, key, value, mask, 0, False, scale=0.125) if positional else
              F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=0.125))
    torch.testing.assert_close(actual, expected)
    # Keep key length block-aligned so both implementations draw the same
    # number of dropout masks for this seeded comparison.
    key, value = key[..., :4, :], value[..., :4, :]
    torch.manual_seed(12)
    expected = original(query, key, value, None, 0.5, True, scale=0.125)
    torch.manual_seed(12)
    actual = F.scaled_dot_product_attention(query, key, value, None, 0.5, True, scale=0.125)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("mode", ["skip", "sparse"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64])
def test_runtime_gqa_batched_masks_and_distinct_value_dimension(runtime, mode, dtype):
    make, original = runtime
    patcher = make(enable_compute_skip=mode == "skip", compute_skip=ComputeSkipConfig(block_size=4, epsilon=0),
                   enable_sparse_v=mode == "sparse", sparse_v_threshold=0)
    torch.manual_seed(9)
    query = torch.randn(2, 6, 3, 16).to(dtype)
    key = torch.randn(2, 2, 5, 16).to(dtype)
    value = torch.randn(2, 2, 5, 8).to(dtype)
    mask = torch.ones(2, 6, 3, 5, dtype=torch.bool)
    mask[0, :, 0] = False
    mask[1, :, :, -2:] = False
    expected = original(query, key.repeat_interleave(3, 1), value.repeat_interleave(3, 1), attn_mask=mask, scale=0.125)
    patcher.apply()
    actual = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=0.125, enable_gqa=True)
    assert actual.dtype == dtype and torch.isfinite(actual).all()
    tolerance = {torch.bfloat16: 0.015, torch.float16: 0.002, torch.float32: 2e-6, torch.float64: 1e-10}[dtype]
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("mode", ["skip", "sparse"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fully_masked_custom_attention_has_zero_output_and_finite_gradients(runtime, mode, dtype):
    make, _ = runtime
    patcher = make(enable_compute_skip=mode == "skip", compute_skip=ComputeSkipConfig(block_size=4, epsilon=0),
                   enable_sparse_v=mode == "sparse", sparse_v_threshold=0)
    tensors = [t.requires_grad_() for t in _inputs(dtype)]
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[0] = False
    with patcher:
        output = F.scaled_dot_product_attention(*tensors, attn_mask=mask)
        assert torch.count_nonzero(output[..., 0, :]) == 0
        output.square().sum().backward()
    assert all(torch.isfinite(t.grad).all() for t in tensors)
    assert torch.count_nonzero(tensors[0].grad[..., 0, :]) == 0


def test_cache_does_not_keep_original_query_and_kv_alive(runtime):
    make, _ = runtime
    patcher = make(enable_turbo_quant=True)
    with patcher:
        tensors = _inputs()
        refs = [weakref.ref(t) for t in tensors]
        F.scaled_dot_product_attention(*tensors)
        del tensors
        assert all(ref() is None for ref in refs)


def test_nested_context_keeps_outer_patch_installed(runtime):
    make, original = runtime
    patcher = make()
    with patcher:
        installed = F.scaled_dot_product_attention
        with patcher:
            assert F.scaled_dot_product_attention is installed
        assert F.scaled_dot_product_attention is installed
    assert F.scaled_dot_product_attention is original


def test_restore_during_inflight_preprocessing_does_not_repopulate_cache(runtime, monkeypatch):
    make, original = runtime
    started, proceed = Event(), Event()

    def delayed_quantize(tensor, **kwargs):
        started.set()
        assert proceed.wait(5)
        return tensor + 1

    monkeypatch.setattr("vitriol.patches.kv_runtime_patches.turbo_quantize", delayed_quantize)
    patcher = make(enable_turbo_quant=True)
    patcher.apply()
    with ThreadPoolExecutor(max_workers=1) as executor:
        task = executor.submit(F.scaled_dot_product_attention, *_inputs())
        try:
            assert started.wait(5)
            patcher.restore()
        finally:
            proceed.set()
        assert torch.isfinite(task.result(timeout=5)).all()
    assert F.scaled_dot_product_attention is original
    assert patcher.stats()["preprocess_cache_entries"] == 0


def test_compute_skip_preserves_double_precision_for_value_norms(runtime):
    make, original = runtime
    patcher = make(enable_compute_skip=True, compute_skip=ComputeSkipConfig(block_size=4, epsilon=0))
    query, key, value = _inputs(torch.float64)
    value = value * 1e20
    expected = original(query, key, value)
    with patcher:
        actual = F.scaled_dot_product_attention(query, key, value)
    torch.testing.assert_close(actual, expected, atol=0, rtol=1e-10)
