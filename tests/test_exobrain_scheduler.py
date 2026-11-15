"""Regression coverage for ExoBrain scheduling and KV compression boundaries."""

import pytest
import torch

from vitriol.kv.exobrain.scheduler import (
    AdaptiveInjectionScheduler,
    BrainKVCompressor,
    KVPrefetcher,
    compute_perplexity_from_logits,
)
from vitriol.kv.exobrain.teacher import TeacherKVCache


def test_mean_pool_compression_keeps_partial_tail_window():
    key = torch.arange(10, dtype=torch.float32).reshape(1, 1, 10, 1)
    value = key + 100

    compressed_key, compressed_value = BrainKVCompressor(
        method="mean_pool",
        pool_window=4,
    ).compress(0, key, value)

    assert compressed_key.shape == (1, 1, 3, 1)
    assert compressed_value.shape == compressed_key.shape
    assert torch.equal(compressed_key.flatten(), torch.tensor([1.5, 5.5, 8.5]))
    assert torch.equal(compressed_value.flatten(), torch.tensor([101.5, 105.5, 108.5]))


def test_topk_compression_keeps_selected_positions_in_sequence_order():
    key = torch.tensor([1.0, 3.0, 4.0, 2.0]).reshape(1, 1, 4, 1)
    value = key + 10

    compressed_key, compressed_value = BrainKVCompressor(
        method="topk_spatial",
        compression_ratio=0.5,
    ).compress(0, key, value)

    assert torch.equal(compressed_key.flatten(), torch.tensor([3.0, 4.0]))
    assert torch.equal(compressed_value.flatten(), torch.tensor([13.0, 14.0]))


def test_topk_compression_supports_distinct_key_value_dimensions():
    key = torch.tensor([1.0, 3.0, 4.0, 2.0]).reshape(1, 1, 4, 1)
    value = torch.arange(12, dtype=torch.float32).reshape(1, 1, 4, 3)

    compressed_key, compressed_value = BrainKVCompressor(
        method="topk_spatial",
        compression_ratio=0.5,
    ).compress(0, key, value)

    assert compressed_key.shape == (1, 1, 2, 1)
    assert compressed_value.shape == (1, 1, 2, 3)
    assert torch.equal(compressed_value, value[:, :, 1:3, :])


@pytest.mark.parametrize(
    "options",
    [
        {"method": "unknown"},
        {"compression_ratio": 0.0},
        {"compression_ratio": 1.1},
        {"compression_ratio": float("nan")},
        {"method": "quantize_8bit", "quantize_bits": 3},
        {"method": "mean_pool", "pool_window": 0},
        {"method": "svd_lowrank", "svd_rank": 0},
    ],
)
def test_compressor_rejects_invalid_configuration(options):
    with pytest.raises(ValueError):
        BrainKVCompressor(**options)


def test_compressor_rejects_key_value_sequence_mismatch():
    key = torch.zeros(1, 1, 4, 2)
    value = torch.zeros(1, 1, 3, 2)

    with pytest.raises(ValueError, match="sequence"):
        BrainKVCompressor(method="mean_pool").compress(0, key, value)


def test_perplexity_rejects_out_of_vocab_targets_instead_of_clamping():
    logits = torch.zeros(1, 1, 3)
    targets = torch.tensor([[3]])

    with pytest.raises(ValueError, match="target"):
        compute_perplexity_from_logits(logits, targets)


def test_perplexity_uses_epsilon_to_bound_extreme_loss():
    logits = torch.tensor([[[0.0, -1000.0]]])
    targets = torch.tensor([[1]])

    perplexity = compute_perplexity_from_logits(logits, targets, eps=1e-6)

    assert perplexity == pytest.approx(1e6, rel=1e-5)


def test_perplexity_matches_uniform_distribution():
    perplexity = compute_perplexity_from_logits(
        torch.zeros(1, 2, 2),
        torch.tensor([[0, 1]]),
    )

    assert perplexity == pytest.approx(2.0)


@pytest.mark.parametrize(
    "logits,targets",
    [
        (torch.zeros(1, 2), torch.zeros(1, dtype=torch.long)),
        (torch.zeros(1, 2, 3), torch.zeros(1, 1, dtype=torch.long)),
        (torch.empty(1, 0, 3), torch.empty(1, 0, dtype=torch.long)),
    ],
)
def test_perplexity_rejects_invalid_shapes_and_empty_targets(logits, targets):
    with pytest.raises(ValueError):
        compute_perplexity_from_logits(logits, targets)


@pytest.mark.parametrize("eps", [0.0, -1e-3, 1.1, float("nan")])
def test_perplexity_rejects_invalid_epsilon(eps):
    with pytest.raises(ValueError, match="eps"):
        compute_perplexity_from_logits(
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, dtype=torch.long),
            eps=eps,
        )


@pytest.mark.parametrize("options", [{"window_size": 0}, {"warmup_steps": -1}, {"min_injection_rate": 1.1}])
def test_scheduler_rejects_invalid_configuration(options):
    with pytest.raises(ValueError):
        AdaptiveInjectionScheduler(**options)


def test_scheduler_maintains_minimum_injection_rate():
    scheduler = AdaptiveInjectionScheduler(
        strategy="never",
        warmup_steps=0,
        min_injection_rate=0.5,
    )

    decisions = [scheduler.should_inject() for _ in range(4)]

    assert decisions == [True, False, True, False]
    assert scheduler.stats["injection_rate"] == 0.5


def test_threshold_scheduler_applies_warmup_then_strict_threshold():
    scheduler = AdaptiveInjectionScheduler(
        strategy="threshold",
        ppl_threshold=2.0,
        warmup_steps=1,
        min_injection_rate=0.0,
    )

    assert scheduler.should_inject(ppl=0.5)
    assert not scheduler.should_inject(ppl=2.0)
    assert scheduler.should_inject(ppl=2.1)


def test_relative_scheduler_uses_recent_perplexity_window():
    scheduler = AdaptiveInjectionScheduler(
        strategy="relative",
        relative_alpha=2.0,
        warmup_steps=0,
        window_size=2,
        min_injection_rate=0.0,
    )

    assert scheduler.should_inject(ppl=5.0)  # No observations yet: fail safe.
    scheduler.record(2.0)
    assert not scheduler.should_inject(ppl=3.0)
    scheduler.record(2.0)
    assert scheduler.should_inject(ppl=5.0)
    scheduler.record(10.0)
    assert not scheduler.should_inject(ppl=11.0)  # Recent mean is (2 + 10) / 2.
    assert scheduler.stats["avg_ppl"] == 6.0


def test_entropy_and_always_scheduler_strategies():
    entropy_scheduler = AdaptiveInjectionScheduler(
        strategy="entropy",
        entropy_threshold=0.8,
        warmup_steps=0,
        min_injection_rate=0.0,
    )
    always_scheduler = AdaptiveInjectionScheduler(
        strategy="always",
        warmup_steps=0,
        min_injection_rate=0.0,
    )

    assert not entropy_scheduler.should_inject(entropy=0.8)
    assert entropy_scheduler.should_inject(entropy=0.81)
    assert always_scheduler.should_inject()


def test_scheduler_reports_zero_skip_rate_before_any_steps_and_after_reset():
    scheduler = AdaptiveInjectionScheduler()

    assert scheduler.stats["skip_rate"] == 0.0
    scheduler.should_inject(ppl=100.0)
    scheduler.reset()
    assert scheduler.stats["skip_rate"] == 0.0


@pytest.mark.parametrize("ppl", [float("nan"), float("inf"), -1.0])
def test_scheduler_rejects_invalid_perplexity_observations(ppl):
    scheduler = AdaptiveInjectionScheduler()

    with pytest.raises(ValueError, match="ppl"):
        scheduler.record(ppl)


def test_fake_quantization_reports_actual_storage_size():
    key = torch.linspace(-1, 1, 16).reshape(1, 1, 4, 4)
    value = key + 0.1
    compressor = BrainKVCompressor(method="quantize_8bit")

    compressed_key, compressed_value = compressor.compress(0, key, value)

    assert compressed_key.dtype == key.dtype
    assert compressed_value.dtype == value.dtype
    assert compressor.stats["avg_compression_ratio"] == 1.0


def test_compress_teacher_cache_preserves_prompt_metadata():
    key = torch.arange(10, dtype=torch.float32).reshape(1, 1, 5, 2)
    cache = TeacherKVCache(
        kv_pairs={0: (key, key + 10)},
        model_id="teacher",
        num_layers=1,
        hidden_size=2,
        num_heads=1,
        head_dim=2,
        sequence_length=5,
    )

    compressed = BrainKVCompressor(
        method="topk_spatial",
        compression_ratio=0.4,
    ).compress_teacher_cache(cache)

    assert compressed.model_id == "teacher_compressed"
    assert compressed.sequence_length == cache.sequence_length
    assert compressed.kv_pairs[0][0].shape[2] == 2


def test_svd_failure_logs_and_retains_original_tensor(monkeypatch, caplog):
    key = torch.eye(3).reshape(1, 1, 3, 3)

    def fail_svd(*args, **kwargs):
        raise RuntimeError("synthetic SVD failure")

    monkeypatch.setattr(torch.linalg, "svd", fail_svd)
    compressor = BrainKVCompressor(method="svd_lowrank", svd_rank=1)

    with caplog.at_level("WARNING", logger="vitriol.kv.exobrain.scheduler"):
        compressed_key, _ = compressor.compress(0, key, key)

    assert torch.equal(compressed_key, key)
    assert "synthetic SVD failure" in caplog.text


def test_empty_topk_cache_is_preserved():
    key = torch.empty(1, 1, 0, 2)
    value = torch.empty_like(key)

    compressed_key, compressed_value = BrainKVCompressor(method="topk_spatial").compress(0, key, value)

    assert compressed_key.shape == key.shape
    assert compressed_value.shape == value.shape


def test_kv_prefetcher_cache_hit_miss_and_replacement():
    prefetcher = KVPrefetcher(brain_bus=object(), device="cpu")
    key = torch.ones(1, 1, 2, 3, requires_grad=True)
    value = torch.full((1, 1, 2, 3), 2.0, requires_grad=True)

    assert prefetcher.cache_projected_kv({2: (key, value)}) == 1
    assert prefetcher.is_cached(2)
    assert prefetcher.get_projected_kv(1) is None
    cached_key, cached_value = prefetcher.get_projected_kv(2)
    assert not cached_key.requires_grad
    assert not cached_value.requires_grad
    assert torch.equal(cached_key, key)
    assert torch.equal(cached_value, value)

    assert prefetcher.cache_projected_kv({4: (key, value)}) == 1
    assert not prefetcher.is_cached(2)
    assert prefetcher.is_cached(4)
    assert prefetcher.stats == {
        "cache_hits": 1,
        "cache_misses": 1,
        "prefetch_count": 2,
        "hit_rate": 0.5,
        "cached_layers": [4],
    }


def test_svd_lowrank_applies_requested_rank_instead_of_silent_noop():
    key = torch.diag(torch.tensor([3.0, 2.0, 1.0])).reshape(1, 1, 3, 3)
    value = torch.diag(torch.tensor([2.0, 1.0, 0.5])).reshape(1, 1, 3, 3)

    compressed_key, compressed_value = BrainKVCompressor(
        method="svd_lowrank",
        svd_rank=1,
    ).compress(0, key, value)

    assert compressed_key.shape == key.shape
    assert compressed_value.shape == value.shape
    assert not torch.allclose(compressed_key, key)
    assert torch.allclose(compressed_key[0, 0], torch.diag(torch.tensor([3.0, 0.0, 0.0])), atol=1e-5)
