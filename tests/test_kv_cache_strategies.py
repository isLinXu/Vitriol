import pytest
import torch
import torch.nn.functional as F

from vitriol.kv.cache_store import KVCacheStore, KVCacheStoreConfig
from vitriol.kv.codec import PackedKVTensor, ResidualQJLPackedTensor, approx_inner_product_with_qjl_residual
from vitriol.kv.policy import (
    KVLayerType,
    KVPolicyPreset,
    Turbo3ExactKApproxVPolicy,
    apply_policy_to_store_cfg,
    build_policy,
    classify_kv_layer,
    list_policy_presets,
    resolve_layer_strategy,
)
from vitriol.patches.kv_runtime_patches import KVRuntimePatcher
from vitriol.bench.runner import _collect_policy_insights
from vitriol.patches.kv_runtime_patches import KVRuntimePatchConfig
from vitriol.patches.qwen35_attention_patches import Qwen35AttentionPatcher
from vitriol.patches.qwen35_attention_patches import Qwen35AttentionPatchConfig
from vitriol.patches.turboquant import _qjl_residual_sketch, resolve_turbo_kv_formats, turbo_quantize


def test_resolve_turbo_kv_formats_fractional_bits() -> None:
    k_fmt, v_fmt = resolve_turbo_kv_formats(turbo_bits=3.5)
    assert k_fmt == 3
    assert v_fmt == 4


def test_turbo_quantize_is_deterministic_and_shape_stable() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 5, 64, dtype=torch.float32)

    y1 = turbo_quantize(x, format_type="turbo3")
    y2 = turbo_quantize(x, format_type="turbo3")

    assert y1.shape == x.shape
    assert y1.dtype == x.dtype
    assert torch.allclose(y1, y2)
    assert not torch.allclose(y1, x)


def test_turbo_quantize_rejects_invalid_runtime_params() -> None:
    x = torch.randn(1, 1, 2, 8, dtype=torch.float32)

    with pytest.raises(ValueError, match="block_size"):
        turbo_quantize(x, format_type="turbo3", block_size=0)

    with pytest.raises(ValueError, match="residual_strength"):
        turbo_quantize(x, format_type="turbo3", residual_strength=-0.1)


def test_kv_quant_configs_reject_invalid_thresholds() -> None:
    from vitriol.patches.kv_runtime_patches import KVRuntimePatchConfig
    from vitriol.patches.qwen35_attention_patches import Qwen35AttentionPatchConfig

    with pytest.raises(ValueError, match="turbo_block_size"):
        KVCacheStoreConfig(enable_turbo_quant=True, turbo_block_size=0)

    with pytest.raises(ValueError, match="quantized_kv_start"):
        KVRuntimePatchConfig(enable_turbo_quant=True, quantized_kv_start=-1)

    with pytest.raises(ValueError, match="sparse_v_threshold"):
        Qwen35AttentionPatchConfig(enable_sparse_v=True, sparse_v_threshold=-0.1)


def test_turbo_quantize_residual_hook_can_adjust_output() -> None:
    torch.manual_seed(1)
    x = torch.randn(1, 1, 4, 32, dtype=torch.float32)

    base = turbo_quantize(x, format_type="turbo3")
    adjusted = turbo_quantize(
        x,
        format_type="turbo3",
        residual_hook=lambda residual: 0.1 * residual,
    )

    assert adjusted.shape == x.shape
    assert not torch.allclose(base, adjusted)


def test_turbo_quantize_default_residual_qjl_changes_output() -> None:
    torch.manual_seed(2)
    x = torch.randn(1, 2, 4, 32, dtype=torch.float32)

    without_residual = turbo_quantize(x, format_type="turbo3", use_residual_qjl=False)
    with_residual = turbo_quantize(x, format_type="turbo3", use_residual_qjl=True)

    assert with_residual.shape == x.shape
    assert not torch.allclose(without_residual, with_residual)


def test_runtime_qjl_residual_sketch_does_not_systematically_shrink_residual_projection() -> None:
    torch.manual_seed(22)
    residual = torch.randn(1, 2, 16, 32, dtype=torch.float32)

    correction = _qjl_residual_sketch(residual, strength=1.0)
    ratio = float(((correction * residual).sum(dim=-1).mean()) / ((residual * residual).sum(dim=-1).mean()))

    assert ratio >= 0.78
    assert ratio <= 1.2


def test_turbo_quantize_runtime_residual_qjl_does_not_worsen_reconstruction_error() -> None:
    torch.manual_seed(23)
    x = torch.randn(1, 2, 16, 32, dtype=torch.float32)

    without_residual = turbo_quantize(
        x,
        format_type="turbo3",
        use_residual_qjl=False,
        residual_strength=1.0,
    )
    with_residual = turbo_quantize(
        x,
        format_type="turbo3",
        use_residual_qjl=True,
        residual_strength=1.0,
    )

    mse_without = torch.mean((x - without_residual) ** 2).item()
    mse_with = torch.mean((x - with_residual) ** 2).item()
    mae_without = torch.mean(torch.abs(x - without_residual)).item()
    mae_with = torch.mean(torch.abs(x - with_residual)).item()

    assert mse_with <= mse_without
    assert mae_with <= mae_without


def test_turbo_quantize_custom_hook_overrides_default_residual_qjl() -> None:
    torch.manual_seed(3)
    x = torch.randn(1, 1, 4, 32, dtype=torch.float32)

    overridden = turbo_quantize(
        x,
        format_type="turbo3",
        use_residual_qjl=True,
        residual_hook=lambda residual: torch.zeros_like(residual),
    )
    no_residual = turbo_quantize(x, format_type="turbo3", use_residual_qjl=False)

    assert torch.allclose(overridden, no_residual)


def test_runtime_patch_config_aligns_fractional_bits() -> None:
    cfg = KVRuntimePatchConfig(enable_turbo_quant=True, turbo_bits=3.5, quantized_kv_start=256)
    assert cfg.turbo_k_format == 3
    assert cfg.turbo_v_format == 4
    assert cfg.quantized_kv_start == 256
    assert cfg.enable_turbo_residual_qjl is True


def test_qwen35_patch_config_aligns_fractional_bits() -> None:
    cfg = Qwen35AttentionPatchConfig(enable_turbo_quant=True, turbo_bits=3.5, quantized_kv_start=512)
    assert cfg.turbo_k_format == 3
    assert cfg.turbo_v_format == 4
    assert cfg.quantized_kv_start == 512
    assert cfg.enable_turbo_residual_qjl is True


def test_kv_cache_store_delays_quantization_until_threshold() -> None:
    torch.manual_seed(0)
    cfg = KVCacheStoreConfig(enable_turbo_quant=True, turbo_bits=3.5, quantized_kv_start=8)
    store = KVCacheStore(cfg)

    key_prefill = torch.randn(1, 2, 4, 8)
    value_prefill = torch.randn(1, 2, 4, 8)
    store.set_prefill(key_prefill, value_prefill)

    assert torch.allclose(store._k_enc, store._k_raw)
    assert torch.allclose(store._v_enc, store._v_raw)

    key_decode = torch.randn(1, 2, 4, 8)
    value_decode = torch.randn(1, 2, 4, 8)
    store.append(key_decode, value_decode)

    assert store.seq_len == 8
    assert isinstance(store._k_enc, ResidualQJLPackedTensor)
    assert isinstance(store._v_enc, ResidualQJLPackedTensor)


def test_kv_cache_store_can_quantize_values_only() -> None:
    torch.manual_seed(1)
    cfg = KVCacheStoreConfig(
        enable_turbo_quant=True,
        turbo_bits=3.5,
        turbo_quantize_k=False,
        turbo_quantize_v=True,
        quantized_kv_start=0,
    )
    store = KVCacheStore(cfg)

    key = torch.randn(1, 2, 4, 8)
    value = torch.randn(1, 2, 4, 8)
    store.set_prefill(key, value)

    assert torch.allclose(store._k_enc, store._k_raw)
    assert isinstance(store._v_enc, ResidualQJLPackedTensor)


def test_estimated_kv_bytes_uses_real_packed_storage() -> None:
    torch.manual_seed(2)
    cfg = KVCacheStoreConfig(enable_turbo_quant=True, turbo_bits=3.5, quantized_kv_start=0)
    store = KVCacheStore(cfg)

    key = torch.randn(1, 2, 8, 32)
    value = torch.randn(1, 2, 8, 32)
    store.set_prefill(key, value)

    quantized_bytes = store.estimated_kv_bytes()
    bf16_bytes = int((key.numel() + value.numel()) * 2)

    assert quantized_bytes > 0
    assert quantized_bytes < bf16_bytes


def test_kv_cache_store_uses_residual_qjl_packed_tensors_when_enabled() -> None:
    torch.manual_seed(10)
    cfg = KVCacheStoreConfig(
        enable_turbo_quant=True,
        turbo_bits=3.5,
        quantized_kv_start=0,
        enable_turbo_residual_qjl=True,
    )
    store = KVCacheStore(cfg)

    key = torch.randn(1, 2, 8, 32)
    value = torch.randn(1, 2, 8, 32)
    store.set_prefill(key, value)

    assert isinstance(store._k_enc, ResidualQJLPackedTensor)
    assert isinstance(store._v_enc, ResidualQJLPackedTensor)


def test_kv_cache_store_uses_base_packed_tensors_when_residual_qjl_disabled() -> None:
    torch.manual_seed(11)
    cfg = KVCacheStoreConfig(
        enable_turbo_quant=True,
        turbo_bits=3.5,
        quantized_kv_start=0,
        enable_turbo_residual_qjl=False,
    )
    store = KVCacheStore(cfg)

    key = torch.randn(1, 2, 8, 32)
    value = torch.randn(1, 2, 8, 32)
    store.set_prefill(key, value)

    assert isinstance(store._k_enc, PackedKVTensor)
    assert isinstance(store._v_enc, PackedKVTensor)


def test_kv_cache_store_residual_strength_changes_decoded_tensor() -> None:
    torch.manual_seed(12)
    key = torch.randn(1, 2, 8, 32)
    value = torch.randn(1, 2, 8, 32)

    weak = KVCacheStore(
        KVCacheStoreConfig(
            enable_turbo_quant=True,
            turbo_bits=3.5,
            quantized_kv_start=0,
            enable_turbo_residual_qjl=True,
            turbo_residual_strength=0.0,
        )
    )
    strong = KVCacheStore(
        KVCacheStoreConfig(
            enable_turbo_quant=True,
            turbo_bits=3.5,
            quantized_kv_start=0,
            enable_turbo_residual_qjl=True,
            turbo_residual_strength=1.0,
        )
    )

    weak.set_prefill(key, value)
    strong.set_prefill(key, value)

    weak_decoded = weak._decode_tensor(weak._k_enc)
    strong_decoded = strong._decode_tensor(strong._k_enc)
    assert not torch.allclose(weak_decoded, strong_decoded)


def test_kv_cache_store_can_append_when_residual_qjl_packed_tensors_are_enabled() -> None:
    torch.manual_seed(13)
    cfg = KVCacheStoreConfig(
        enable_turbo_quant=True,
        turbo_bits=3.5,
        quantized_kv_start=0,
        enable_turbo_residual_qjl=True,
    )
    store = KVCacheStore(cfg)

    key_prefill = torch.randn(1, 2, 4, 32)
    value_prefill = torch.randn(1, 2, 4, 32)
    key_decode = torch.randn(1, 2, 4, 32)
    value_decode = torch.randn(1, 2, 4, 32)

    store.set_prefill(key_prefill, value_prefill)
    store.append(key_decode, value_decode)

    assert store.seq_len == 8
    assert isinstance(store._k_enc, ResidualQJLPackedTensor)
    assert isinstance(store._v_enc, ResidualQJLPackedTensor)
    decoded_k = store._decode_tensor(store._k_enc)
    decoded_v = store._decode_tensor(store._v_enc)
    assert decoded_k.shape[-2] == 8
    assert decoded_v.shape[-2] == 8


def test_kv_cache_store_packed_storage_does_not_reuse_turbo_quantize_path(monkeypatch) -> None:
    torch.manual_seed(14)
    cfg = KVCacheStoreConfig(
        enable_turbo_quant=True,
        turbo_bits=3.5,
        quantized_kv_start=0,
        enable_turbo_residual_qjl=True,
    )
    store = KVCacheStore(cfg)
    key = torch.randn(1, 2, 8, 32)
    value = torch.randn(1, 2, 8, 32)
    calls = {"turbo_quantize": 0}

    def fake_turbo_quantize(tensor, format_type="turbo3", block_size=32, **kwargs):
        calls["turbo_quantize"] += 1
        return tensor

    monkeypatch.setattr(
        "vitriol.kv.cache_store._get_turbo_utils",
        lambda: (lambda *args, **kwargs: None, fake_turbo_quantize, resolve_turbo_kv_formats),
    )

    store.set_prefill(key, value)

    assert calls["turbo_quantize"] == 0
    assert isinstance(store._k_enc, ResidualQJLPackedTensor)
    assert isinstance(store._v_enc, ResidualQJLPackedTensor)


def test_kv_cache_store_attention_uses_residual_qjl_proxy_for_packed_keys(monkeypatch) -> None:
    torch.manual_seed(20)
    cfg = KVCacheStoreConfig(enable_turbo_quant=True, turbo_bits=3.5, quantized_kv_start=0)
    store = KVCacheStore(cfg)

    key = torch.randn(1, 2, 8, 32)
    value = torch.randn(1, 2, 8, 32)
    query = torch.randn(1, 2, 4, 32)
    store.set_prefill(key, value)

    called = {"proxy": 0}

    def fake_proxy(*args, **kwargs):
        called["proxy"] += 1
        return torch.zeros(1, 2, 4, 32)

    monkeypatch.setattr(store, "_attention_with_residual_qjl_proxy", fake_proxy, raising=False)
    store.attention(query)

    assert called["proxy"] == 1


def test_kv_cache_store_residual_proxy_attention_avoids_dense_key_unpack(monkeypatch) -> None:
    torch.manual_seed(20)
    cfg = KVCacheStoreConfig(
        enable_turbo_quant=True,
        turbo_bits=3.5,
        quantized_kv_start=0,
        enable_turbo_residual_qjl=True,
        turbo_quantize_v=False,
    )
    store = KVCacheStore(cfg)

    key = torch.randn(1, 2, 8, 32)
    value = torch.randn(1, 2, 8, 32)
    query = torch.randn(1, 2, 4, 32)
    store.set_prefill(key, value)

    def fail_unpack(*args, **kwargs):
        raise AssertionError("dense unpack should not be used for proxy attention")

    monkeypatch.setattr("vitriol.kv.cache_store.unpack_qjl_residual_tensor", fail_unpack)
    monkeypatch.setattr("vitriol.kv.cache_store.unpack_blockwise_tensor", fail_unpack)

    out = store.attention(query)

    assert out.shape == (1, 2, 4, 32)


def test_kv_cache_store_residual_proxy_attention_is_not_worse_than_base_decode() -> None:
    torch.manual_seed(21)
    query = torch.randn(1, 2, 4, 32)
    key = torch.randn(1, 2, 16, 32)
    value = torch.randn(1, 2, 16, 32)

    ref = F.scaled_dot_product_attention(query, key, value)

    base_store = KVCacheStore(
        KVCacheStoreConfig(
            enable_turbo_quant=True,
            turbo_bits=3.5,
            quantized_kv_start=0,
            enable_turbo_residual_qjl=False,
            turbo_quantize_v=False,
        )
    )
    proxy_store = KVCacheStore(
        KVCacheStoreConfig(
            enable_turbo_quant=True,
            turbo_bits=3.5,
            quantized_kv_start=0,
            enable_turbo_residual_qjl=True,
            turbo_quantize_v=False,
        )
    )
    base_store.set_prefill(key, value)
    proxy_store.set_prefill(key, value)

    base_out = base_store.attention(query)
    proxy_out = proxy_store.attention(query)

    base_err = torch.mean((base_out - ref) ** 2).item()
    proxy_err = torch.mean((proxy_out - ref) ** 2).item()
    assert proxy_err <= base_err


def test_kv_cache_store_attention_skips_proxy_when_compute_skip_enabled(monkeypatch) -> None:
    torch.manual_seed(22)
    cfg = KVCacheStoreConfig(
        enable_turbo_quant=True,
        turbo_bits=3.5,
        quantized_kv_start=0,
        enable_compute_skip=True,
    )
    store = KVCacheStore(cfg)
    key = torch.randn(1, 2, 8, 32)
    value = torch.randn(1, 2, 8, 32)
    query = torch.randn(1, 2, 4, 32)
    store.set_prefill(key, value)

    called = {"proxy": 0}

    def fake_proxy(*args, **kwargs):
        called["proxy"] += 1
        raise AssertionError("proxy should not be used")

    monkeypatch.setattr(store, "_attention_with_residual_qjl_proxy", fake_proxy, raising=False)
    store.attention(query)
    assert called["proxy"] == 0


def test_kv_cache_store_attention_uses_compute_skip_residual_proxy_when_available(monkeypatch) -> None:
    torch.manual_seed(30)
    cfg = KVCacheStoreConfig(
        enable_turbo_quant=True,
        turbo_bits=3.5,
        quantized_kv_start=0,
        enable_compute_skip=True,
        enable_turbo_residual_qjl=True,
    )
    store = KVCacheStore(cfg)
    key = torch.randn(1, 2, 8, 32)
    value = torch.randn(1, 2, 8, 32)
    query = torch.randn(1, 2, 4, 32)
    store.set_prefill(key, value)

    called = {"joint": 0}

    def fake_joint(*args, **kwargs):
        called["joint"] += 1
        return torch.zeros_like(query)

    monkeypatch.setattr(store, "_attention_with_compute_skip_residual_proxy", fake_joint, raising=False)
    store.attention(query)
    assert called["joint"] == 1


def test_compute_skip_residual_proxy_avoids_dense_residual_rebuild(monkeypatch) -> None:
    torch.manual_seed(30)
    cfg = KVCacheStoreConfig(
        enable_turbo_quant=True,
        turbo_bits=3.5,
        quantized_kv_start=0,
        enable_compute_skip=True,
        enable_turbo_residual_qjl=True,
        turbo_quantize_v=False,
    )
    store = KVCacheStore(cfg)
    key = torch.randn(1, 2, 8, 32)
    value = torch.randn(1, 2, 8, 32)
    query = torch.randn(1, 2, 4, 32)
    store.set_prefill(key, value)

    def fail_unpack(*args, **kwargs):
        raise AssertionError("dense residual rebuild should not be used")

    monkeypatch.setattr("vitriol.kv.cache_store.unpack_qjl_residual_tensor", fail_unpack)

    out = store.attention(query)

    assert out.shape == (1, 2, 4, 32)


def test_compute_skip_residual_proxy_stays_closer_to_residual_reference_than_legacy_skip() -> None:
    torch.manual_seed(31)
    query = torch.randn(1, 1, 4, 32)
    key = torch.randn(1, 1, 256, 32) * 0.01
    value = torch.randn(1, 1, 256, 32) * 0.01
    for idx, scale in zip([64, 128, 233], [8.0, 7.0, 6.0]):
        key[..., idx, :] = query[..., 0, :] * scale
        value[..., idx, :] = query[..., 0, :] * (scale / 2.0)
    ref = F.scaled_dot_product_attention(query, key, value)

    base_store = KVCacheStore(
        KVCacheStoreConfig(
            enable_turbo_quant=True,
            turbo_bits=3.5,
            quantized_kv_start=0,
            enable_compute_skip=True,
            enable_turbo_residual_qjl=False,
            turbo_quantize_v=False,
        )
    )
    proxy_store = KVCacheStore(
        KVCacheStoreConfig(
            enable_turbo_quant=True,
            turbo_bits=3.5,
            quantized_kv_start=0,
            enable_compute_skip=True,
            enable_turbo_residual_qjl=True,
            turbo_quantize_v=False,
        )
    )
    residual_ref_store = KVCacheStore(
        KVCacheStoreConfig(
            enable_turbo_quant=True,
            turbo_bits=3.5,
            quantized_kv_start=0,
            enable_compute_skip=False,
            enable_turbo_residual_qjl=True,
            turbo_quantize_v=False,
        )
    )
    base_store.set_prefill(key, value)
    proxy_store.set_prefill(key, value)
    residual_ref_store.set_prefill(key, value)

    base_out = base_store.attention(query)
    proxy_out = proxy_store.attention(query)
    residual_ref = residual_ref_store.attention(query)

    base_err = torch.mean((base_out - residual_ref) ** 2).item()
    proxy_err = torch.mean((proxy_out - residual_ref) ** 2).item()
    assert proxy_err <= base_err


def test_compute_skip_keeps_legacy_path_for_non_residual_keys(monkeypatch) -> None:
    torch.manual_seed(32)
    cfg = KVCacheStoreConfig(
        enable_turbo_quant=True,
        turbo_bits=3.5,
        quantized_kv_start=0,
        enable_compute_skip=True,
        enable_turbo_residual_qjl=False,
    )
    store = KVCacheStore(cfg)
    key = torch.randn(1, 2, 8, 32)
    value = torch.randn(1, 2, 8, 32)
    query = torch.randn(1, 2, 4, 32)
    store.set_prefill(key, value)

    called = {"joint": 0}

    def fake_joint(*args, **kwargs):
        called["joint"] += 1
        raise AssertionError("joint helper should not be used")

    monkeypatch.setattr(store, "_attention_with_compute_skip_residual_proxy", fake_joint, raising=False)
    store.attention(query)
    assert called["joint"] == 0


def test_kv_cache_store_attention_uses_sparse_v_residual_proxy_when_available(monkeypatch) -> None:
    torch.manual_seed(40)
    cfg = KVCacheStoreConfig(
        enable_turbo_quant=True,
        turbo_bits=3.5,
        quantized_kv_start=0,
        enable_sparse_v=True,
        enable_turbo_residual_qjl=True,
        enable_compute_skip=False,
    )
    store = KVCacheStore(cfg)
    key = torch.randn(1, 2, 16, 32)
    value = torch.randn(1, 2, 16, 32)
    query = torch.randn(1, 2, 4, 32)
    store.set_prefill(key, value)

    called = {"joint": 0}

    def fake_joint(*args, **kwargs):
        called["joint"] += 1
        return torch.zeros_like(query)

    monkeypatch.setattr(store, "_attention_with_sparse_v_residual_proxy", fake_joint, raising=False)
    store.attention(query)
    assert called["joint"] == 1


def test_sparse_v_residual_proxy_keeps_needle_positions_at_least_as_well_as_legacy_sparse_v() -> None:
    torch.manual_seed(41)
    query = torch.randn(1, 1, 4, 32)
    key = torch.randn(1, 1, 256, 32) * 0.01
    value = torch.randn(1, 1, 256, 32) * 0.01
    needle_positions = [64, 128, 233]
    for idx, scale in zip(needle_positions, [8.0, 7.0, 6.0]):
        key[..., idx, :] = query[..., 0, :] * scale
        value[..., idx, :] = query[..., 0, :] * (scale / 2.0)

    legacy_sparse = KVCacheStore(
        KVCacheStoreConfig(
            enable_turbo_quant=True,
            turbo_bits=3.5,
            quantized_kv_start=0,
            enable_sparse_v=True,
            enable_turbo_residual_qjl=False,
            turbo_quantize_v=False,
            enable_compute_skip=False,
        )
    )
    residual_sparse = KVCacheStore(
        KVCacheStoreConfig(
            enable_turbo_quant=True,
            turbo_bits=3.5,
            quantized_kv_start=0,
            enable_sparse_v=True,
            enable_turbo_residual_qjl=True,
            turbo_quantize_v=False,
            enable_compute_skip=False,
        )
    )
    legacy_sparse.set_prefill(key, value)
    residual_sparse.set_prefill(key, value)

    scale_factor = 1.0 / (query.size(-1) ** 0.5)
    threshold = float(legacy_sparse.cfg.sparse_v_threshold)

    legacy_key = legacy_sparse._decode_tensor(legacy_sparse._k_enc)
    legacy_scores = (query @ legacy_key.transpose(-2, -1)) * scale_factor
    legacy_w = torch.softmax(legacy_scores, dim=-1)
    legacy_keep = legacy_w > threshold

    residual_scores = approx_inner_product_with_qjl_residual(query, residual_sparse._k_enc) * scale_factor
    residual_w = torch.softmax(residual_scores, dim=-1)
    residual_keep = residual_w > threshold

    legacy_hits = 0
    residual_hits = 0
    for idx in needle_positions:
        if bool(legacy_keep[..., idx].any()):
            legacy_hits += 1
        if bool(residual_keep[..., idx].any()):
            residual_hits += 1

    assert residual_hits >= legacy_hits


def test_sparse_v_keeps_legacy_path_for_non_residual_keys(monkeypatch) -> None:
    torch.manual_seed(42)
    cfg = KVCacheStoreConfig(
        enable_turbo_quant=True,
        turbo_bits=3.5,
        quantized_kv_start=0,
        enable_sparse_v=True,
        enable_turbo_residual_qjl=False,
        enable_compute_skip=False,
    )
    store = KVCacheStore(cfg)
    key = torch.randn(1, 2, 16, 32)
    value = torch.randn(1, 2, 16, 32)
    query = torch.randn(1, 2, 4, 32)
    store.set_prefill(key, value)

    called = {"joint": 0}

    def fake_joint(*args, **kwargs):
        called["joint"] += 1
        raise AssertionError("joint helper should not be used")

    monkeypatch.setattr(store, "_attention_with_sparse_v_residual_proxy", fake_joint, raising=False)
    store.attention(query)
    assert called["joint"] == 0


class _MockHandle:
    def __init__(self, layer_types):
        self.layer_types = list(layer_types)

    def __len__(self) -> int:
        return len(self.layer_types)


def test_policy_classifies_non_full_attention_layers() -> None:
    handle = _MockHandle([
        "full_attention",
        "sliding_attention",
        "mla_attention",
        "linear_attention",
        "compressed_attention",
        "hash_attention",
        "global_attention",
        "mamba",
    ])
    assert classify_kv_layer(handle, 0) == KVLayerType.FULL_ATTENTION
    assert classify_kv_layer(handle, 1) == KVLayerType.SLIDING_WINDOW
    assert classify_kv_layer(handle, 2) == KVLayerType.MLA
    assert classify_kv_layer(handle, 3) == KVLayerType.LINEAR
    assert classify_kv_layer(handle, 4) == KVLayerType.COMPRESSED_ATTENTION
    assert classify_kv_layer(handle, 5) == KVLayerType.HASH_ATTENTION
    assert classify_kv_layer(handle, 6) == KVLayerType.FULL_ATTENTION
    assert classify_kv_layer(handle, 7) == KVLayerType.LINEAR


def test_policy_resolves_full_attention_only_approximation() -> None:
    handle = _MockHandle(["full_attention", "sliding_attention", "full_attention", "mla_attention"])
    policy = Turbo3ExactKApproxVPolicy(
        v_quantize_only_first_n_full_attention_layers=1,
        enable_sparse_v=True,
        sparse_v_only_first_n_full_attention_layers=1,
        enable_compute_skip=True,
        compute_skip_only_first_n_full_attention_layers=1,
        quantized_kv_start=1024,
    )

    s0 = resolve_layer_strategy(policy, handle, 0)
    s1 = resolve_layer_strategy(policy, handle, 1)
    s2 = resolve_layer_strategy(policy, handle, 2)

    assert s0.turbo_quantize_k is True
    assert s0.turbo_quantize_v is True
    assert s0.enable_sparse_v is True
    assert s0.enable_compute_skip is True

    assert s1.layer_type == KVLayerType.SLIDING_WINDOW
    assert s1.turbo_quantize_k is False
    assert s1.turbo_quantize_v is False
    assert s1.enable_sparse_v is False

    assert s2.layer_type == KVLayerType.FULL_ATTENTION
    assert s2.turbo_quantize_k is True
    assert s2.turbo_quantize_v is False
    assert s2.enable_sparse_v is False
    assert s2.enable_compute_skip is False


def test_apply_policy_to_store_cfg_disables_non_full_attention_quantization() -> None:
    base_cfg = KVCacheStoreConfig(enable_turbo_quant=True, turbo_bits=3.5)
    handle = _MockHandle(["full_attention", "sliding_attention", "mla_attention"])
    policy = Turbo3ExactKApproxVPolicy(v_quantize_only_first_n_full_attention_layers=1, quantized_kv_start=2048)

    full_cfg = apply_policy_to_store_cfg(base_cfg, policy, handle, 0)
    sliding_cfg = apply_policy_to_store_cfg(base_cfg, policy, handle, 1)
    mla_cfg = apply_policy_to_store_cfg(base_cfg, policy, handle, 2)

    assert full_cfg.turbo_quantize_k is True
    assert full_cfg.turbo_quantize_v is True
    assert full_cfg.quantized_kv_start == 2048

    assert sliding_cfg.turbo_quantize_k is False
    assert sliding_cfg.turbo_quantize_v is False
    assert mla_cfg.turbo_quantize_k is False
    assert mla_cfg.turbo_quantize_v is False


def test_apply_policy_to_store_cfg_propagates_residual_qjl_controls() -> None:
    base_cfg = KVCacheStoreConfig(enable_turbo_quant=True, turbo_bits=3.5)
    handle = _MockHandle(["full_attention"])
    policy = Turbo3ExactKApproxVPolicy(
        v_quantize_only_first_n_full_attention_layers=1,
        enable_turbo_residual_qjl=False,
        turbo_residual_strength=0.25,
    )

    cfg = apply_policy_to_store_cfg(base_cfg, policy, handle, 0)
    assert cfg.enable_turbo_residual_qjl is False
    assert cfg.turbo_residual_strength == 0.25


def test_builtin_policy_presets_include_ultra_long() -> None:
    presets = {preset.name: preset for preset in list_policy_presets()}
    assert {"safe", "balanced", "fast-balanced", "aggressive", "ultra-long", "deepseek-v4", "hy3"} <= set(presets)

    aggressive = presets["aggressive"]
    fast_balanced = presets["fast-balanced"]
    ultra_long = presets["ultra-long"]
    deepseek_v4 = presets["deepseek-v4"]
    hy3 = presets["hy3"]

    assert fast_balanced.params["enable_turbo_residual_qjl"] is False
    assert aggressive.params["enable_sparse_v"] is True
    assert "enable_compute_skip" not in aggressive.params or aggressive.params["enable_compute_skip"] is False

    assert ultra_long.params["enable_sparse_v"] is True
    assert ultra_long.params["enable_compute_skip"] is True
    assert ultra_long.params["quantized_kv_start"] == 512
    assert deepseek_v4.params["quantized_kv_start"] == 0
    assert deepseek_v4.params["enable_sparse_v"] is False
    assert hy3.params["enable_sparse_v"] is True
    assert hy3.params["v_protect_last_n_full_attention_layers"] == 2


def test_ultra_long_policy_builds_expected_shape() -> None:
    preset = KVPolicyPreset.ultra_long_default()
    assert preset.name == "ultra-long"
    assert preset.policy_type == "Turbo3ExactKApproxVPolicy"
    assert preset.params["enable_compute_skip"] is True


def test_collect_policy_insights_reports_layer_strategy_counts() -> None:
    class _Config:
        layer_types = ["full_attention", "sliding_attention", "full_attention", "mla_attention"]
        num_hidden_layers = 4

    policy = Turbo3ExactKApproxVPolicy(
        v_quantize_only_first_n_full_attention_layers=1,
        quantized_kv_start=1024,
        enable_sparse_v=True,
        sparse_v_only_first_n_full_attention_layers=1,
        enable_compute_skip=True,
        compute_skip_only_first_n_full_attention_layers=1,
    )
    insights = _collect_policy_insights(_Config(), policy, chosen_n=1)

    assert insights["quantized_kv_start"] == 1024
    assert insights["counts"]["full_attention"] == 2
    assert insights["counts"]["sliding_window"] == 1
    assert insights["counts"]["mla"] == 1
    assert insights["counts"]["turbo_k"] == 2
    assert insights["counts"]["turbo_v"] == 1
    assert insights["counts"]["sparse_v"] == 1
    assert insights["counts"]["compute_skip"] == 1


def test_collect_policy_insights_infers_deepseek_v4_compressed_and_hash_layers() -> None:
    class _Config:
        model_type = "deepseek_v4"
        architectures = ["DeepseekV4ForCausalLM"]
        num_hidden_layers = 6
        compress_ratios = [0, 0, 4, 128, 4, 0]
        num_hash_layers = 2
        sliding_window = 128

    preset = KVPolicyPreset.deepseek_v4_default()
    policy = build_policy(preset)
    insights = _collect_policy_insights(_Config(), policy, chosen_n=1)

    assert insights["counts"]["hash_attention"] == 2
    assert insights["counts"]["compressed_attention"] == 3
    assert insights["counts"]["sliding_window"] == 1
    assert insights["counts"]["full_attention"] == 0
    assert insights["counts"]["turbo_k"] == 0
    assert insights["counts"]["turbo_v"] == 0


def test_runtime_patch_uses_quantized_tensors_in_default_attention(monkeypatch) -> None:
    recorded = {}

    def fake_original(query, key, value, *args, **kwargs):
        recorded["key"] = key.clone()
        recorded["value"] = value.clone()
        return torch.zeros_like(query)

    def fake_quantize(tensor, format_type="turbo3", block_size=32, **kwargs):
        return tensor + 7.0

    monkeypatch.setattr(F, "scaled_dot_product_attention", fake_original)
    monkeypatch.setattr("vitriol.patches.kv_runtime_patches.turbo_quantize", fake_quantize)

    patcher = KVRuntimePatcher(KVRuntimePatchConfig(enable_turbo_quant=True, quantized_kv_start=0))
    patcher.apply()

    query = torch.zeros(1, 1, 1, 4)
    key = torch.ones(1, 1, 2, 4)
    value = torch.full((1, 1, 2, 4), 2.0)

    F.scaled_dot_product_attention(query, key, value)
    patcher.restore()

    assert torch.allclose(recorded["key"], key + 7.0)
    assert torch.allclose(recorded["value"], value + 7.0)


def test_runtime_patch_preprocess_cache_is_bounded_and_evicts(monkeypatch) -> None:
    """The preprocess cache must be bounded to avoid unbounded growth in long-running processes."""

    def fake_original(query, key, value, *args, **kwargs):
        return torch.zeros_like(query)

    call_counter = {"n": 0}

    def fake_quantize(tensor, format_type="turbo3", block_size=32, **kwargs):
        call_counter["n"] += 1
        return tensor + 1.0

    monkeypatch.setattr(F, "scaled_dot_product_attention", fake_original)
    monkeypatch.setattr("vitriol.patches.kv_runtime_patches.turbo_quantize", fake_quantize)

    patcher = KVRuntimePatcher(
        KVRuntimePatchConfig(
            enable_turbo_quant=True,
            quantized_kv_start=0,
            preprocess_cache_max_entries=2,
        )
    )
    patcher.apply()

    for _ in range(5):
        query = torch.zeros(1, 1, 1, 4)
        key = torch.ones(1, 1, 2, 4)
        value = torch.full((1, 1, 2, 4), 2.0)
        F.scaled_dot_product_attention(query, key, value)

    stats = patcher.stats()
    patcher.restore()

    assert stats["preprocess_cache_entries"] <= 2
    assert stats["preprocess_cache_evictions"] >= 1


def test_qwen35_patch_uses_quantized_tensors_in_default_attention(monkeypatch) -> None:
    import transformers.modeling_utils as mu
    m = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

    recorded = {}

    def fake_sdpa(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        return torch.zeros_like(query), None

    def fake_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        recorded["key"] = key.clone()
        recorded["value"] = value.clone()
        return torch.zeros_like(query), None

    def fake_quantize(tensor, format_type="turbo3", block_size=32, **kwargs):
        return tensor + 5.0

    monkeypatch.setattr(mu, "sdpa_attention_forward", fake_sdpa)
    monkeypatch.setitem(mu.ALL_ATTENTION_FUNCTIONS, "sdpa", fake_sdpa)
    monkeypatch.setattr(m, "eager_attention_forward", fake_eager)
    monkeypatch.setattr(m, "repeat_kv", lambda tensor, groups: tensor)
    monkeypatch.setattr("vitriol.patches.qwen35_attention_patches.turbo_quantize", fake_quantize)

    patcher = Qwen35AttentionPatcher(Qwen35AttentionPatchConfig(enable_turbo_quant=True, quantized_kv_start=0))
    patcher.apply()

    class Qwen3_5Attention:
        num_key_value_groups = 1

    module = Qwen3_5Attention()
    query = torch.zeros(1, 1, 1, 4)
    key = torch.ones(1, 1, 2, 4)
    value = torch.full((1, 1, 2, 4), 3.0)

    m.eager_attention_forward(module, query, key, value, None, 1.0, dropout=0.0)
    patcher.restore()

    assert torch.allclose(recorded["key"], key + 5.0)
    assert torch.allclose(recorded["value"], value + 5.0)


def test_runtime_patch_passes_residual_qjl_controls_to_quantizer(monkeypatch) -> None:
    captured = {}

    def fake_original(query, key, value, *args, **kwargs):
        return torch.zeros_like(query)

    def fake_quantize(tensor, format_type="turbo3", block_size=32, **kwargs):
        captured.update(kwargs)
        return tensor

    monkeypatch.setattr(F, "scaled_dot_product_attention", fake_original)
    monkeypatch.setattr("vitriol.patches.kv_runtime_patches.turbo_quantize", fake_quantize)

    patcher = KVRuntimePatcher(
        KVRuntimePatchConfig(
            enable_turbo_quant=True,
            quantized_kv_start=0,
            enable_turbo_residual_qjl=False,
            turbo_residual_strength=0.25,
        )
    )
    patcher.apply()
    query = torch.zeros(1, 1, 1, 4)
    key = torch.ones(1, 1, 2, 4)
    value = torch.ones(1, 1, 2, 4)
    F.scaled_dot_product_attention(query, key, value)
    patcher.restore()

    assert captured["use_residual_qjl"] is False
    assert captured["residual_strength"] == 0.25
