import torch

from vitriol.kv.cache_store import KVCacheStore, KVCacheStoreConfig
from vitriol.kv.codec import (
    pack_blockwise_tensor,
    unpack_blockwise_tensor,
    walsh_hadamard_rotate,
)


def test_turboquant_packed_storage_shrinks_vs_bf16():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 64, 32, dtype=torch.float32)

    packed3 = pack_blockwise_tensor(x, levels=8, block_size=32, bit_width=3)
    packed4 = pack_blockwise_tensor(x, levels=16, block_size=32, bit_width=4)
    bf16_bytes = x.numel() * 2

    assert packed3.storage_nbytes() < bf16_bytes
    assert packed4.storage_nbytes() < bf16_bytes
    assert packed3.storage_nbytes() <= packed4.storage_nbytes()


def test_turboquant_error_improves_with_higher_bit_width():
    torch.manual_seed(1)
    x = torch.randn(1, 2, 64, 32, dtype=torch.float32)

    packed2 = pack_blockwise_tensor(x, levels=4, block_size=32, bit_width=2)
    packed3 = pack_blockwise_tensor(x, levels=8, block_size=32, bit_width=3)
    packed4 = pack_blockwise_tensor(x, levels=16, block_size=32, bit_width=4)

    err2 = (x - unpack_blockwise_tensor(packed2)).abs().mean().item()
    err3 = (x - unpack_blockwise_tensor(packed3)).abs().mean().item()
    err4 = (x - unpack_blockwise_tensor(packed4)).abs().mean().item()

    assert err4 <= err3
    assert err3 <= err2


def test_turboquant_current_path_still_has_inner_product_bias():
    torch.manual_seed(2)
    query = torch.randn(1, 2, 16, 32, dtype=torch.float32)
    key = torch.randn(1, 2, 16, 32, dtype=torch.float32)

    packed = pack_blockwise_tensor(key, levels=8, block_size=32, bit_width=3)
    restored = unpack_blockwise_tensor(packed)

    original_ip = (query * key).sum(dim=-1)
    restored_ip = (query * restored).sum(dim=-1)
    mean_bias = (restored_ip - original_ip).mean().abs().item()

    assert mean_bias > 0.0


def test_rotation_does_not_worsen_high_kurtosis_quantization():
    torch.manual_seed(3)
    x = torch.randn(1, 2, 32, 32, dtype=torch.float32) * 0.01
    x[..., 0] = 10.0

    plain = unpack_blockwise_tensor(
        pack_blockwise_tensor(x, levels=8, block_size=32, bit_width=3)
    )
    rotated_x = walsh_hadamard_rotate(x)
    rotated = unpack_blockwise_tensor(
        pack_blockwise_tensor(rotated_x, levels=8, block_size=32, bit_width=3)
    )

    plain_err = (x - plain).abs().mean().item()
    rotated_err = (rotated_x - rotated).abs().mean().item()

    assert rotated_err <= plain_err + 1e-6


def test_real_packed_bytes_imply_higher_token_capacity_than_bf16():
    torch.manual_seed(4)
    cfg = KVCacheStoreConfig(enable_turbo_quant=True, turbo_bits=3.5, quantized_kv_start=0)
    store = KVCacheStore(cfg)

    key = torch.randn(1, 2, 64, 32)
    value = torch.randn(1, 2, 64, 32)
    store.set_prefill(key, value)

    packed_bytes = store.estimated_kv_bytes()
    bf16_bytes = int((key.numel() + value.numel()) * 2)
    capacity_gain = bf16_bytes / packed_bytes

    assert packed_bytes < bf16_bytes
    assert capacity_gain > 1.0
