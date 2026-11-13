"""Sampling must stay bounded for views and honor the public sample budget."""

import torch
from safetensors.torch import save_file

from vitriol.viz.weight_inspector import _compute_tensor_stats, inspect_weights


def test_sampling_does_not_flatten_a_large_noncontiguous_view(monkeypatch):
    tensor = torch.arange(128, dtype=torch.float32).reshape(16, 8).t()
    original_reshape = torch.Tensor.reshape

    def bounded_reshape(self, *shape):
        if self.numel() > 16 and not self.is_contiguous():
            raise AssertionError("Sampling materialized the full noncontiguous view")
        return original_reshape(self, *shape)

    monkeypatch.setattr(torch.Tensor, "reshape", bounded_reshape)
    stats = _compute_tensor_stats(tensor, seed=42, sample_size=16)
    generator = torch.Generator().manual_seed(42)
    indices = torch.randint(0, tensor.numel(), (16,), generator=generator)
    expected = tensor[indices // tensor.shape[1], indices % tensor.shape[1]]

    assert "error" not in stats
    assert stats["mean"] == expected.mean().item()
    assert stats["std"] == expected.std(unbiased=False).item()
    assert stats["sample_size"] == 16
    assert stats["sampled"] is True


def test_sampling_partially_broadcast_tensor_stays_bounded(monkeypatch):
    tensor = torch.tensor([0.0, 1.0, 2.0, 3.0]).expand(1_000_000, -1)
    original_reshape = torch.Tensor.reshape

    def bounded_reshape(self, *shape):
        if self.numel() > 16 and not self.is_contiguous():
            raise AssertionError("Expanded tensor was materialized before sampling")
        return original_reshape(self, *shape)

    monkeypatch.setattr(torch.Tensor, "reshape", bounded_reshape)
    stats = _compute_tensor_stats(tensor, sample_size=16)
    assert "error" not in stats
    assert stats["sample_size"] == 16
    assert stats["storage_bytes"] == 16
    assert stats["numel"] == 4_000_000


def test_inspect_weights_respects_public_sample_budget(tmp_path):
    tensor = torch.arange(1000, dtype=torch.float32)
    save_file({"weight": tensor}, tmp_path / "model.safetensors")
    result = inspect_weights(str(tmp_path), max_per_layer_sample=7)
    expected = _compute_tensor_stats(tensor, sample_size=7)
    assert result["layers"][0]["mean"] == expected["mean"]
    assert result["layers"][0]["sample_size"] == 7


def test_unsampled_scalar_and_empty_tensor_are_explicit():
    scalar = _compute_tensor_stats(torch.tensor(3.0), sample_size=1)
    empty = _compute_tensor_stats(torch.empty(0), sample_size=1)
    assert scalar["sampled"] is False
    assert scalar["sample_size"] == 1
    assert scalar["l2_norm"] == 3.0
    assert empty["sampled"] is False
    assert empty["sample_size"] == 0
