"""Fingerprint real model dtypes and small normalization/parameter shapes."""

import json

import pytest
import torch
from torch import nn

from vitriol.utils.fingerprint import ArchitectureHasher, FingerprintEngine, WeightsHasher


@pytest.mark.parametrize("affine", [False, True])
def test_batchnorm_architecture_hash_is_supported(affine):
    model = nn.BatchNorm2d(8, affine=affine)
    hasher = ArchitectureHasher()
    assert hasher.hash(model) == hasher.hash(model)
    assert hasher.hash(model) != hasher.hash(nn.BatchNorm2d(8, affine=not affine))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_fingerprint_and_comparison_are_supported(dtype):
    model = nn.Linear(4, 4).to(dtype)
    other = nn.Linear(4, 4).to(dtype)
    engine = FingerprintEngine()
    assert engine.fingerprint(model).weights_hash
    json.dumps(engine.weights_hasher._compute_weight_stats("weight", model.weight), allow_nan=False)
    result = engine.compare_models(model, other)
    assert 0 <= result["weights_similarity"] <= 1


@pytest.mark.parametrize("shape", [(0,), (1, 1), (1, 4), (4, 1)])
def test_empty_and_singleton_weight_statistics_are_finite(shape):
    stats = WeightsHasher()._compute_weight_stats("weight", torch.zeros(shape))
    json.dumps(stats, allow_nan=False)
