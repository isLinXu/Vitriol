"""Regression tests for NAS evaluator score and local dataset edge cases."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vitriol.nas.evaluator import (
    HybridEvaluator,
    RankMeProxy,
    VitriolExpressivityProxy,
)
from vitriol.nas.search_space import ArchitectureGene


@pytest.fixture
def gene():
    return ArchitectureGene(
        n_layers=1,
        hidden_size=4,
        n_heads=1,
        attention_type="MHA",
        ffn_type="Standard",
        activation="gelu",
        norm_type="LayerNorm",
        vocab_size=8,
    )


class _ConstantRepresentationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward(self, inputs, **kwargs):
        hidden = torch.ones(inputs.size(0), inputs.size(1), 4) * self.weight
        logits = torch.zeros(inputs.size(0), inputs.size(1), 8)
        return SimpleNamespace(hidden_states=(hidden,), logits=logits)


def test_rankme_returns_finite_score_for_collapsed_representations(gene):
    model = _ConstantRepresentationModel()
    inputs = torch.ones(2, 3, dtype=torch.long)

    score = RankMeProxy().score(gene, model, inputs)

    assert score == 0.0


def test_expressivity_handles_single_token_sequences(gene):
    model = _ConstantRepresentationModel()
    inputs = torch.ones(2, 1, dtype=torch.long)

    score = VitriolExpressivityProxy().score(gene, model, inputs)

    assert score == 0.0


class _FakeDataset:
    def __init__(self, samples):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def shuffle(self, seed):
        return self

    def select(self, indices):
        self.samples = [self.samples[index] for index in indices]
        return self

    def __iter__(self):
        return iter(self.samples)


@pytest.mark.parametrize(
    ("filename", "load_type"),
    [("samples.jsonl", "json"), ("samples.json", "json"), ("samples.parquet", "parquet")],
)
def test_load_local_dataset_file_path(tmp_path, monkeypatch, filename, load_type):
    data_file = tmp_path / filename
    data_file.write_text("placeholder")
    load_dataset = Mock(return_value=_FakeDataset([{"text": "hello"}]))
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    evaluator = HybridEvaluator(str(tmp_path / "out"))

    samples = evaluator.load_dataset(str(data_file), split="validation")

    assert samples == [{"text": "hello"}]
    load_dataset.assert_called_once_with(
        load_type,
        data_files={"validation": [str(data_file)]},
        split="validation",
    )


def test_load_local_directory_includes_jsonl_files(tmp_path, monkeypatch):
    data_file = tmp_path / "nested" / "samples.jsonl"
    data_file.parent.mkdir()
    data_file.write_text("placeholder")
    load_dataset = Mock(return_value=_FakeDataset([{"text": "hello"}]))
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    evaluator = HybridEvaluator(str(tmp_path / "out"))

    samples = evaluator.load_dataset(str(tmp_path), n_samples=1)

    assert samples == [{"text": "hello"}]
    load_dataset.assert_called_once_with(
        "json",
        data_files={"train": [str(data_file)]},
        split="train",
    )
