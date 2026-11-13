from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from vitriol.bench.ppl_evaluator import PPLConfig, PPLEvaluator


class NextTokenModel:
    """Token n deterministically favors n + 1, exposing shifted scoring."""

    config = SimpleNamespace(eos_token_id=None)

    def __init__(self):
        self.calls = []

    def eval(self):
        pass

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        self.calls.append((input_ids.clone(), attention_mask))
        logits = torch.zeros(*input_ids.shape, 8)
        logits.scatter_(-1, ((input_ids + 1) % 8).unsqueeze(-1), 4.0)
        return SimpleNamespace(logits=logits, past_key_values=())


def _evaluator():
    evaluator = PPLEvaluator(PPLConfig(device="cpu", dtype="float32"))
    evaluator._model = NextTokenModel()
    evaluator._tokenizer = SimpleNamespace(eos_token_id=None)
    return evaluator


def test_generation_nll_scores_each_token_before_feeding_it_back():
    evaluator = _evaluator()
    ids = torch.tensor([[0, 1]])
    ppl, generated, nlls = evaluator._compute_ppl_for_generation(ids, torch.ones_like(ids), 3)
    expected_nll = F.cross_entropy(torch.tensor([[4.0] + [0.0] * 7]), torch.tensor([0])).item()
    assert generated == [2, 3, 4]
    assert ppl == pytest.approx(torch.exp(torch.tensor(expected_nll)).item())
    assert sum(row["tokens"] for row in nlls) == 4
    assert [mask.shape[-1] for _, mask in evaluator._model.calls] == [2, 3, 4]


def test_ppl_excludes_padding_and_does_not_decode_after_eos():
    evaluator = _evaluator()
    evaluator._tokenizer.eos_token_id = 2
    ids = torch.tensor([[7, 0, 1]])
    ppl, generated, nlls = evaluator._compute_ppl_for_generation(ids, torch.tensor([[0, 1, 1]]), 10)
    assert generated == [2]
    assert sum(row["tokens"] for row in nlls) == 2
    assert len(evaluator._model.calls) == 1
    assert ppl < 2.0


def test_token_agreement_does_not_fabricate_distribution_metrics():
    evaluator = _evaluator()
    assert evaluator._compute_overall_metrics([], [], "", "") == (None, [])
    assert evaluator._compute_overall_metrics([1], [2], "a", "b") == (None, [])


@pytest.mark.parametrize("stats", [{}, {"estimated_kv_bytes": 0}, RuntimeError("no telemetry")])
def test_missing_cache_stats_do_not_claim_memory_savings(monkeypatch, stats):
    evaluator = _evaluator()
    monkeypatch.setattr(evaluator, "_estimate_kv_memory", lambda: 100)
    backend = Mock()
    if isinstance(stats, Exception):
        backend.stats.side_effect = stats
    else:
        backend.stats.return_value = stats
    assert evaluator._estimate_kv_memory_tuned(backend) == 100


def test_tuned_generation_failure_restores_both_patchers(monkeypatch):
    evaluator = _evaluator()
    evaluator._tokenizer.decode = lambda *_args, **_kwargs: "text"
    monkeypatch.setattr(evaluator, "_tokenize", lambda _: (torch.tensor([[0, 1]]), None))
    monkeypatch.setattr(evaluator, "_estimate_kv_memory", lambda: 100)
    generation = Mock(side_effect=[(1.0, [2], []), RuntimeError("decode failed")])
    monkeypatch.setattr(evaluator, "_compute_ppl_for_generation", generation)
    cache, attention = Mock(), Mock()
    from vitriol.bench import runner
    monkeypatch.setattr(runner, "_select_preset", lambda *_: object())
    monkeypatch.setattr(runner, "_preset_to_kv_cfg", lambda *_: (None, 0, None))
    monkeypatch.setattr(runner, "_apply_vitriol_universal", lambda *a, **kw: (None, cache, attention))
    with pytest.raises(RuntimeError, match="decode failed"):
        evaluator.evaluate(prompt_override="prompt")
    cache.restore.assert_called_once()
    attention.restore.assert_called_once()
