"""Tests for NAS modules: search_space, searcher, controller, evaluator"""

import pytest
from unittest.mock import patch, MagicMock

from vitriol.nas.search_space import ArchitectureGene, LLMSearchSpace
from vitriol.nas.searcher import Searcher, RandomSearcher, EvolutionarySearcher
from vitriol.nas.controller import NASController
from vitriol.nas.evaluator import ZeroCostProxy, ParamCountProxy


# ─────────────────────────────────────────────────────────────────────────────
# search_space tests
# ─────────────────────────────────────────────────────────────────────────────

class TestArchitectureGene:
    def test_basic_creation(self):
        gene = ArchitectureGene(
            n_layers=12,
            hidden_size=768,
            n_heads=12,
            attention_type="MHA",
            ffn_type="Standard",
            activation="gelu",
            norm_type="LayerNorm",
        )
        assert gene.n_layers == 12
        assert gene.hidden_size == 768
        assert gene.intermediate_size == 768 * 4  # Standard multiplier

    def test_swiglu_intermediate_size(self):
        gene = ArchitectureGene(
            n_layers=2,
            hidden_size=768,
            n_heads=12,
            attention_type="MHA",
            ffn_type="SwiGLU",
            activation="silu",
            norm_type="RMSNorm",
        )
        expected = int(768 * 8 / 3)
        assert gene.intermediate_size == expected

    def test_mqa_kv_heads(self):
        gene = ArchitectureGene(
            n_layers=2,
            hidden_size=256,
            n_heads=8,
            attention_type="MQA",
            ffn_type="Standard",
            activation="gelu",
            norm_type="LayerNorm",
        )
        assert gene.num_kv_heads == 1

    def test_gqa_kv_heads(self):
        gene = ArchitectureGene(
            n_layers=2,
            hidden_size=256,
            n_heads=8,
            attention_type="GQA",
            ffn_type="Standard",
            activation="gelu",
            norm_type="LayerNorm",
        )
        assert gene.num_kv_heads == 2  # 8 // 4

    def test_hidden_size_divisible(self):
        gene = ArchitectureGene(
            n_layers=2,
            hidden_size=100,  # Not divisible by 6
            n_heads=6,
            attention_type="MHA",
            ffn_type="Standard",
            activation="gelu",
            norm_type="LayerNorm",
        )
        assert gene.hidden_size % gene.n_heads == 0

    def test_to_dict(self):
        gene = ArchitectureGene(
            n_layers=2, hidden_size=128, n_heads=4,
            attention_type="MHA", ffn_type="Standard",
            activation="gelu", norm_type="LayerNorm",
        )
        d = gene.to_dict()
        assert d["n_layers"] == 2
        assert d["hidden_size"] == 128
        assert "intermediate_size" in d

    def test_from_dict(self):
        gene = ArchitectureGene(
            n_layers=2, hidden_size=128, n_heads=4,
            attention_type="MHA", ffn_type="Standard",
            activation="gelu", norm_type="LayerNorm",
        )
        d = gene.to_dict()
        restored = ArchitectureGene.from_dict(d)
        assert restored.n_layers == 2
        assert restored.hidden_size == 128

    def test_from_config_mha(self):
        config = {
            "hidden_size": 128,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "intermediate_size": 512,
        }
        gene = ArchitectureGene.from_config(config)
        assert gene.attention_type == "MHA"

    def test_from_config_mqa(self):
        config = {
            "hidden_size": 128,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "intermediate_size": 512,
        }
        gene = ArchitectureGene.from_config(config)
        assert gene.attention_type == "MQA"

    def test_from_config_gqa(self):
        config = {
            "hidden_size": 128,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "intermediate_size": 512,
        }
        gene = ArchitectureGene.from_config(config)
        assert gene.attention_type == "GQA"


class TestLLMSearchSpace:
    def test_init(self):
        space = LLMSearchSpace()
        assert space is not None

    def test_sample(self):
        space = LLMSearchSpace()
        gene = space.sample()
        assert isinstance(gene, ArchitectureGene)
        assert gene.n_layers > 0
        assert gene.hidden_size > 0

    def test_sample_variations(self):
        space = LLMSearchSpace()
        genes = [space.sample() for _ in range(10)]
        # Should get some variation
        layers = [g.n_layers for g in genes]
        assert len(set(layers)) > 1 or len(set(g.hidden_size for g in genes)) > 1


# ─────────────────────────────────────────────────────────────────────────────
# searcher tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSearcher:
    def test_abstract_search(self):
        space = LLMSearchSpace()
        evaluator = MagicMock()
        evaluator.evaluate.return_value = {"score": 0.5}
        searcher = Searcher(space, evaluator)
        with pytest.raises(NotImplementedError):
            searcher.search(1)

    def test_evaluate(self):
        space = LLMSearchSpace()
        evaluator = MagicMock()
        evaluator.evaluate.return_value = {"score": 0.75}
        searcher = Searcher(space, evaluator)
        gene = space.sample()
        score = searcher._evaluate(gene)
        assert score == 0.75
        assert len(searcher.history) == 1

    def test_evaluate_with_checkpoint(self):
        space = LLMSearchSpace()
        evaluator = MagicMock()
        evaluator.evaluate.return_value = {"score": 0.5}
        cb = MagicMock()
        searcher = Searcher(space, evaluator, save_checkpoint_callback=cb)
        gene = space.sample()
        searcher._evaluate(gene)
        assert cb.called


class TestRandomSearcher:
    def test_search(self):
        space = LLMSearchSpace()
        evaluator = MagicMock()
        evaluator.evaluate.return_value = {"score": 0.5}
        searcher = RandomSearcher(space, evaluator)
        gene = searcher.search(3)
        assert gene is not None
        assert len(searcher.history) == 3

    def test_search_finds_best(self):
        space = LLMSearchSpace()
        evaluator = MagicMock()
        scores = [0.1, 0.5, 0.3]
        evaluator.evaluate.side_effect = [{"score": s} for s in scores]
        searcher = RandomSearcher(space, evaluator)
        gene = searcher.search(3)
        assert len(searcher.history) == 3


class TestEvolutionarySearcher:
    def test_init(self):
        space = LLMSearchSpace()
        evaluator = MagicMock()
        searcher = EvolutionarySearcher(space, evaluator, population_size=5)
        assert searcher.population_size == 5
        assert searcher.population == []

    def test_search(self):
        space = LLMSearchSpace()
        evaluator = MagicMock()
        evaluator.evaluate.return_value = {"score": 0.5}
        searcher = EvolutionarySearcher(space, evaluator, population_size=10)
        gene = searcher.search(2)
        assert gene is not None


# ─────────────────────────────────────────────────────────────────────────────
# controller tests
# ─────────────────────────────────────────────────────────────────────────────

class TestNASController:
    @patch("vitriol.nas.controller.LLMSearchSpace")
    @patch("vitriol.nas.controller.HybridEvaluator")
    def test_init(self, mock_eval, mock_space):
        ctrl = NASController(output_dir="/tmp/nas_test")
        assert ctrl.output_dir.name == "nas_test"

    @patch("vitriol.nas.controller.LLMSearchSpace")
    @patch("vitriol.nas.controller.HybridEvaluator")
    @patch("vitriol.nas.controller.RandomSearcher")
    def test_run_random(self, mock_searcher_class, mock_eval, mock_space):
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = MagicMock(to_config=lambda: {})
        mock_searcher_class.return_value = mock_searcher

        ctrl = NASController()
        result = ctrl.run(algorithm="random", n_iterations=2)
        assert "best_gene" in result
        assert "history" in result

    @patch("vitriol.nas.controller.LLMSearchSpace")
    @patch("vitriol.nas.controller.HybridEvaluator")
    def test_run_invalid_algorithm(self, mock_eval, mock_space):
        ctrl = NASController()
        with pytest.raises(ValueError, match="Unknown algorithm"):
            ctrl.run(algorithm="invalid", n_iterations=1)


# ─────────────────────────────────────────────────────────────────────────────
# evaluator tests
# ─────────────────────────────────────────────────────────────────────────────

class TestZeroCostProxy:
    def test_abstract(self):
        proxy = ZeroCostProxy()
        with pytest.raises(NotImplementedError):
            proxy.score(MagicMock())


class TestParamCountProxy:
    def test_score(self):
        proxy = ParamCountProxy()
        gene = ArchitectureGene(
            n_layers=2, hidden_size=128, n_heads=4,
            attention_type="MHA", ffn_type="Standard",
            activation="gelu", norm_type="LayerNorm",
            vocab_size=1000,
        )
        score = proxy.score(gene)
        assert score > 0
        # Should be roughly: emb + layers + output
        # emb = 1000 * 128 = 128k
        # layer = 4*128*128 + 3*128*512 + 2*128 = ~262k
        # total ~ 128k + 2*262k + 128k = ~780k
        assert 500000 < score < 2000000

    def test_score_varies_with_size(self):
        proxy = ParamCountProxy()
        small = ArchitectureGene(
            n_layers=2, hidden_size=64, n_heads=4,
            attention_type="MHA", ffn_type="Standard",
            activation="gelu", norm_type="LayerNorm",
            vocab_size=1000,
        )
        large = ArchitectureGene(
            n_layers=4, hidden_size=256, n_heads=8,
            attention_type="MHA", ffn_type="Standard",
            activation="gelu", norm_type="LayerNorm",
            vocab_size=1000,
        )
        assert proxy.score(small) < proxy.score(large)

