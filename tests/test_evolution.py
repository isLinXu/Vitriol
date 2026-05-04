"""Tests for evolution modules: tree_builder, compare, simulator, recommender"""

from unittest.mock import patch, MagicMock

from vitriol.evolution.tree_builder import (
    ArchNode, ArchInnovation, ArchitectureMetrics,
    EvolutionTree, FALLBACK_PARAMS
)
from vitriol.evolution.compare import (
    ComparisonResult, ArchComparator,
    ATTENTION_TYPES, FFN_TYPES, POSITION_ENCODING
)
from vitriol.evolution.simulator import (
    SimulationResult, ArchSimulator,
    BYTES_FP32, BYTES_FP16, BYTES_BF16
)
from vitriol.evolution.recommender import (
    UseCase, RecommendationCriteria, ArchitectureRecommendation,
    ArchitectureRecommender
)


# ─────────────────────────────────────────────────────────────────────────────
# tree_builder tests
# ─────────────────────────────────────────────────────────────────────────────

class TestArchNode:
    def test_model_name(self):
        node = ArchNode(model_id="org/model-name", config={})
        assert node.model_name == "model-name"

    def test_model_name_no_slash(self):
        node = ArchNode(model_id="model-name", config={})
        assert node.model_name == "model-name"

    def test_family_qwen(self):
        node = ArchNode(model_id="qwen/Qwen2-7B", config={})
        assert node.family == "Qwen"

    def test_family_llama(self):
        node = ArchNode(model_id="meta-llama/Llama-2-7b", config={})
        assert node.family == "LLaMA"

    def test_family_deepseek(self):
        node = ArchNode(model_id="deepseek-ai/DeepSeek-V3", config={})
        assert node.family == "DeepSeek"

    def test_family_glm(self):
        node = ArchNode(model_id="THUDM/glm-4-9b", config={})
        assert node.family == "GLM"

    def test_family_fallback(self):
        node = ArchNode(model_id="unknown/model", config={})
        assert node.family == "unknown"

    def test_get_key_params(self):
        node = ArchNode(model_id="test", config={
            "hidden_size": 128,
            "num_hidden_layers": 12,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "intermediate_size": 512,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "model_type": "llama",
            "num_local_experts": 8,
        })
        params = node.get_key_params()
        assert params["hidden_size"] == 128
        assert params["is_moe"] is True
        assert params["num_experts"] == 8

    def test_arch_innovation(self):
        innov = ArchInnovation(
            name="GQA",
            description="Grouped Query Attention",
            introduced_in="Llama-2-70B",
            year=2023
        )
        assert innov.name == "GQA"
        assert innov.year == 2023


class TestArchitectureMetrics:
    def test_to_dict(self):
        m = ArchitectureMetrics(
            total_params=1e9,
            trainable_params=1e9,
            flops_per_token=1e12,
            vram_estimate_gb=16.0,
            inference_latency_ms=50.0,
            memory_bandwidth_gbs=100.0,
        )
        d = m.to_dict()
        assert d["total_params"] == 1e9
        assert d["vram_estimate_gb"] == 16.0


class TestEvolutionTree:
    def test_init(self):
        tree = EvolutionTree()
        assert tree.nodes == {}
        assert len(tree.families) > 0  # DEFAULT_FAMILIES copied in __init__

    def test_add_model(self):
        tree = EvolutionTree()
        node = tree.add_model("test/model", config={"hidden_size": 128})
        assert "test/model" in tree.nodes
        assert tree.nodes["test/model"].config["hidden_size"] == 128

    def test_add_model_with_parent(self):
        tree = EvolutionTree()
        tree.add_model("parent", config={})
        tree.add_model("child", config={}, parent="parent")
        assert "child" in tree.nodes
        assert tree.nodes["child"].parent == "parent"

    def test_fallback_params_exists(self):
        assert isinstance(FALLBACK_PARAMS, dict)
        assert len(FALLBACK_PARAMS) > 0

    def test_load_builtin_families(self):
        tree = EvolutionTree()
        tree.load_builtin_families()
        assert len(tree.families) > 0

    def test_build(self):
        tree = EvolutionTree()
        tree.add_model("test", config={"hidden_size": 128})
        tree.build()
        assert "test" in tree.nodes


# ─────────────────────────────────────────────────────────────────────────────
# compare tests
# ─────────────────────────────────────────────────────────────────────────────

class TestComparisonResult:
    def test_to_dict(self):
        r = ComparisonResult(
            model1_id="m1",
            model2_id="m2",
            similarity_score=75.0,
            summary="Test summary",
        )
        d = r.to_dict()
        assert d["model1"] == "m1"
        assert d["model2"] == "m2"
        assert d["similarity_score"] == 75.0


class TestArchComparator:
    def test_init(self):
        c = ArchComparator()
        assert c is not None

    def test_compare_identical(self):
        c = ArchComparator()
        node1 = ArchNode(model_id="test1", config={"hidden_size": 128, "num_hidden_layers": 12})
        node2 = ArchNode(model_id="test2", config={"hidden_size": 128, "num_hidden_layers": 12})
        result = c.compare(node1, node2)
        assert result.model1_id == "test1"
        assert result.model2_id == "test2"
        assert result.similarity_score >= 0
        assert result.similarity_score <= 100

    def test_compare_different(self):
        c = ArchComparator()
        node1 = ArchNode(model_id="test1", config={"hidden_size": 128, "num_hidden_layers": 12})
        node2 = ArchNode(model_id="test2", config={"hidden_size": 4096, "num_hidden_layers": 80})
        result = c.compare(node1, node2)
        assert result.similarity_score >= 0
        assert result.similarity_score <= 100

    def test_feature_constants(self):
        assert "multi_head" in ATTENTION_TYPES
        assert "moe" in FFN_TYPES
        assert "rope" in POSITION_ENCODING


# ─────────────────────────────────────────────────────────────────────────────
# simulator tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulationResult:
    def test_to_dict(self):
        r = SimulationResult(
            model_id="test",
            config={},
            total_params=1e9,
            trainable_params=1e9,
            active_params_per_token=1e9,
            flops_per_token=1e12,
            flops_per_second=1e15,
            vram_full_model=16.0,
            vram_inference=8.0,
            vram_training=32.0,
            kv_cache_estimate=2.0,
            inference_latency_ms=50.0,
            tokens_per_second=20.0,
            memory_bandwidth_gbs=100.0,
            params_per_vram=1e9/16,
            flops_per_param=1e3,
        )
        d = r.to_dict()
        assert d["model_id"] == "test"
        assert d["total_params"] == 1e9
        assert isinstance(d["vram_full_model"], float)


class TestArchSimulator:
    def test_init(self):
        sim = ArchSimulator()
        assert sim is not None

    def test_simulate_basic(self):
        sim = ArchSimulator()
        config = {
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 512,
            "vocab_size": 1000,
            "max_position_embeddings": 1024,
        }
        result = sim.simulate("test/model", config)
        assert result.model_id == "test/model"
        assert result.total_params > 0
        assert result.vram_full_model > 0
        assert result.inference_latency_ms > 0

    def test_simulate_moe(self):
        sim = ArchSimulator()
        config = {
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "intermediate_size": 1024,
            "vocab_size": 1000,
            "max_position_embeddings": 1024,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
        }
        result = sim.simulate("test/moe", config)
        assert result.active_params_per_token < result.total_params

    def test_memory_constants(self):
        assert BYTES_FP32 == 4
        assert BYTES_FP16 == 2
        assert BYTES_BF16 == 2


# ─────────────────────────────────────────────────────────────────────────────
# recommender tests
# ─────────────────────────────────────────────────────────────────────────────

class TestUseCase:
    def test_values(self):
        assert UseCase.CHAT == "chat"
        assert UseCase.CODE == "code"
        assert UseCase.LONG_CONTEXT == "long_context"


class TestRecommendationCriteria:
    def test_defaults(self):
        c = RecommendationCriteria()
        assert c.max_params is None
        assert c.use_case == UseCase.GENERAL
        assert c.prefer_moe is False

    def test_custom(self):
        c = RecommendationCriteria(
            max_params=7e9,
            max_vram=16.0,
            use_case=UseCase.CODE,
            prefer_moe=True,
        )
        assert c.max_params == 7e9
        assert c.use_case == UseCase.CODE
        assert c.prefer_moe is True


class TestArchitectureRecommendation:
    def test_creation(self):
        r = ArchitectureRecommendation(
            model_id="test/model",
            family="Test",
            params_b=7.0,
            vram_gb=16.0,
            score=95.0,
            match_reasons=["Good fit"],
        )
        assert r.model_id == "test/model"
        assert r.score == 95.0


class TestArchitectureRecommender:
    @patch("vitriol.evolution.recommender.EvolutionTree")
    def test_init(self, mock_tree_class):
        mock_tree = MagicMock()
        mock_tree.nodes = {}
        mock_tree_class.return_value = mock_tree
        rec = ArchitectureRecommender()
        assert rec is not None

    @patch("vitriol.evolution.recommender.EvolutionTree")
    def test_recommend_empty_tree(self, mock_tree_class):
        mock_tree = MagicMock()
        mock_tree.nodes = {}
        mock_tree_class.return_value = mock_tree
        rec = ArchitectureRecommender()
        results = rec.recommend()
        assert isinstance(results, list)

    @patch("vitriol.evolution.recommender.EvolutionTree")
    def test_vram_coefficient(self, mock_tree_class):
        mock_tree = MagicMock()
        mock_tree.nodes = {}
        mock_tree_class.return_value = mock_tree
        rec = ArchitectureRecommender()
        assert rec.VRAM_COEFFICIENT == 2.0

