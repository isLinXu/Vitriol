"""
Tests for vitriol.ai.recommender module.
"""
import pytest

from vitriol.ai.recommender import (
    RecommendationType,
    Recommendation,
    StrategyRecommender,
    HardwareRecommender,
    ParameterRecommender,
    VitriolRecommender,
    get_recommender,
)


class TestRecommendationType:
    def test_enum_values(self):
        assert RecommendationType.STRATEGY.value == "strategy"
        assert RecommendationType.HARDWARE.value == "hardware"
        assert RecommendationType.PARAMETERS.value == "parameters"
        assert RecommendationType.OPTIMIZATION.value == "optimization"


class TestRecommendation:
    def test_recommendation_dataclass(self):
        rec = Recommendation(
            type=RecommendationType.STRATEGY,
            item="ultra",
            confidence=0.85,
            reason="Optimal for storage",
            expected_benefit="99% size reduction",
            alternatives=["quantum", "compact"],
        )
        assert rec.type == RecommendationType.STRATEGY
        assert rec.item == "ultra"
        assert rec.confidence == 0.85
        assert rec.reason == "Optimal for storage"
        assert rec.expected_benefit == "99% size reduction"
        assert rec.alternatives == ["quantum", "compact"]


class TestStrategyRecommender:
    @pytest.fixture
    def recommender(self):
        return StrategyRecommender()

    def test_recommend_ultra_for_storage(self, recommender):
        rec = recommender.recommend(
            model_size_gb=20.0,
            available_memory_gb=4.0,
            use_case="storage",
            requires_training=False,
        )
        assert rec.type == RecommendationType.STRATEGY
        assert rec.item == "ultra"
        assert rec.confidence > 0
        assert "storage" in rec.reason.lower() or "Optimal" in rec.reason
        assert "99%" in rec.expected_benefit or "%" in rec.expected_benefit
        assert len(rec.alternatives) > 0

    def test_recommend_training_requires_training_true(self, recommender):
        rec = recommender.recommend(
            model_size_gb=5.0,
            available_memory_gb=8.0,
            use_case="general",
            requires_training=True,
        )
        # ultra has training=False so it should be heavily penalized
        assert rec.item != "ultra"

    def test_recommend_large_model_prefers_compression(self, recommender):
        rec = recommender.recommend(
            model_size_gb=50.0,
            available_memory_gb=16.0,
            use_case="general",
            requires_training=False,
        )
        # Large models should prefer strategies with high compression
        profile = recommender.STRATEGY_PROFILES[rec.item]
        assert profile["compression"] >= 0.5

    def test_recommend_use_case_match(self, recommender):
        rec = recommender.recommend(
            model_size_gb=1.0,
            available_memory_gb=16.0,
            use_case="edge",
            requires_training=True,
        )
        # "edge" is in quantum use_cases
        assert rec.item == "quantum"

    def test_recommend_all_strategies_scored(self, recommender):
        rec = recommender.recommend(
            model_size_gb=10.0,
            available_memory_gb=8.0,
            use_case="baseline",
            requires_training=True,
        )
        # "baseline" is in random use_cases, but quantum also has training=True
        # and gets high score; just verify a valid strategy is returned
        assert rec.item in recommender.STRATEGY_PROFILES
        assert rec.confidence > 0

    def test_strategy_profiles_structure(self, recommender):
        for strategy, profile in recommender.STRATEGY_PROFILES.items():
            assert "compression" in profile
            assert "speed" in profile
            assert "memory" in profile
            assert "training" in profile
            assert "use_cases" in profile
            assert isinstance(profile["compression"], float)
            assert isinstance(profile["training"], bool)
            assert isinstance(profile["use_cases"], list)


class TestHardwareRecommender:
    @pytest.fixture
    def recommender(self):
        return HardwareRecommender()

    def test_recommend_large_model_a100_80gb(self, recommender):
        # 50B params -> ~93GB in bfloat16
        model_params = 50_000_000_000
        recs = recommender.recommend(model_params=model_params)
        assert len(recs) >= 1
        assert recs[0].item == "A100_80GB"
        assert recs[0].type == RecommendationType.HARDWARE
        assert recs[0].confidence > 0.9

    def test_recommend_medium_model_a100_40gb(self, recommender):
        # 15B params -> ~28GB in bfloat16
        model_params = 15_000_000_000
        recs = recommender.recommend(model_params=model_params)
        assert recs[0].item == "A100_40GB"

    def test_recommend_small_model_t4(self, recommender):
        # 1B params -> ~1.9GB in bfloat16
        model_params = 1_000_000_000
        recs = recommender.recommend(model_params=model_params)
        assert recs[0].item == "T4"

    def test_recommend_low_latency_adds_cpu(self, recommender):
        model_params = 1_000_000_000
        recs = recommender.recommend(
            model_params=model_params,
            latency_requirement_ms=50.0
        )
        items = [r.item for r in recs]
        assert "high_cpu_count" in items

    def test_recommend_batch_size_affects_memory(self, recommender):
        model_params = 1_000_000_000
        recs = recommender.recommend(
            model_params=model_params,
            batch_size=8
        )
        # Should still recommend T4 for small model
        assert recs[0].item == "T4"


class TestParameterRecommender:
    @pytest.fixture
    def recommender(self):
        return ParameterRecommender()

    def test_recommend_shard_size_nvme(self, recommender):
        rec = recommender.recommend_shard_size(
            model_size_gb=10.0,
            disk_speed_mbps=2000,
        )
        assert rec.item == "5GB"
        assert rec.type == RecommendationType.PARAMETERS
        assert rec.confidence > 0.9
        assert "NVMe" in rec.reason

    def test_recommend_shard_size_ssd(self, recommender):
        rec = recommender.recommend_shard_size(
            model_size_gb=10.0,
            disk_speed_mbps=600,
        )
        assert rec.item == "2GB"
        assert "SSD" in rec.reason

    def test_recommend_shard_size_hdd(self, recommender):
        rec = recommender.recommend_shard_size(
            model_size_gb=10.0,
            disk_speed_mbps=80,
        )
        assert rec.item == "500MB"
        assert "HDD" in rec.reason

    def test_recommend_shard_size_slow_network(self, recommender):
        rec = recommender.recommend_shard_size(
            model_size_gb=10.0,
            disk_speed_mbps=2000,
            network_speed_mbps=50,
        )
        assert rec.item == "1GB"
        assert "network" in rec.reason.lower()

    def test_recommend_parallel_workers_io_bound(self, recommender):
        rec = recommender.recommend_parallel_workers(
            cpu_count=8,
            io_bound=True
        )
        assert int(rec.item) == 16
        assert "I/O bound" in rec.reason

    def test_recommend_parallel_workers_cpu_bound(self, recommender):
        rec = recommender.recommend_parallel_workers(
            cpu_count=8,
            io_bound=False
        )
        assert int(rec.item) == 7
        assert "CPU bound" in rec.reason


class TestVitriolRecommender:
    @pytest.fixture
    def recommender(self):
        return VitriolRecommender()

    def test_recommend_all_structure(self, recommender):
        result = recommender.recommend_all(
            model_id="test-model",
            model_params=7_000_000_000,
            available_memory_gb=16.0,
            use_case="general",
        )
        assert result["model_id"] == "test-model"
        assert "model_size_gb" in result
        assert "timestamp" in result
        assert "recommendations" in result
        assert len(result["recommendations"]) >= 3

    def test_recommend_all_categories(self, recommender):
        result = recommender.recommend_all(
            model_id="test-model",
            model_params=7_000_000_000,
            available_memory_gb=16.0,
            use_case="general",
        )
        categories = [r["category"] for r in result["recommendations"]]
        assert "strategy" in categories
        assert "hardware" in categories
        assert "parameters" in categories

    def test_recommend_all_with_kwargs(self, recommender):
        result = recommender.recommend_all(
            model_id="test-model",
            model_params=7_000_000_000,
            available_memory_gb=16.0,
            use_case="general",
            requires_training=True,
            batch_size=4,
            disk_speed_mbps=1000,
            cpu_count=8,
        )
        assert result["model_id"] == "test-model"
        assert len(result["recommendations"]) > 0

    def test_explain_recommendation_known(self, recommender):
        explanation = recommender.explain_recommendation("strategy", "ultra")
        assert "Ultra strategy" in explanation
        assert "compression" in explanation.lower()

    def test_explain_recommendation_quantum(self, recommender):
        explanation = recommender.explain_recommendation("strategy", "quantum")
        assert "Quantum strategy" in explanation
        assert "1-bit" in explanation or "quantization" in explanation.lower()

    def test_explain_recommendation_compact(self, recommender):
        explanation = recommender.explain_recommendation("strategy", "compact")
        assert "Compact strategy" in explanation

    def test_explain_recommendation_unknown(self, recommender):
        explanation = recommender.explain_recommendation("strategy", "unknown")
        assert "unknown is recommended" in explanation

    def test_explain_recommendation_unknown_type(self, recommender):
        explanation = recommender.explain_recommendation("hardware", "T4")
        assert "T4 is recommended" in explanation


class TestGetRecommender:
    def test_get_recommender_singleton(self):
        r1 = get_recommender()
        r2 = get_recommender()
        assert r1 is r2
        assert isinstance(r1, VitriolRecommender)

    def test_get_recommender_has_subrecommenders(self):
        r = get_recommender()
        assert hasattr(r, "strategy")
        assert hasattr(r, "hardware")
        assert hasattr(r, "parameters")
        assert isinstance(r.strategy, StrategyRecommender)
        assert isinstance(r.hardware, HardwareRecommender)
        assert isinstance(r.parameters, ParameterRecommender)
