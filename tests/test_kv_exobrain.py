"""Tests for kv/exobrain.py."""


import pytest
import torch

from vitriol.kv.exobrain import (
    AdaptiveLayerSelector,
    ExoBrainBus,
    ExoBrainConfig,
    LocalWeightSource,
    MultiTeacherRouter,
    ShellProjection,
    VectorDBSource,
    APIKnowledgeSource,
    compute_attention_entropy,
    compute_gate,
    cross_attention_fusion,
)


class TestShellProjection:
    """Tests for ShellProjection."""

    def test_linear_mode_creation(self):
        proj = ShellProjection(768, 4096, mode="linear")
        assert proj.shell_hidden_dim == 768
        assert proj.brain_hidden_dim == 4096
        assert proj.mode == "linear"
        assert proj.num_parameters > 0

    def test_mlp_mode_creation(self):
        proj = ShellProjection(768, 4096, mode="mlp")
        assert proj.mode == "mlp"
        assert proj.num_parameters > 0

    def test_linear_ln_mode_creation(self):
        proj = ShellProjection(768, 4096, mode="linear_ln")
        assert proj.mode == "linear_ln"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="unknown mode"):
            ShellProjection(768, 4096, mode="invalid")

    def test_forward_3d(self):
        proj = ShellProjection(768, 4096, mode="linear")
        x = torch.randn(2, 10, 768)
        out = proj.forward(x)
        assert out.shape == (2, 10, 4096)

    def test_forward_4d(self):
        proj = ShellProjection(768, 4096, mode="linear")
        x = torch.randn(2, 8, 10, 768)
        out = proj.forward(x)
        assert out.shape == (2, 8, 10, 4096)

    def test_forward_invalid_dims(self):
        proj = ShellProjection(768, 4096, mode="linear")
        with pytest.raises(ValueError, match="got 2D"):
            proj.forward(torch.randn(2, 768))

    def test_project_query(self):
        proj = ShellProjection(768, 4096, mode="linear")
        q = torch.randn(2, 8, 10, 768)
        out = proj.project_query(q)
        assert out.shape == (2, 8, 10, 4096)

    def test_parameter_count_str(self):
        proj = ShellProjection(10, 20, mode="linear")
        s = proj.parameter_count_str
        assert "K" in s or "M" in s or s.isdigit()


class TestVectorDBSource:
    """Tests for VectorDBSource."""

    def test_empty_source(self):
        source = VectorDBSource()
        assert source.num_docs == 0
        assert source.name == "vector_db"
        result = source.retrieve_kv(torch.randn(1, 4, 5, 64), layer_idx=0)
        assert result is None

    def test_add_document(self):
        source = VectorDBSource()
        key = torch.randn(4, 5, 64)
        value = torch.randn(4, 5, 64)
        emb = torch.randn(64)
        idx = source.add_document(key, value, emb, text="test")
        assert idx == 0
        assert source.num_docs == 1

    def test_remove_document(self):
        source = VectorDBSource()
        key = torch.randn(4, 5, 64)
        value = torch.randn(4, 5, 64)
        emb = torch.randn(64)
        source.add_document(key, value, emb)
        assert source.remove_document(0) is True
        assert source.remove_document(0) is False

    def test_recompute_embeddings(self):
        source = VectorDBSource()
        key = torch.randn(4, 5, 64)
        value = torch.randn(4, 5, 64)
        emb = torch.randn(64)
        source.add_document(key, value, emb)
        source.recompute_embeddings(normalize=True)
        assert source.num_docs == 1

    def test_retrieve_kv(self):
        source = VectorDBSource()
        key = torch.randn(2, 4, 5, 64)  # 2 docs
        value = torch.randn(2, 4, 5, 64)
        emb = torch.randn(2, 64)
        source._keys = key
        source._values = value
        source._embeddings = emb
        source._texts = ["a", "b"]
        query = torch.randn(1, 4, 3, 64)
        result = source.retrieve_kv(query, layer_idx=0, top_k=1)
        assert result is not None
        ext_k, ext_v = result
        assert ext_k.ndim == 4
        assert ext_v.ndim == 4

    def test_search_texts(self):
        source = VectorDBSource()
        source._texts = ["hello world", "foo bar", "hello there"]
        results = source.search_texts(["hello"], top_k=2)
        assert len(results) == 1
        assert len(results[0]) <= 2

    def test_get_document(self):
        source = VectorDBSource()
        key = torch.randn(4, 5, 64)
        value = torch.randn(4, 5, 64)
        emb = torch.randn(64)
        source.add_document(key, value, emb, text="test", metadata={"x": 1})
        doc = source.get_document(0)
        assert doc["text"] == "test"
        assert doc["metadata"] == {"x": 1}
        assert source.get_document(99) is None

    def test_stats(self):
        source = VectorDBSource()
        assert source.stats == {"retrievals": 0, "hits": 0}


class TestLocalWeightSource:
    """Tests for LocalWeightSource."""

    def test_empty_source(self):
        source = LocalWeightSource()
        assert source.name == "local_weight"
        result = source.retrieve_kv(torch.randn(1, 4, 5, 64), layer_idx=0)
        assert result is None

    def test_set_and_retrieve(self):
        source = LocalWeightSource()
        key = torch.randn(1, 4, 10, 64)
        value = torch.randn(1, 4, 10, 64)
        source.set_teacher_kv(0, key, value)
        query = torch.randn(1, 4, 5, 64)
        result = source.retrieve_kv(query, layer_idx=0, top_k=3)
        assert result is not None
        ext_k, ext_v = result
        assert ext_k.shape[0] == 1
        assert ext_k.shape[1] == 4


class TestExoBrainConfig:
    """Tests for ExoBrainConfig."""

    def test_default_creation(self):
        cfg = ExoBrainConfig()
        assert cfg.fusion_mode == "replace"
        assert cfg.retrieval_top_k == 5
        assert cfg.key_layers == []
        assert cfg.fallback_on_error is True

    def test_invalid_fusion_mode(self):
        with pytest.raises(ValueError, match="invalid fusion_mode"):
            ExoBrainConfig(fusion_mode="invalid")

    def test_is_key_layer_empty_list(self):
        cfg = ExoBrainConfig(key_layers=[])
        assert cfg.is_key_layer(0) is True
        assert cfg.is_key_layer(99) is True

    def test_is_key_layer_with_layers(self):
        cfg = ExoBrainConfig(key_layers=[3, 4, 5])
        assert cfg.is_key_layer(3) is True
        assert cfg.is_key_layer(4) is True
        assert cfg.is_key_layer(0) is False
        assert cfg.is_key_layer(99) is False

    def test_active_layers_alias(self):
        cfg = ExoBrainConfig()
        cfg.active_layers = [1, 2]
        assert cfg.key_layers == [1, 2]
        assert cfg.active_layers == [1, 2]


class TestAdaptiveLayerSelector:
    """Tests for AdaptiveLayerSelector."""

    def test_select_all(self):
        selector = AdaptiveLayerSelector(total_layers=8, strategy="all")
        assert selector.select() == list(range(8))

    def test_select_middle_heavy(self):
        selector = AdaptiveLayerSelector(total_layers=8, strategy="middle_heavy")
        result = selector.select()
        assert len(result) > 0
        assert all(2 <= r <= 6 for r in result)

    def test_select_by_top_k(self):
        selector = AdaptiveLayerSelector(total_layers=8, strategy="entropy_top_k", top_k_ratio=0.5)
        entropy = {i: float(i) for i in range(8)}
        selector.observe(entropy)
        result = selector.select()
        assert len(result) >= selector.min_layers

    def test_select_by_threshold(self):
        selector = AdaptiveLayerSelector(total_layers=8, strategy="entropy_threshold", entropy_threshold=3.0)
        entropy = {i: float(i) for i in range(8)}
        selector.observe(entropy)
        result = selector.select()
        assert all(r >= 3 for r in result)

    def test_is_stable(self):
        selector = AdaptiveLayerSelector(total_layers=8, stability_window=3)
        assert selector.is_stable() is False
        for i in range(3):
            selector.observe({0: 0.5})
        assert selector.is_stable() is True

    def test_cached_selection(self):
        selector = AdaptiveLayerSelector(total_layers=4, strategy="middle_heavy")
        r1 = selector.select()
        r2 = selector.select()
        assert r1 == r2
        assert selector._cached_selection is not None

    def test_stats(self):
        selector = AdaptiveLayerSelector(total_layers=4, strategy="all")
        stats = selector.stats
        assert stats["strategy"] == "all"
        assert stats["total_layers"] == 4


class TestExoBrainBus:
    """Tests for ExoBrainBus."""

    def test_empty_bus(self):
        bus = ExoBrainBus()
        assert bus.sources == []
        query = torch.randn(1, 4, 5, 64)
        result = bus.retrieve(query, layer_idx=0)
        assert result is None

    def test_add_remove_source(self):
        bus = ExoBrainBus()
        source = VectorDBSource(name="test")
        bus.add_source(source)
        assert len(bus.sources) == 1
        bus.remove_source("test")
        assert len(bus.sources) == 0

    def test_inject_kv(self):
        bus = ExoBrainBus()
        key = torch.randn(1, 4, 5, 64)
        value = torch.randn(1, 4, 5, 64)
        bus.inject_kv(0, key, value)
        query = torch.randn(1, 4, 3, 64)
        result = bus.retrieve(query, layer_idx=0)
        assert result is not None
        ext_k, ext_v = result
        assert torch.equal(ext_k, key)

    def test_clear_injected(self):
        bus = ExoBrainBus()
        bus.inject_kv(0, torch.randn(1, 4, 5, 64), torch.randn(1, 4, 5, 64))
        bus.clear_injected()
        assert len(bus._injected_kv) == 0

    def test_stats(self):
        bus = ExoBrainBus()
        stats = bus.stats
        assert "hit_rate" in stats
        assert stats["hit_rate"] == 0.0

    def test_retrieve_with_source(self):
        bus = ExoBrainBus()
        source = VectorDBSource()
        key = torch.randn(2, 4, 5, 64)
        value = torch.randn(2, 4, 5, 64)
        emb = torch.randn(2, 64)
        source._keys = key
        source._values = value
        source._embeddings = emb
        source._texts = ["a", "b"]
        bus.add_source(source)
        query = torch.randn(1, 4, 3, 64)
        result = bus.retrieve(query, layer_idx=0)
        assert result is not None


class TestMultiTeacherRouter:
    """Tests for MultiTeacherRouter."""

    def test_empty_router(self):
        router = MultiTeacherRouter()
        query = torch.randn(1, 4, 5, 64)
        result = router.route(query, layer_idx=0)
        assert result is None

    def test_add_remove_teacher(self):
        router = MultiTeacherRouter()
        bus = ExoBrainBus()
        router.add_teacher("teacher1", bus)
        assert "teacher1" in router.teachers
        router.remove_teacher("teacher1")
        assert "teacher1" not in router.teachers

    def test_round_robin(self):
        bus1 = ExoBrainBus()
        bus1.inject_kv(0, torch.randn(1, 4, 5, 64), torch.randn(1, 4, 5, 64))
        bus2 = ExoBrainBus()
        bus2.inject_kv(0, torch.randn(1, 4, 5, 64), torch.randn(1, 4, 5, 64))
        router = MultiTeacherRouter(
            teachers={"t1": bus1, "t2": bus2},
            strategy="round_robin",
        )
        query = torch.randn(1, 4, 3, 64)
        result = router.route(query, layer_idx=0)
        assert result is not None

    def test_first_available(self):
        bus1 = ExoBrainBus()
        bus2 = ExoBrainBus()
        bus2.inject_kv(0, torch.randn(1, 4, 5, 64), torch.randn(1, 4, 5, 64))
        router = MultiTeacherRouter(
            teachers={"t1": bus1, "t2": bus2},
            strategy="first_available",
        )
        query = torch.randn(1, 4, 3, 64)
        result = router.route(query, layer_idx=0)
        assert result is not None

    def test_ensemble(self):
        bus1 = ExoBrainBus()
        bus1.inject_kv(0, torch.randn(1, 4, 5, 64), torch.randn(1, 4, 5, 64))
        bus2 = ExoBrainBus()
        bus2.inject_kv(0, torch.randn(1, 4, 5, 64), torch.randn(1, 4, 5, 64))
        router = MultiTeacherRouter(
            teachers={"t1": bus1, "t2": bus2},
            strategy="ensemble",
        )
        query = torch.randn(1, 4, 3, 64)
        result = router.route(query, layer_idx=0)
        assert result is not None

    def test_stats(self):
        router = MultiTeacherRouter()
        assert router.stats["total_routes"] == 0

    def test_align_kv(self):
        router = MultiTeacherRouter()
        kv = torch.randn(1, 4, 5, 64)
        ref = torch.Size([1, 4, 10, 64])
        aligned = router._align_kv(kv, ref)
        assert aligned.shape == ref


class TestCrossAttentionFusion:
    """Tests for cross_attention_fusion."""

    def test_basic_fusion(self):
        query = torch.randn(2, 4, 5, 64)
        ext_k = torch.randn(2, 4, 10, 64)
        ext_v = torch.randn(2, 4, 10, 64)
        out = cross_attention_fusion(query, ext_k, ext_v)
        assert out.shape == query.shape

    def test_fusion_with_mask(self):
        query = torch.randn(2, 4, 5, 64)
        ext_k = torch.randn(2, 4, 10, 64)
        ext_v = torch.randn(2, 4, 10, 64)
        mask = torch.ones(2, 4, 5, 10, dtype=torch.bool)
        out = cross_attention_fusion(query, ext_k, ext_v, attn_mask=mask)
        assert out.shape == query.shape

    def test_fusion_with_scale(self):
        query = torch.randn(2, 4, 5, 64)
        ext_k = torch.randn(2, 4, 10, 64)
        ext_v = torch.randn(2, 4, 10, 64)
        out = cross_attention_fusion(query, ext_k, ext_v, scale=0.5)
        assert out.shape == query.shape

    def test_fusion_with_dropout(self):
        query = torch.randn(2, 4, 5, 64)
        ext_k = torch.randn(2, 4, 10, 64)
        ext_v = torch.randn(2, 4, 10, 64)
        out = cross_attention_fusion(query, ext_k, ext_v, dropout_p=0.1, training=True)
        assert out.shape == query.shape


class TestComputeGate:
    """Tests for compute_gate."""

    def test_max_similarity(self):
        query = torch.randn(2, 4, 5, 64)
        ext_k = torch.randn(2, 4, 10, 64)
        gate = compute_gate(query, ext_k, mode="max_similarity")
        assert gate.shape == (2, 4, 5, 1)
        assert (gate >= 0).all() and (gate <= 1).all()

    def test_mean_similarity(self):
        query = torch.randn(2, 4, 5, 64)
        ext_k = torch.randn(2, 4, 10, 64)
        gate = compute_gate(query, ext_k, mode="mean_similarity")
        assert gate.shape == (2, 4, 5, 1)

    def test_per_head_entropy(self):
        query = torch.randn(2, 4, 5, 64)
        ext_k = torch.randn(2, 4, 10, 64)
        gate = compute_gate(query, ext_k, mode="per_head_entropy")
        assert gate.shape == (2, 4, 5, 1)

    def test_learned_mode(self):
        query = torch.randn(2, 4, 5, 64)
        ext_k = torch.randn(2, 4, 10, 64)
        learned_proj = torch.nn.Linear(64, 1)
        gate = compute_gate(query, ext_k, mode="learned", learned_proj=learned_proj)
        assert gate.shape == (2, 4, 5, 1)


class TestComputeAttentionEntropy:
    """Tests for compute_attention_entropy."""

    def test_entropy_shape(self):
        attn = torch.softmax(torch.randn(2, 4, 5, 10), dim=-1)
        entropy = compute_attention_entropy(attn)
        assert entropy.shape == (2, 4, 5)

    def test_entropy_positive(self):
        attn = torch.softmax(torch.randn(2, 4, 5, 10), dim=-1)
        entropy = compute_attention_entropy(attn)
        assert (entropy >= 0).all()

    def test_uniform_high_entropy(self):
        # Uniform distribution should have max entropy
        attn = torch.ones(1, 1, 1, 100) / 100
        entropy = compute_attention_entropy(attn)
        # log(100) ≈ 4.6
        assert entropy.item() > 3.0

    def test_peaked_low_entropy(self):
        # Peaked distribution should have low entropy
        attn = torch.zeros(1, 1, 1, 100)
        attn[0, 0, 0, 0] = 1.0
        entropy = compute_attention_entropy(attn)
        assert entropy.item() < 0.1


class TestAPIKnowledgeSource:
    """Tests for APIKnowledgeSource."""

    def test_creation(self):
        source = APIKnowledgeSource(endpoint="http://test", api_key="secret")
        assert source.name == "api_source"
        assert source._endpoint == "http://test"
        assert source._api_key == "secret"

    def test_no_endpoint_returns_none(self):
        source = APIKnowledgeSource()
        query = torch.randn(1, 4, 5, 64)
        result = source.retrieve_kv(query, layer_idx=0)
        assert result is None

    def test_inject_kv(self):
        source = APIKnowledgeSource()
        key = torch.randn(1, 4, 5, 64)
        value = torch.randn(1, 4, 5, 64)
        source.inject_kv("test_key", key, value)
        query = torch.randn(1, 4, 3, 64)
        result = source.retrieve_kv(query, layer_idx=0)
        assert result is not None

    def test_cache_ttl(self):
        source = APIKnowledgeSource()
        source.set_cache_ttl(3600)
        assert source._cache_ttl_seconds == 3600

    def test_clear_cache(self):
        source = APIKnowledgeSource()
        source.inject_kv("k", torch.randn(1, 4, 5, 64), torch.randn(1, 4, 5, 64))
        source.clear_cache()
        assert len(source._cache) == 0

    def test_clear_expired(self):
        source = APIKnowledgeSource()
        source.set_cache_ttl(0)
        source.inject_kv("k", torch.randn(1, 4, 5, 64), torch.randn(1, 4, 5, 64))
        import time
        time.sleep(0.1)
        removed = source.clear_expired()
        assert removed >= 0

    def test_stats(self):
        source = APIKnowledgeSource()
        stats = source.stats
        assert "api_calls" in stats
        assert "cache_hits" in stats
