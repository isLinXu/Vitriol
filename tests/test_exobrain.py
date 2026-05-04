#!/usr/bin/env python3
"""
ExoBrain Comprehensive Test Suite.

Run with: python tests/test_exobrain.py
"""

import unittest

import torch

from vitriol.kv.exobrain import (
    ExoBrainBackend, ExoBrainBus, ExoBrainConfig,
    VectorDBSource, APIKnowledgeSource, LocalWeightSource,
    cross_attention_fusion, compute_gate,
)
from vitriol.kv.cache_store import KVCacheStoreConfig
from vitriol.kv.exobrain_inference import HeadDimProjection


# ═══════════════════════════════════════════════════════════════════
# Cross-Attention Fusion Tests
# ═══════════════════════════════════════════════════════════════════

class TestCrossAttentionFusion(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.B, self.H, self.Q, self.KV, self.D = 2, 4, 3, 8, 64
        self.query = torch.randn(self.B, self.H, self.Q, self.D)
        self.ext_k = torch.randn(self.B, self.H, self.KV, self.D)
        self.ext_v = torch.randn(self.B, self.H, self.KV, self.D)

    def test_output_shape(self):
        out = cross_attention_fusion(self.query, self.ext_k, self.ext_v)
        self.assertEqual(out.shape, self.query.shape)

    def test_no_nans(self):
        out = cross_attention_fusion(self.query, self.ext_k, self.ext_v)
        self.assertFalse(torch.isnan(out).any())

    def test_zero_kv(self):
        zeros = torch.zeros_like(self.ext_k)
        out = cross_attention_fusion(self.query, zeros, zeros)
        self.assertEqual(out.shape, self.query.shape)
        self.assertTrue(torch.allclose(out, torch.zeros_like(out), atol=1e-5))

    def test_with_mask(self):
        mask = torch.ones(self.B, self.H, self.Q, self.KV, dtype=torch.bool)
        mask[:, :, :, -2:] = False
        out = cross_attention_fusion(self.query, self.ext_k, self.ext_v, attn_mask=mask)
        self.assertEqual(out.shape, self.query.shape)

    def test_dropout_training(self):
        out = cross_attention_fusion(self.query, self.ext_k, self.ext_v, dropout_p=0.1, training=True)
        self.assertEqual(out.shape, self.query.shape)

    def test_dropout_reproducible_seed(self):
        out1 = cross_attention_fusion(self.query, self.ext_k, self.ext_v, dropout_p=0.5, training=True, dropout_seed=42)
        out2 = cross_attention_fusion(self.query, self.ext_k, self.ext_v, dropout_p=0.5, training=True, dropout_seed=42)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-5))

    def test_single_head(self):
        q = torch.randn(1, 1, 2, 32)
        k = torch.randn(1, 1, 4, 32)
        v = torch.randn(1, 1, 4, 32)
        out = cross_attention_fusion(q, k, v)
        self.assertEqual(out.shape, q.shape)


# ═══════════════════════════════════════════════════════════════════
# Compute Gate Tests
# ═══════════════════════════════════════════════════════════════════

class TestComputeGate(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.q = torch.randn(2, 4, 3, 64)
        self.ext_k = torch.randn(2, 4, 8, 64)

    def test_max_similarity_gate(self):
        gate = compute_gate(self.q, self.ext_k, temperature=1.0, mode="max_similarity")
        self.assertEqual(gate.shape, (2, 4, 3, 1))
        self.assertTrue((gate >= 0).all() and (gate <= 1).all())

    def test_mean_similarity_gate(self):
        gate = compute_gate(self.q, self.ext_k, temperature=1.0, mode="mean_similarity")
        self.assertEqual(gate.shape, (2, 4, 3, 1))
        self.assertTrue((gate >= 0).all() and (gate <= 1).all())

    def test_temperature_bounds(self):
        for temp in [0.01, 0.1, 1.0, 10.0, 100.0]:
            gate = compute_gate(self.q, self.ext_k, temperature=temp)
            self.assertTrue((gate >= 0).all() and (gate <= 1).all())


# ═══════════════════════════════════════════════════════════════════
# ExoBrainConfig Tests
# ═══════════════════════════════════════════════════════════════════

class TestExoBrainConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = ExoBrainConfig()
        self.assertEqual(cfg.fusion_mode, "replace")
        self.assertEqual(cfg.residual_alpha, 0.1)
        self.assertEqual(cfg.retrieval_top_k, 5)
        self.assertTrue(cfg.auto_project)

    def test_valid_modes(self):
        for mode in ["replace", "residual", "gated"]:
            cfg = ExoBrainConfig(fusion_mode=mode)
            self.assertEqual(cfg.fusion_mode, mode)

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            ExoBrainConfig(fusion_mode="invalid")

    def test_active_layers(self):
        cfg = ExoBrainConfig(key_layers=[0, 2, 5])
        self.assertEqual(cfg.key_layers, [0, 2, 5])


# ═══════════════════════════════════════════════════════════════════
# VectorDBSource Tests
# ═══════════════════════════════════════════════════════════════════

class TestVectorDBSource(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.n_docs, self.dim = 20, 64
        self.keys = torch.randn(self.n_docs, self.dim) * 0.5
        self.values = torch.randn(self.n_docs, self.dim) * 0.5
        self.embeddings = self.keys.clone()
        self.source = VectorDBSource(
            keys=self.keys, values=self.values, embeddings=self.embeddings,
        )

    def test_num_docs(self):
        self.assertEqual(self.source.num_docs, self.n_docs)

    def test_retrieve_basic(self):
        query = torch.randn(1, 4, 1, self.dim)
        result = self.source.retrieve_kv(query, layer_idx=0, top_k=5)
        self.assertIsNotNone(result)
        ext_k, ext_v = result
        self.assertEqual(ext_k.shape[0], 1)   # batch
        self.assertEqual(ext_k.shape[1], 4)   # heads
        self.assertEqual(ext_k.shape[2], 5)   # top_k
        self.assertEqual(ext_k.shape[3], self.dim)

    def test_retrieve_empty_db(self):
        empty = VectorDBSource()
        result = empty.retrieve_kv(torch.randn(1, 4, 1, 64), layer_idx=0)
        self.assertIsNone(result)

    def test_top_k_limiting(self):
        query = torch.randn(1, 4, 1, self.dim)
        for k in [1, 3, 10, 20]:
            result = self.source.retrieve_kv(query, layer_idx=0, top_k=k)
            actual_k = min(k, self.n_docs)
            self.assertEqual(result[0].shape[2], actual_k)

    def test_get_document(self):
        doc = self.source.get_document(0)
        self.assertIsNotNone(doc)
        self.assertIn("text", doc)

    def test_recompute_embeddings(self):
        self.source.recompute_embeddings(normalize=True)

    def test_stats(self):
        query = torch.randn(1, 4, 1, self.dim)
        self.source.retrieve_kv(query, layer_idx=0, top_k=5)
        stats = self.source.stats
        self.assertEqual(stats["retrievals"], 1)
        self.assertEqual(stats["hits"], 1)


# ═══════════════════════════════════════════════════════════════════
# APIKnowledgeSource Tests
# ═══════════════════════════════════════════════════════════════════

class TestAPIKnowledgeSource(unittest.TestCase):

    def test_manual_inject_and_retrieve(self):
        source = APIKnowledgeSource()
        key = torch.randn(1, 4, 8, 64)
        value = torch.randn(1, 4, 8, 64)
        source.inject_kv("test_key", key, value)
        query = torch.randn(1, 4, 1, 64)
        result = source.retrieve_kv(query, layer_idx=0)
        self.assertIsNotNone(result)
        self.assertEqual(source.stats["injected_hits"], 1)

    def test_no_endpoint_returns_none_without_injection(self):
        source = APIKnowledgeSource()
        result = source.retrieve_kv(torch.randn(1, 4, 1, 64), layer_idx=0)
        self.assertIsNone(result)

    def test_clear_cache(self):
        source = APIKnowledgeSource()
        source.inject_kv("k1", torch.randn(1, 4, 8, 64), torch.randn(1, 4, 8, 64))
        self.assertEqual(len(source._cache), 1)
        source.clear_cache()
        self.assertEqual(len(source._cache), 0)

    def test_cache_key_deterministic(self):
        source = APIKnowledgeSource()
        q1 = torch.randn(1, 4, 1, 64)
        q2 = q1.clone()
        k1 = source._query_to_cache_key(q1)
        k2 = source._query_to_cache_key(q2)
        self.assertEqual(k1, k2)


# ═══════════════════════════════════════════════════════════════════
# LocalWeightSource Tests
# ═══════════════════════════════════════════════════════════════════

class TestLocalWeightSource(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        kv = {}
        for i in range(4):
            kv[i] = (torch.randn(1, 4, 16, 64) * 0.3, torch.randn(1, 4, 16, 64) * 0.3)
        self.source = LocalWeightSource(teacher_kv=kv)

    def test_retrieve_existing_layer(self):
        result = self.source.retrieve_kv(torch.randn(1, 4, 1, 64), layer_idx=0, top_k=5)
        self.assertIsNotNone(result)

    def test_retrieve_nonexistent_layer(self):
        result = self.source.retrieve_kv(torch.randn(1, 4, 1, 64), layer_idx=99, top_k=5)
        self.assertIsNone(result)

    def test_top_k_limiting(self):
        for top_k in [1, 3, 8, 16]:
            result = self.source.retrieve_kv(torch.randn(1, 4, 1, 64), layer_idx=0, top_k=top_k)
            self.assertEqual(result[0].shape[2], min(top_k, 16))


# ═══════════════════════════════════════════════════════════════════
# ExoBrainBus Tests
# ═══════════════════════════════════════════════════════════════════

class TestExoBrainBus(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.bus = ExoBrainBus()

    def test_default_stats(self):
        stats = self.bus.stats
        self.assertEqual(stats["retrieve_count"], 0)
        self.assertEqual(stats["hit_count"], 0)

    def test_direct_inject_retrieve(self):
        self.bus.inject_kv(0, torch.randn(1, 4, 8, 64), torch.randn(1, 4, 8, 64))
        result = self.bus.retrieve(torch.randn(1, 4, 1, 64), layer_idx=0)
        self.assertIsNotNone(result)
        self.assertEqual(self.bus.stats["hit_count"], 1)

    def test_multi_layer(self):
        for i in range(4):
            self.bus.inject_kv(i, torch.randn(1, 4, 8, 64), torch.randn(1, 4, 8, 64))
        for i in range(4):
            result = self.bus.retrieve(torch.randn(1, 4, 1, 64), layer_idx=i)
            self.assertIsNotNone(result, f"Layer {i} should have KV")

    def test_miss_on_unknown_layer(self):
        result = self.bus.retrieve(torch.randn(1, 4, 1, 64), layer_idx=99)
        self.assertIsNone(result)
        self.assertEqual(self.bus.stats["miss_count"], 1)

    def test_source_priority(self):
        vdb = VectorDBSource(
            keys=torch.randn(10, 64), values=torch.randn(10, 64),
            embeddings=torch.randn(10, 64),
        )
        self.bus.add_source(vdb)
        result = self.bus.retrieve(torch.randn(1, 4, 1, 64), layer_idx=0)
        self.assertIsNotNone(result)

    def test_clear_injected(self):
        self.bus.inject_kv(0, torch.randn(1, 4, 8, 64), torch.randn(1, 4, 8, 64))
        self.assertEqual(len(self.bus._injected_kv), 1)
        self.bus.clear_injected()
        self.assertEqual(len(self.bus._injected_kv), 0)

    def test_add_remove_source(self):
        source = VectorDBSource(name="test_src")
        self.bus.add_source(source)
        self.assertEqual(len(self.bus.sources), 1)
        self.bus.remove_source("test_src")
        self.assertEqual(len(self.bus.sources), 0)


# ═══════════════════════════════════════════════════════════════════
# ExoBrainBackend Integration Tests
# ═══════════════════════════════════════════════════════════════════

class TestExoBrainBackend(unittest.TestCase):

    def test_replace_mode_no_handle(self):
        bus = ExoBrainBus()
        bus.inject_kv(0, torch.randn(1, 4, 8, 64), torch.randn(1, 4, 8, 64))
        cfg = ExoBrainConfig(fusion_mode="replace")
        backend = ExoBrainBackend(
            store_cfg=KVCacheStoreConfig(), brain_bus=bus, brain_cfg=cfg,
        )
        query = torch.randn(1, 4, 1, 64)
        out = backend.read_attention(
            handle=None, layer_idx=0, query=query,
            attn_mask=None, is_causal=False, scale=None,
            info={"dropout_p": 0.0},
        )
        self.assertEqual(out.shape, query.shape)

    def test_all_fusion_modes(self):
        bus = ExoBrainBus()
        bus.inject_kv(0, torch.randn(1, 4, 8, 64), torch.randn(1, 4, 8, 64))
        for mode in ["replace", "residual", "gated"]:
            cfg = ExoBrainConfig(fusion_mode=mode)
            backend = ExoBrainBackend(
                store_cfg=KVCacheStoreConfig(), brain_bus=bus, brain_cfg=cfg,
            )
            query = torch.randn(1, 4, 1, 64)
            out = backend.read_attention(
                handle=None, layer_idx=0, query=query,
                attn_mask=None, is_causal=False, scale=None,
                info={"dropout_p": 0.0},
            )
            self.assertEqual(out.shape, query.shape, f"Mode {mode} should work")

    def test_injected_priority_over_source(self):
        vdb = VectorDBSource(
            keys=torch.randn(10, 64), values=torch.randn(10, 64),
            embeddings=torch.randn(10, 64),
        )
        bus = ExoBrainBus(sources=[vdb])
        bus.inject_kv(0, torch.randn(1, 4, 8, 64) * 5.0, torch.randn(1, 4, 8, 64) * 5.0)
        result = bus.retrieve(torch.randn(1, 4, 1, 64), layer_idx=0)
        self.assertIsNotNone(result)
        self.assertEqual(bus.stats["hit_count"], 1)

    def test_fusion_stats(self):
        bus = ExoBrainBus()
        bus.inject_kv(0, torch.randn(1, 4, 8, 64), torch.randn(1, 4, 8, 64))
        cfg = ExoBrainConfig(fusion_mode="replace")
        backend = ExoBrainBackend(
            store_cfg=KVCacheStoreConfig(), brain_bus=bus, brain_cfg=cfg,
        )
        backend.read_attention(
            handle=None, layer_idx=0, query=torch.randn(1, 4, 1, 64),
            attn_mask=None, is_causal=False, scale=None,
            info={"dropout_p": 0.0},
        )
        self.assertGreater(backend.fusion_stats["replace_count"], 0)


# ═══════════════════════════════════════════════════════════════════
# HeadDimProjection Tests
# ═══════════════════════════════════════════════════════════════════

class TestHeadDimProjection(unittest.TestCase):

    def test_same_dim_no_change(self):
        proj = HeadDimProjection(teacher_head_dim=64, shell_head_dim=64)
        tensor = torch.randn(2, 4, 8, 64)
        out = proj(tensor)
        self.assertEqual(out.shape, tensor.shape)

    def test_truncate(self):
        proj = HeadDimProjection(teacher_head_dim=128, shell_head_dim=64, mode="pad_or_truncate")
        tensor = torch.randn(2, 4, 8, 128)
        out = proj(tensor)
        self.assertEqual(out.shape, (2, 4, 8, 64))

    def test_pad(self):
        proj = HeadDimProjection(teacher_head_dim=32, shell_head_dim=64, mode="pad_or_truncate")
        tensor = torch.randn(2, 4, 8, 32)
        out = proj(tensor)
        self.assertEqual(out.shape, (2, 4, 8, 64))
        self.assertTrue(torch.allclose(out[..., -1], torch.zeros(2, 4, 8), atol=1e-4))

    def test_learned_projection_shape(self):
        proj = HeadDimProjection(
            teacher_head_dim=128, shell_head_dim=64,
            num_kv_heads=4, mode="learned"
        )
        tensor = torch.randn(2, 4, 8, 128)
        out = proj(tensor)
        self.assertEqual(out.shape, (2, 4, 8, 64))

    def test_learned_projection_nonzero(self):
        proj = HeadDimProjection(
            teacher_head_dim=128, shell_head_dim=64,
            num_kv_heads=4, mode="learned"
        )
        tensor = torch.randn(2, 4, 8, 128) * 0.1
        out = proj(tensor)
        self.assertFalse(torch.allclose(out, torch.zeros_like(out), atol=1e-4))

    def test_project_kv_pair(self):
        proj = HeadDimProjection(teacher_head_dim=128, shell_head_dim=64, mode="pad_or_truncate")
        k = torch.randn(2, 4, 8, 128)
        v = torch.randn(2, 4, 8, 128)
        k_out, v_out = proj.project_kv_pair(k, v)
        self.assertEqual(k_out.shape, (2, 4, 8, 64))
        self.assertEqual(v_out.shape, (2, 4, 8, 64))


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Running ExoBrain test suite...")
    unittest.main(verbosity=2)
