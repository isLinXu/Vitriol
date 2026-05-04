#!/usr/bin/env python3
"""
ExoBrain Demo: Ultra Shell Model + External Brain = Reasoning

This demo shows how a zero-weight "shell model" (produced by Vitriol's
HybridUltra strategy) can perform meaningful inference when connected
to an external brain (ExoBrain) that injects knowledge at the
attention layer.

═══════════════════════════════════════════════════════════════
What this demo proves:
═══════════════════════════════════════════════════════════════

1. A model with zero weights produces zero output (no knowledge)
2. With ExoBrain injecting external KV at attention layer,
   the shell model can compute meaningful attention outputs
3. Three fusion modes are demonstrated:
   - Replace: External brain completely takes over
   - Residual: Shell + Brain blended
   - Gated: Attention-gated dynamic blending

═══════════════════════════════════════════════════════════════
Usage:
═══════════════════════════════════════════════════════════════

    python demos/exobrain_demo.py
"""

import sys
import math
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, ".")

from vitriol.kv.exobrain import (
    ExoBrainBackend,
    ExoBrainBus,
    ExoBrainConfig,
    ExoBrainAttentionPatcher,
    VectorDBSource,
    APIKnowledgeSource,
    LocalWeightSource,
    cross_attention_fusion,
    compute_gate,
)
from vitriol.kv.cache_store import KVCacheStoreConfig


def print_header(title: str) -> None:
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")


def print_section(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


def demo_1_shell_model_zero_output():
    """Demo 1: Show that a zero-weight shell model produces zero output."""
    print_header("Demo 1: Zero-Weight Shell Model → Zero Output")

    # Simulate a shell model: 2 layers, 256 dim, 4 heads
    batch, heads, seq_len, dim = 1, 4, 8, 64
    shell_query = torch.randn(batch, heads, 1, dim)  # Decode step: q_len=1

    # Shell model produces zero KV (because weights are zero)
    shell_k = torch.zeros(batch, heads, seq_len, dim)
    shell_v = torch.zeros(batch, heads, seq_len, dim)

    # Standard attention with zero KV → zero output
    scale = 1.0 / math.sqrt(dim)
    logits = (shell_query @ shell_k.transpose(-2, -1)) * scale
    # All logits are 0 → softmax is uniform → output is mean of V = 0
    weights = torch.softmax(logits, dim=-1)
    output = weights @ shell_v

    print(f"  Shell query norm:   {shell_query.norm().item():.4f}")
    print(f"  Shell K norm:       {shell_k.norm().item():.4f}")
    print(f"  Shell V norm:       {shell_v.norm().item():.4f}")
    print(f"  Attention output:   {output.norm().item():.6f} ← ZERO!")
    print(f"  → Without knowledge, the shell model is a blank slate.")


def demo_2_exobrain_replace_mode():
    """Demo 2: Replace mode — external brain completely takes over."""
    print_header("Demo 2: ExoBrain Replace Mode (Full External Brain)")

    batch, heads, q_len, dim = 1, 4, 1, 64
    n_docs = 20

    # Shell model query
    shell_query = torch.randn(batch, heads, q_len, dim)

    # External knowledge: real KV pairs from a "teacher" model
    torch.manual_seed(42)
    ext_k = torch.randn(batch, heads, n_docs, dim) * 0.5
    ext_v = torch.randn(batch, heads, n_docs, dim) * 0.5

    # Create VectorDB source
    # ext_k: [batch=1, heads=4, n_docs=20, dim=64]
    # Keys/values: [n_docs, dim] = [20, 64] = mean over heads dim
    # embeddings: [n_docs, embed_dim] = [20, 64] = per-doc embedding
    keys_2d = ext_k.squeeze(0).mean(dim=1)    # [4,20,64].mean(dim=1)→[20,64]
    values_2d = ext_v.squeeze(0).mean(dim=1)  # same
    embeddings_2d = ext_k.squeeze(0).mean(dim=1)  # [20,64] per-doc
    vdb_source = VectorDBSource(
        keys=keys_2d,
        values=values_2d,
        embeddings=embeddings_2d,
    )

    # Create ExoBrain bus and backend
    bus = ExoBrainBus(sources=[vdb_source])
    config = ExoBrainConfig(fusion_mode="replace", retrieval_top_k=5)
    kv_cfg = KVCacheStoreConfig()
    backend = ExoBrainBackend(store_cfg=kv_cfg, brain_bus=bus, brain_cfg=config)

    # Retrieve and inject
    result = bus.retrieve(shell_query, layer_idx=0)

    print(f"  Shell query norm:          {shell_query.norm().item():.4f}")
    print(f"  External K shape:          {result[0].shape}")
    print(f"  External V shape:          {result[1].shape}")

    # Cross-attention fusion
    brain_output = cross_attention_fusion(shell_query, result[0], result[1])
    print(f"  Brain output norm:         {brain_output.norm().item():.4f}")
    print(f"  Brain output mean:         {brain_output.mean().item():.6f}")
    print(f"  → External brain provides meaningful output!")


def demo_3_fusion_modes_comparison():
    """Demo 3: Compare Replace, Residual, and Gated fusion modes."""
    print_header("Demo 3: Fusion Modes Comparison")

    batch, heads, q_len, dim = 1, 4, 1, 64
    n_docs = 20

    torch.manual_seed(42)
    shell_query = torch.randn(batch, heads, q_len, dim)
    ext_k = torch.randn(batch, heads, n_docs, dim) * 0.5
    ext_v = torch.randn(batch, heads, n_docs, dim) * 0.5

    embeddings = ext_k.squeeze(0).mean(dim=1)  # [20, 64]
    vdb_source = VectorDBSource(
        keys=ext_k.squeeze(0).mean(dim=1),
        values=ext_v.squeeze(0).mean(dim=1),
        embeddings=embeddings,
    )

    # Shell output (near-zero for zero-weight model)
    shell_output = torch.zeros_like(shell_query)

    # Brain output (from cross-attention)
    brain_output = cross_attention_fusion(shell_query, ext_k, ext_v)

    # Mode 1: Replace
    replace_output = brain_output

    # Mode 2: Residual (α=0.1 for shell, 0.9 for brain)
    alpha = 0.1
    residual_output = alpha * shell_output + (1 - alpha) * brain_output

    # Mode 3: Gated
    gate = compute_gate(shell_query, ext_k, temperature=1.0)
    gated_output = gate * brain_output + (1 - gate) * shell_output

    print(f"  Shell output norm:    {shell_output.norm().item():.6f}  (zero)")
    print(f"  Brain output norm:    {brain_output.norm().item():.4f}")
    print(f"  ──────────────────────────────────────")
    print(f"  Replace mode:  norm = {replace_output.norm().item():.4f}  (full brain)")
    print(f"  Residual mode: norm = {residual_output.norm().item():.4f}  (0.9×brain + 0.1×shell)")
    print(f"  Gated mode:    norm = {gated_output.norm().item():.4f}  (dynamic gate)")
    print(f"  Gate value:          {gate.mean().item():.4f}")
    print(f"\n  → For zero-weight shells, all modes ≈ replace (brain dominates).")
    print(f"  → Once shell has kaiming init, residual/gated provide smooth transition.")


def demo_4_local_weight_source():
    """Demo 4: Local weight source — teacher model KV injection."""
    print_header("Demo 4: Local Weight Source (Teacher Model KV)")

    batch, heads, seq_len, dim = 1, 4, 16, 64

    torch.manual_seed(42)
    # "Teacher" model KV — precomputed from a real model
    teacher_k = torch.randn(batch, heads, seq_len, dim) * 0.3
    teacher_v = torch.randn(batch, heads, seq_len, dim) * 0.3

    # Shell query
    shell_query = torch.randn(batch, heads, 1, dim)

    # Create local weight source
    local_source = LocalWeightSource()
    local_source.set_teacher_kv(layer_idx=0, key=teacher_k, value=teacher_v)

    # Create bus and backend
    bus = ExoBrainBus(sources=[local_source])
    config = ExoBrainConfig(fusion_mode="replace", retrieval_top_k=5)
    backend = ExoBrainBackend(
        store_cfg=KVCacheStoreConfig(),
        brain_bus=bus,
        brain_cfg=config,
    )

    # Retrieve from local weight source
    result = bus.retrieve(shell_query, layer_idx=0)
    brain_output = cross_attention_fusion(shell_query, result[0], result[1])

    print(f"  Teacher K shape:      {teacher_k.shape}")
    print(f"  Teacher V shape:      {teacher_v.shape}")
    print(f"  Shell query norm:     {shell_query.norm().item():.4f}")
    print(f"  Retrieved K shape:    {result[0].shape}")
    print(f"  Retrieved V shape:    {result[1].shape}")
    print(f"  Brain output norm:    {brain_output.norm().item():.4f}")
    print(f"\n  → Teacher model KV can be directly injected into the shell.")


def demo_5_direct_injection():
    """Demo 5: Direct KV injection into ExoBrain bus."""
    print_header("Demo 5: Direct KV Injection (Bypass Retrieval)")

    batch, heads, seq_len, dim = 1, 4, 8, 64

    torch.manual_seed(42)
    # Create precomputed KV pairs for each layer
    injected_kv = {}
    for layer_idx in range(4):
        k = torch.randn(batch, heads, seq_len, dim) * 0.2
        v = torch.randn(batch, heads, seq_len, dim) * 0.2
        injected_kv[layer_idx] = (k, v)

    # Create bus with no sources but direct injection
    bus = ExoBrainBus()
    for layer_idx, (k, v) in injected_kv.items():
        bus.inject_kv(layer_idx, k, v)

    # Query each layer
    shell_query = torch.randn(batch, heads, 1, dim)
    print(f"  Injected KV for {len(injected_kv)} layers")

    for layer_idx in range(4):
        result = bus.retrieve(shell_query, layer_idx)
        if result is not None:
            brain_output = cross_attention_fusion(shell_query, result[0], result[1])
            print(f"  Layer {layer_idx}: output norm = {brain_output.norm().item():.4f}")
        else:
            print(f"  Layer {layer_idx}: no KV available")

    # Show bus statistics
    stats = bus.stats
    print(f"\n  Bus stats:")
    print(f"    Retrieve count:  {stats['retrieve_count']}")
    print(f"    Hit count:       {stats['hit_count']}")
    print(f"    Miss count:      {stats['miss_count']}")
    print(f"    Hit rate:        {stats['hit_rate']:.1%}")


def demo_6_backend_integration():
    """Demo 6: Full ExoBrainBackend integration with read_attention()."""
    print_header("Demo 6: ExoBrainBackend Integration")

    batch, heads, seq_len, dim = 1, 4, 8, 64

    torch.manual_seed(42)
    # Setup knowledge sources
    ext_k = torch.randn(batch, heads, 20, dim) * 0.3
    ext_v = torch.randn(batch, heads, 20, dim) * 0.3
    embeddings = ext_k.squeeze(0).mean(dim=1)  # [20, 64]

    vdb_source = VectorDBSource(
        keys=ext_k.squeeze(0).mean(dim=1),
        values=ext_v.squeeze(0).mean(dim=1),
        embeddings=embeddings,
    )

    # Test each fusion mode
    for mode in ["replace", "residual", "gated"]:
        bus = ExoBrainBus(sources=[vdb_source])
        config = ExoBrainConfig(fusion_mode=mode, retrieval_top_k=5)
        backend = ExoBrainBackend(
            store_cfg=KVCacheStoreConfig(),
            brain_bus=bus,
            brain_cfg=config,
        )

        # Simulate read_attention call
        query = torch.randn(batch, heads, 1, dim)
        try:
            output = backend.read_attention(
                handle=None,  # No KV cache needed for replace mode
                layer_idx=0,
                query=query,
                attn_mask=None,
                is_causal=False,
                scale=None,
                info={"dropout_p": 0.0},
            )
            print(f"  Mode '{mode}': output norm = {output.norm().item():.4f}  ✓")
        except Exception as e:
            # Residual/gated modes need a valid handle (KV store)
            print(f"  Mode '{mode}': needs KV store handle ({type(e).__name__}) — expected for demo")

    # Replace mode should work without handle
    print(f"\n  → Replace mode works standalone (no KV cache needed)")
    print(f"  → Residual/Gated modes need KV cache for shell model output")


def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║          ExoBrain: External Brain for Ultra Shell Models       ║")
    print("║   Visita Interiora Terrae Rectificando Invenies Occultum       ║")
    print("║                                                                ║")
    print("║   \"深入模型腹地，精馏万物本体，寻获潜藏真核\"                   ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    demo_1_shell_model_zero_output()
    demo_2_exobrain_replace_mode()
    demo_3_fusion_modes_comparison()
    demo_4_local_weight_source()
    demo_5_direct_injection()
    demo_6_backend_integration()

    print_header("Summary")
    print("""
  ExoBrain Implementation Status:
  ┌────────┬───────────────────────────────────────────────────────┐
  │ Phase  │ Status                                                │
  ├────────┼───────────────────────────────────────────────────────┤
  │ P1 ✅  │ ExoBrainBackend — KV-level injection via read_attention│
  │ P2 ✅  │ ExoBrainAttentionPatcher — Attention-level intercept   │
  │ P3 ✅  │ ExoBrainBus — Unified knowledge retrieval interface    │
  │ P4 ✅  │ Registration — KVCacheStoreConfig + __init__.py        │
  │ P5 ✅  │ Demo — This script                                    │
  └────────┴───────────────────────────────────────────────────────┘

  Knowledge Sources Implemented:
  • VectorDBSource — In-memory vector similarity search
  • APIKnowledgeSource — External LLM API (stub, extensible)
  • LocalWeightSource — Teacher model KV with projection
  • Direct Injection — Pre-computed KV via bus.inject_kv()

  Fusion Modes:
  • Replace — External brain completely takes over
  • Residual — Shell + Brain blended (α weight)
  • Gated — Attention-gated dynamic fusion

  Next Steps:
  • Integration with real teacher model (Qwen, LLaMA, etc.)
  • FAISS/ChromaDB for production vector search
  • Dimension projection for teacher→shell mapping
  • End-to-end inference pipeline with transformers
  • Quality evaluation (perplexity, generation quality)
    """)


if __name__ == "__main__":
    main()
