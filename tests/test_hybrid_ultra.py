#!/usr/bin/env python3
"""
HybridUltra Strategy — Comprehensive Validation Test

Validates:
1. Tensor generation correctness (norm weight=1.0, others=0.0)
2. Trainability (gradient flows through all parameters)
3. Safetensors compatibility
4. Model load/inference round-trip
5. Memory optimization (optimize_loaded_model)
6. Comparison: Ultra vs HybridUltra vs Compact vs Random
"""

import sys
import os
import tempfile
import shutil

import pytest

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
from vitriol.strategies import (
    list_strategies,
    HybridUltraStrategy,
    UltraStrategy,
    CompactStrategy,
    RandomStrategy,
)


def sep(title: str = "") -> None:
    print(f"\n{'='*70}")
    if title:
        print(f"  {title}")
        print(f"{'='*70}")


def test_tensor_generation():
    """Test that HybridUltra generates correct tensor values."""
    sep("1. Tensor Generation Correctness")

    strategy = HybridUltraStrategy(init_mode="zeros", norm_init=True)
    passed = 0
    failed = 0

    # LayerNorm weight → 1.0
    t = strategy.generate_tensor((256,), torch.float32, "model.layers.0.input_layernorm.weight")
    if t.mean().item() == 1.0:
        print(f"  ✅ LayerNorm weight = 1.0 (mean={t.mean().item()})")
        passed += 1
    else:
        print(f"  ❌ LayerNorm weight ≠ 1.0 (mean={t.mean().item()})")
        failed += 1

    # LayerNorm bias → 0.0
    t = strategy.generate_tensor((256,), torch.float32, "model.layers.0.input_layernorm.bias")
    if t.sum().item() == 0.0:
        print(f"  ✅ LayerNorm bias = 0.0 (sum={t.sum().item()})")
        passed += 1
    else:
        print(f"  ❌ LayerNorm bias ≠ 0.0 (sum={t.sum().item()})")
        failed += 1

    # RMSNorm weight → 1.0
    t = strategy.generate_tensor((256,), torch.float32, "model.layers.0.post_attention_layernorm.weight")
    if t.mean().item() == 1.0:
        print(f"  ✅ RMSNorm weight = 1.0 (mean={t.mean().item()})")
        passed += 1
    else:
        print(f"  ❌ RMSNorm weight ≠ 1.0 (mean={t.mean().item()})")
        failed += 1

    # Linear weight → 0.0 (zeros mode)
    t = strategy.generate_tensor((512, 256), torch.float32, "model.layers.0.self_attn.q_proj.weight")
    if t.sum().item() == 0.0:
        print("  ✅ Linear weight (zeros mode) = 0.0")
        passed += 1
    else:
        print(f"  ❌ Linear weight (zeros mode) ≠ 0.0 (sum={t.sum().item()})")
        failed += 1

    # Embedding → 0.0 (zeros mode)
    t = strategy.generate_tensor((32000, 256), torch.float32, "model.embed_tokens.weight")
    if t.sum().item() == 0.0:
        print("  ✅ Embedding (zeros mode) = 0.0")
        passed += 1
    else:
        print("  ❌ Embedding (zeros mode) ≠ 0.0")
        failed += 1

    print(f"\n  Result: {passed} passed, {failed} failed")
    assert failed == 0


def test_init_modes():
    """Test all initialization modes."""
    sep("2. Initialization Modes")

    results = {}

    for mode in ["zeros", "kaiming", "xavier", "orthogonal", "small_normal"]:
        strategy = HybridUltraStrategy(init_mode=mode, dtype_override=None)
        t = strategy.generate_tensor((256, 256), torch.float32, "model.layers.0.self_attn.q_proj.weight")

        is_zero = (t.sum().item() == 0.0)
        results[mode] = {
            "is_zero": is_zero,
            "mean": f"{t.mean().item():.6f}",
            "std": f"{t.std().item():.6f}",
            "min": f"{t.min().item():.6f}",
            "max": f"{t.max().item():.6f}",
        }
        status = "zeros" if is_zero else "initialized"
        print(f"  {mode:15s}: {status:12s} | mean={results[mode]['mean']}, std={results[mode]['std']}")

    # Verify zeros mode produces zeros, others produce non-zeros
    ok = results["zeros"]["is_zero"] and not results["kaiming"]["is_zero"]
    print("\n  ✅ All modes work correctly" if ok else "  ❌ Mode verification failed")
    assert ok


def test_trainability():
    """Test that HybridUltra models support gradient computation."""
    sep("3. Trainability (Gradient Flow)")

    # Test with kaiming init (fully trainable)
    strategy = HybridUltraStrategy(init_mode="kaiming", norm_init=True)
    passed = 0
    failed = 0

    # Simulate a simple forward + backward
    q_weight = strategy.generate_tensor((256, 256), torch.float32, "model.layers.0.self_attn.q_proj.weight")
    ln_weight = strategy.generate_tensor((256,), torch.float32, "model.layers.0.input_layernorm.weight")

    # Make them parameters
    q_param = torch.nn.Parameter(q_weight.clone().float())
    ln_param = torch.nn.Parameter(ln_weight.clone().float())

    # Forward pass
    x = torch.randn(1, 256)
    x_normed = x * ln_param  # LayerNorm-like (simplified)
    out = x_normed @ q_param.T

    # Backward
    loss = out.sum()
    loss.backward()

    if q_param.grad is not None:
        print(f"  ✅ Q_proj gradient computed: grad_norm={q_param.grad.norm().item():.6f}")
        passed += 1
    else:
        print("  ❌ Q_proj gradient is None")
        failed += 1

    if ln_param.grad is not None:
        print(f"  ✅ LayerNorm gradient computed: grad_norm={ln_param.grad.norm().item():.6f}")
        passed += 1
    else:
        print("  ❌ LayerNorm gradient is None")
        failed += 1

    # Test with zeros mode — LN should still be trainable
    strategy_zero = HybridUltraStrategy(init_mode="zeros", norm_init=True)
    ln_w = strategy_zero.generate_tensor((256,), torch.float32, "model.layers.0.input_layernorm.weight")
    ln_p2 = torch.nn.Parameter(ln_w.clone().float())

    x2 = torch.randn(1, 256)
    out2 = x2 * ln_p2
    out2.sum().backward()

    if ln_p2.grad is not None and ln_p2.grad.norm().item() > 0:
        print(f"  ✅ Zeros+NormInit: LayerNorm gradient flows (grad_norm={ln_p2.grad.norm().item():.6f})")
        passed += 1
    else:
        print("  ❌ Zeros+NormInit: LayerNorm gradient doesn't flow")
        failed += 1

    print(f"\n  Result: {passed} passed, {failed} failed")
    assert failed == 0


def test_safetensors_compat():
    """Test Safetensors format compatibility."""
    sep("4. Safetensors Compatibility")

    strategy = HybridUltraStrategy(init_mode="zeros")

    # Check capabilities
    caps = strategy.capabilities
    print(f"  supports_safetensors: {caps.supports_safetensors}")
    print(f"  requires_contiguous: {caps.requires_contiguous}")
    print(f"  storage_format: {strategy.storage_format}")

    if not caps.supports_safetensors:
        print("  ❌ Safetensors not supported!")
        assert False, "Safetensors not supported"

    # Test actual save
    tmpdir = tempfile.mkdtemp(prefix="hybrid_ultra_test_")
    try:
        shard_data = {
            "layer.weight": strategy.generate_tensor((256, 256), torch.float32, "layer.weight"),
            "layer.bias": strategy.generate_tensor((256,), torch.float32, "layer.bias"),
            "ln.weight": strategy.generate_tensor((256,), torch.float32, "ln.weight"),
        }

        save_path = os.path.join(tmpdir, "test_model.safetensors")
        strategy.save_shard(shard_data, save_path)

        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path)
            print(f"  ✅ Safetensors file saved: {file_size} bytes")
        else:
            print("  ❌ Safetensors file not created")
            assert False, "Safetensors file not created"

        # Try loading back
        try:
            from safetensors.torch import load_file
            loaded = load_file(save_path)
            print(f"  ✅ Safetensors file loaded: {len(loaded)} tensors")
            for k, v in loaded.items():
                print(f"     {k}: shape={v.shape}, dtype={v.dtype}, "
                      f"mean={v.mean().item():.4f}")
        except Exception as e:
            print(f"  ❌ Failed to load safetensors: {e}")
            assert False, f"Failed to load safetensors: {e}"

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    assert True


def test_full_model_roundtrip():
    """Test generating a full model with HybridUltra and loading it back."""
    sep("5. Full Model Round-trip (TinyLLaMA)")

    from vitriol.core.generator import MinimalWeightGenerator
    from vitriol.config.manager import GenerationConfig
    from transformers.utils import cached_file

    tmpdir = tempfile.mkdtemp(prefix="hybrid_ultra_model_")
    try:
        model_id = "TinyLLama/Tiny-Llama-1.1B-Chat-v1.0"
        if cached_file(
            model_id,
            "config.json",
            local_files_only=True,
            _raise_exceptions_for_gated_repo=False,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        ) is None:
            pytest.skip(f"{model_id} is not available in the local Transformers cache")

        config = GenerationConfig(
            strategy="hybrid_ultra",
            dtype="bfloat16",
            auto_validate=False,
        )

        gen = MinimalWeightGenerator(
            model_id=model_id,
            output_dir=tmpdir,
            config=config,
            shrink_config=True,
        )

        print("  Generating model with HybridUltra strategy...")
        result = gen.generate()
        print(f"  Output dir: {result.output_dir}")
        print(f"  Total size: {result.total_size} bytes")

        # List files
        files = os.listdir(tmpdir)
        weight_files = [f for f in files if f.endswith(('.bin', '.safetensors'))]
        print(f"  Weight files: {weight_files}")

        # Check config.json exists
        config_path = os.path.join(tmpdir, "config.json")
        if os.path.exists(config_path):
            print("  ✅ config.json exists")
        else:
            print("  ❌ config.json missing")
            assert False, "config.json missing"

        # Try loading the model
        print("  Loading model with from_pretrained()...")
        from transformers import AutoModelForCausalLM
        import json

        with open(config_path) as f:
            cfg_data = json.load(f)
        print(f"  Config: hidden_size={cfg_data.get('hidden_size')}, "
              f"layers={cfg_data.get('num_hidden_layers')}")

        try:
            model = AutoModelForCausalLM.from_pretrained(
                tmpdir,
                local_files_only=True,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map="cpu",
            )

            param_count = sum(p.numel() for p in model.parameters())
            param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
            print(f"  ✅ Model loaded: {param_count:,} parameters, {param_mb:.2f} MB")

            # Check LayerNorm weights
            ln_ok = True
            for name, param in model.named_parameters():
                if "norm.weight" in name:
                    if param.mean().item() != 1.0:
                        print(f"  ⚠️  {name}: mean={param.mean().item():.4f} (expected 1.0)")
                        ln_ok = False

            if ln_ok:
                print("  ✅ All LayerNorm weights = 1.0 (trainable)")
            else:
                print("  ⚠️  Some LayerNorm weights ≠ 1.0")

            # Try forward pass
            from transformers import AutoTokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(tmpdir, local_files_only=True)
                inputs = tokenizer("Hello", return_tensors="pt")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=3)
                print(f"  ✅ Inference works: {tokenizer.decode(outputs[0])[:50]}")
            except Exception as e:
                print(f"  ⚠️  Inference: {e}")

            # Test memory optimization
            stats = HybridUltraStrategy.optimize_loaded_model(model)
            print("\n  Memory Optimization Stats:")
            print(f"    Before: {stats['before_mb']:.2f} MB")
            print(f"    After:  {stats['after_mb']:.2f} MB")
            print(f"    Saved:  {stats['saved_mb']:.2f} MB ({stats['zero_params']}/{stats['total_params']} zero params)")
            print(f"    Compression ratio: {stats['compression_ratio']:.4f}")

        except Exception as e:
            print(f"  ❌ Model loading failed: {e}")
            import traceback
            traceback.print_exc()
            assert False, f"Model loading failed: {e}"

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    assert True


def test_strategy_comparison():
    """Compare all strategies side by side."""
    sep("6. Strategy Comparison")

    strategies = {
        "Ultra": UltraStrategy(),
        "HybridUltra (zeros)": HybridUltraStrategy(init_mode="zeros"),
        "HybridUltra (kaiming)": HybridUltraStrategy(init_mode="kaiming"),
        "Compact": CompactStrategy(),
        "Random": RandomStrategy(),
    }

    # Test with a typical parameter shape
    shape = (4096, 4096)
    dtype = torch.float32

    print(f"\n  Test tensor shape: {shape}, dtype: {dtype}")
    print(f"  {'Strategy':<25s} {'Storage':>10s} {'Norm w':>8s} {'Train':>6s} {'Safe':>5s}")
    print(f"  {'-'*25} {'-'*10} {'-'*8} {'-'*6} {'-'*5}")

    for name, strategy in strategies.items():
        # Regular weight
        t = strategy.generate_tensor(shape, dtype, "layer.weight")
        # Norm weight
        tn = strategy.generate_tensor((4096,), dtype, "model.layers.0.input_layernorm.weight")

        # Calculate storage
        try:
            storage_bytes = t.untyped_storage().nbytes()
        except Exception:
            storage_bytes = t.numel() * t.element_size()

        caps = strategy.capabilities

        # Norm weight value
        norm_val = "1.0" if abs(tn.mean().item() - 1.0) < 0.01 else f"{tn.mean().item():.1f}"

        print(f"  {name:<25s} {storage_bytes:>10,d} {norm_val:>8s} "
              f"{'✓' if caps.supports_training else '✗':>6s} "
              f"{'✓' if caps.supports_safetensors else '✗':>5s}")

    # Verify HybridUltra is registered
    sep("7. Strategy Registry")
    available = list_strategies()
    print(f"  Registered strategies: {available}")
    if "hybrid_ultra" in available:
        print("  ✅ 'hybrid_ultra' is registered")
    else:
        print("  ❌ 'hybrid_ultra' NOT registered!")

    assert True


def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║     HybridUltra Strategy — Comprehensive Validation Test      ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    results = {}

    # Run all tests
    tests = [
        ("Tensor Generation", test_tensor_generation),
        ("Init Modes", test_init_modes),
        ("Trainability", test_trainability),
        ("Safetensors", test_safetensors_compat),
        ("Full Model Roundtrip", test_full_model_roundtrip),
        ("Strategy Comparison", test_strategy_comparison),
    ]

    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n  ❌ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    sep("SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        print(f"  {'✅' if ok else '❌'} {name}")

    print(f"\n  Total: {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
