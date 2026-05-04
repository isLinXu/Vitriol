#!/usr/bin/env python3
"""Batch validation of all Vitriol-generated models.

Tests: config load, tokenizer load, model load, inference, arch-viz.
"""
import os
import sys
import json
import time
import traceback
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

OUTPUT_DIR = Path(__file__).parent.parent / "output"

# Models to validate (skip empty dirs)
SKIP_DIRS = {"model", "llama-3.1-8b-Vitriol-ultra-dummy"}  # empty dirs

results = {}


def validate_model(model_dir: Path) -> dict:
    """Validate a single model directory."""
    name = model_dir.name
    result = {
        "name": name,
        "path": str(model_dir),
        "size_mb": 0,
        "files": [],
        "has_config": False,
        "has_tokenizer": False,
        "has_weights": False,
        "has_viz": False,
        "config_load": None,
        "tokenizer_load": None,
        "model_load": None,
        "inference": None,
        "errors": [],
    }

    # Check files
    files = list(model_dir.iterdir())
    result["files"] = [f.name for f in files]
    result["size_mb"] = sum(f.stat().st_size for f in files if f.is_file()) / 1024 / 1024

    # Check key files
    result["has_config"] = (model_dir / "config.json").exists()
    result["has_tokenizer"] = (model_dir / "tokenizer.json").exists() or (model_dir / "tokenizer.model").exists()
    result["has_weights"] = any(
        f.name.startswith("pytorch_model") or f.name.startswith("model")
        for f in files
        if f.is_file()
    )
    result["has_viz"] = any(
        f.name in ("architecture.html", "architecture.png", "arch_viz.svg")
        for f in files
        if f.is_file()
    )

    # Try config load
    if result["has_config"]:
        try:
            from transformers import AutoConfig
            t0 = time.time()
            config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
            result["config_load"] = f"OK ({time.time()-t0:.2f}s)"
            result["config_type"] = getattr(config, "model_type", "unknown")
            result["hidden_size"] = getattr(config, "hidden_size", None)
            result["num_layers"] = getattr(config, "num_hidden_layers", None)
        except Exception as e:
            result["config_load"] = f"FAIL: {e}"
            result["errors"].append(f"config_load: {e}")
    else:
        result["config_load"] = "SKIP (no config.json)"

    # Try tokenizer load
    if result["has_tokenizer"]:
        try:
            from transformers import AutoTokenizer
            t0 = time.time()
            tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
            result["tokenizer_load"] = f"OK ({time.time()-t0:.2f}s)"
            result["vocab_size"] = len(tok)
            # Test encode/decode
            text = "Hello World"
            ids = tok.encode(text)
            decoded = tok.decode(ids, skip_special_tokens=True)
            result["tokenizer_roundtrip"] = "OK" if "Hello" in decoded else f"PARTIAL: '{decoded}'"
        except Exception as e:
            result["tokenizer_load"] = f"FAIL: {e}"
            result["errors"].append(f"tokenizer_load: {e}")
    else:
        result["tokenizer_load"] = "SKIP (no tokenizer)"

    # Try model load (only if config loaded OK)
    if result["has_config"] and result["config_load"] and result["config_load"].startswith("OK"):
        try:
            from transformers import AutoModelForCausalLM
            import torch
            t0 = time.time()
            model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map="cpu",
            )
            load_time = time.time() - t0
            n_params = sum(p.numel() for p in model.parameters())
            result["model_load"] = f"OK ({load_time:.2f}s, {n_params/1e6:.1f}M params)"
            result["n_params_M"] = n_params / 1e6

            # Try inference
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                inputs = tok.encode("Hello", return_tensors="pt")
                t0 = time.time()
                with torch.no_grad():
                    outputs = model.generate(inputs, max_new_tokens=3, do_sample=False)
                gen_time = time.time() - t0
                result["inference"] = f"OK ({gen_time:.2f}s)"
            except Exception as e:
                result["inference"] = f"FAIL: {e}"
                result["errors"].append(f"inference: {e}")

            # Free memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            result["model_load"] = f"FAIL: {e}"
            result["errors"].append(f"model_load: {e}")
    else:
        result["model_load"] = "SKIP"

    return result


def main():
    print("=" * 80)
    print("Vitriol Batch Model Validation")
    print("=" * 80)

    model_dirs = sorted(OUTPUT_DIR.iterdir())
    model_dirs = [d for d in model_dirs if d.is_dir() and d.name not in SKIP_DIRS and not d.name.startswith(".")]

    print(f"\nFound {len(model_dirs)} model directories to validate:\n")
    for d in model_dirs:
        print(f"  - {d.name}/")
    print()

    all_results = []
    for i, model_dir in enumerate(model_dirs):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(model_dirs)}] Validating: {model_dir.name}")
        print(f"{'='*60}")

        try:
            result = validate_model(model_dir)
            all_results.append(result)

            # Print summary
            print(f"  Size: {result['size_mb']:.1f} MB")
            print(f"  Files: {len(result['files'])}")
            print(f"  Config: {result['config_load']}")
            print(f"  Tokenizer: {result['tokenizer_load']}")
            print(f"  Model: {result['model_load']}")
            print(f"  Inference: {result['inference']}")
            print(f"  Viz: {'✅' if result['has_viz'] else '❌'}")
            if result['errors']:
                for err in result['errors']:
                    print(f"  ⚠️  {err[:100]}")
        except Exception as e:
            print(f"  💥 UNEXPECTED ERROR: {e}")
            traceback.print_exc()
            all_results.append({"name": model_dir.name, "errors": [str(e)]})

    # Summary table
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"{'Model':<40} {'Config':<12} {'Tokenizer':<12} {'Model':<12} {'Inference':<12} {'Viz':<5}")
    print("-" * 93)

    for r in all_results:
        name = r['name'][:39]
        cfg = "✅" if r.get('config_load', '').startswith("OK") else ("❌" if "FAIL" in str(r.get('config_load', '')) else "⏭️")
        tok = "✅" if r.get('tokenizer_load', '').startswith("OK") else ("❌" if "FAIL" in str(r.get('tokenizer_load', '')) else "⏭️")
        mdl = "✅" if r.get('model_load', '').startswith("OK") else ("❌" if "FAIL" in str(r.get('model_load', '')) else "⏭️")
        inf = "✅" if r.get('inference', '').startswith("OK") else ("❌" if r.get('inference') and "FAIL" in str(r.get('inference', '')) else "⏭️")
        viz = "✅" if r.get('has_viz') else "❌"
        print(f"{name:<40} {cfg:<12} {tok:<12} {mdl:<12} {inf:<12} {viz:<5}")

    # Count
    ok_count = sum(1 for r in all_results if r.get('model_load', '').startswith("OK"))
    total = len(all_results)
    print(f"\n✅ Models fully loaded: {ok_count}/{total}")

    # Save results
    results_path = OUTPUT_DIR / "batch_validation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {results_path}")


if __name__ == "__main__":
    main()
