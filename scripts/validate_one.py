#!/usr/bin/env python3
"""Validate a single Vitriol-generated model. Called by batch runner."""
import sys
import json
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

MODEL_DIR = Path(sys.argv[1])

result = {
    "name": MODEL_DIR.name,
    "path": str(MODEL_DIR),
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
    "arch_viz": None,
    "errors": [],
}

files = list(MODEL_DIR.iterdir())
result["files"] = sorted(f.name for f in files if f.is_file())
result["size_mb"] = sum(f.stat().st_size for f in files if f.is_file()) / 1024 / 1024

result["has_config"] = (MODEL_DIR / "config.json").exists()
result["has_tokenizer"] = (MODEL_DIR / "tokenizer.json").exists() or (MODEL_DIR / "tokenizer.model").exists()
result["has_weights"] = any(
    f.name.startswith("pytorch_model") or (f.name.startswith("model") and f.suffix in (".bin", ".safetensors"))
    for f in files if f.is_file()
)
result["has_viz"] = any(
    f.name in ("architecture.html", "architecture.png", "arch_viz.svg", "_smoke_arch.html")
    for f in files if f.is_file()
)

# Config
if result["has_config"]:
    try:
        from transformers import AutoConfig
        t0 = time.time()
        config = AutoConfig.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
        result["config_load"] = f"OK ({time.time()-t0:.2f}s)"
        result["config_type"] = getattr(config, "model_type", "unknown")
        result["hidden_size"] = getattr(config, "hidden_size", None)
        result["num_layers"] = getattr(config, "num_hidden_layers", None)
    except Exception as e:
        result["config_load"] = f"FAIL: {e}"
        result["errors"].append(f"config: {e}")
else:
    result["config_load"] = "SKIP (no config.json)"

# Tokenizer
if result["has_tokenizer"]:
    try:
        from transformers import AutoTokenizer
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
        result["tokenizer_load"] = f"OK ({time.time()-t0:.2f}s)"
        result["vocab_size"] = len(tok)
        ids = tok.encode("Hello World")
        decoded = tok.decode(ids, skip_special_tokens=True)
        result["tokenizer_roundtrip"] = "OK" if "Hello" in decoded else f"PARTIAL: '{decoded}'"
    except Exception as e:
        result["tokenizer_load"] = f"FAIL: {e}"
        result["errors"].append(f"tokenizer: {e}")
else:
    result["tokenizer_load"] = "SKIP (no tokenizer)"

# Model load + inference
if result["has_config"] and result["config_load"] and result["config_load"].startswith("OK"):
    try:
        from transformers import AutoModelForCausalLM
        import torch
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR), trust_remote_code=True, torch_dtype=torch.float32, device_map="cpu",
        )
        load_time = time.time() - t0
        n_params = sum(p.numel() for p in model.parameters())
        result["model_load"] = f"OK ({load_time:.2f}s, {n_params/1e6:.1f}M params)"
        result["n_params_M"] = n_params / 1e6

        # Inference
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            inputs = tok.encode("Hello", return_tensors="pt")
            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(inputs, max_new_tokens=3, do_sample=False)
            result["inference"] = f"OK ({time.time()-t0:.2f}s)"
        except Exception as e:
            result["inference"] = f"FAIL: {e}"
            result["errors"].append(f"inference: {e}")

        del model
    except Exception as e:
        result["model_load"] = f"FAIL: {e}"
        result["errors"].append(f"model: {e}")
else:
    result["model_load"] = "SKIP"

print(json.dumps(result, indent=2, default=str))
