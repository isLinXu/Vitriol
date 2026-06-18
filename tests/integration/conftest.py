"""Shared fixtures for integration tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def tiny_llama_model_dir(tmp_path: Path) -> Path:
    """Minimal local model directory suitable for offline golden-path tests."""
    model_dir = tmp_path / "TinyLlama"
    model_dir.mkdir()
    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "vocab_size": 128,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "intermediate_size": 128,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
    }
    (model_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (model_dir / "meta-config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return model_dir
