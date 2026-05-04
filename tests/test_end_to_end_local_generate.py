import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

from vitriol.config.manager import GenerationConfig, SecurityOptions
from vitriol.core.generator import MinimalWeightGenerator


def test_end_to_end_local_ultra_generate_is_self_consistent(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    out_dir = tmp_path / "out"
    model_dir.mkdir()
    out_dir.mkdir()

    raw_cfg = {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "tie_word_embeddings": False,
        "vocab_size": 1000,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 256,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 64,
        "shared_expert_intermediate_size": 64,
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "vocab_size": 1000,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 64,
            "shared_expert_intermediate_size": 64,
            "layer_types": ["full_attention", "full_attention"],
            "tie_word_embeddings": False,
        },
    }
    (model_dir / "config.json").write_text(json.dumps(raw_cfg, indent=2, ensure_ascii=False))

    (model_dir / "pytorch_model.bin.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {
                    "_dummy_0": "pytorch_model-00001-of-00002.bin",
                    "_dummy_1": "pytorch_model-00002-of-00002.bin",
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    cfg = GenerationConfig(strategy="ultra")
    cfg.security = SecurityOptions(
        trust_remote_code=False,
        allow_network=False,
        local_files_only=True,
    )

    g = MinimalWeightGenerator(
        model_id=str(model_dir),
        output_dir=str(out_dir),
        config=cfg,
        shrink_config=True,
    )
    g.generate()

    assert (out_dir / "config.json").exists()
    assert (out_dir / "meta-config.json").exists()
    assert (out_dir / "pytorch_model.bin.index.json").exists()
    assert (out_dir / "vitriol-manifest.json").exists()

    meta = json.loads((out_dir / "meta-config.json").read_text())
    assert meta == raw_cfg

    manifest = json.loads((out_dir / "vitriol-manifest.json").read_text())
    assert manifest.get("source", {}).get("meta_config_equals_source_config") is True
    assert manifest.get("artifacts", {}).get("reconcile", {}).get("sha256")
    assert manifest.get("loadability", {}).get("checked") is True

    out_cfg = json.loads((out_dir / "config.json").read_text())
    assert out_cfg.get("model_type") == "qwen3_5_moe"
    assert int(out_cfg.get("num_hidden_layers") or 0) > 0
    assert int(out_cfg.get("num_attention_heads") or 0) > 0

    assert (out_dir / "architecture.html").exists()
    assert not any(p.suffix == ".py" for p in out_dir.rglob("*.py"))


def test_end_to_end_local_glm_ultra_generate_runs_forward(tmp_path: Path) -> None:
    model_dir = tmp_path / "glm-model"
    out_dir = tmp_path / "glm-out"
    model_dir.mkdir()
    out_dir.mkdir()

    raw_cfg = {
        "model_type": "glm_moe_dsa",
        "architectures": ["GlmMoeDsaForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 0,
        "eos_token_id": [1],
        "hidden_act": "silu",
        "hidden_size": 6144,
        "intermediate_size": 12288,
        "kv_lora_rank": 512,
        "max_position_embeddings": 4096,
        "mlp_layer_types": ["dense", "dense", "dense", "sparse"],
        "moe_intermediate_size": 2048,
        "moe_layer_freq": 1,
        "n_group": 1,
        "n_routed_experts": 16,
        "n_shared_experts": 1,
        "norm_topk_prob": True,
        "num_attention_heads": 64,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 4,
        "num_key_value_heads": 64,
        "pad_token_id": 0,
        "pretraining_tp": 1,
        "q_lora_rank": 2048,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
        "routed_scaling_factor": 2.5,
        "scoring_func": "sigmoid",
        "tie_word_embeddings": False,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "use_cache": True,
        "v_head_dim": 256,
        "vocab_size": 1024,
    }
    (model_dir / "config.json").write_text(json.dumps(raw_cfg, indent=2, ensure_ascii=False))

    (model_dir / "pytorch_model.bin.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {
                    "_dummy_0": "pytorch_model-00001-of-00002.bin",
                    "_dummy_1": "pytorch_model-00002-of-00002.bin",
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    cfg = GenerationConfig(strategy="ultra")
    cfg.security = SecurityOptions(
        trust_remote_code=False,
        allow_network=False,
        local_files_only=True,
    )

    g = MinimalWeightGenerator(
        model_id=str(model_dir),
        output_dir=str(out_dir),
        config=cfg,
        shrink_config=True,
    )
    g.generate()

    out_cfg = json.loads((out_dir / "config.json").read_text())
    assert out_cfg["qk_nope_head_dim"] + out_cfg["qk_rope_head_dim"] == out_cfg["qk_head_dim"]

    model = AutoModelForCausalLM.from_pretrained(
        str(out_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    with torch.no_grad():
        outputs = model(input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long))
    assert tuple(outputs.logits.shape[:2]) == (1, 3)


def test_end_to_end_local_t5_ultra_generate_loads_seq2seq(tmp_path: Path) -> None:
    model_dir = tmp_path / "t5-model"
    out_dir = tmp_path / "t5-out"
    model_dir.mkdir()
    out_dir.mkdir()

    raw_cfg = {
        "model_type": "t5",
        "architectures": ["T5ForConditionalGeneration"],
        "vocab_size": 512,
        "d_model": 64,
        "d_ff": 128,
        "d_kv": 16,
        "num_layers": 2,
        "num_decoder_layers": 2,
        "num_heads": 4,
        "feed_forward_proj": "relu",
        "pad_token_id": 0,
        "eos_token_id": 1,
        "decoder_start_token_id": 0,
        "is_encoder_decoder": True,
    }
    (model_dir / "config.json").write_text(json.dumps(raw_cfg, indent=2, ensure_ascii=False))

    (model_dir / "pytorch_model.bin.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {
                    "_dummy_0": "pytorch_model-00001-of-00002.bin",
                    "_dummy_1": "pytorch_model-00002-of-00002.bin",
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    cfg = GenerationConfig(strategy="ultra")
    cfg.security = SecurityOptions(
        trust_remote_code=False,
        allow_network=False,
        local_files_only=True,
    )

    g = MinimalWeightGenerator(
        model_id=str(model_dir),
        output_dir=str(out_dir),
        config=cfg,
        shrink_config=True,
    )
    g.generate()

    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(out_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    assert model is not None
