import json
from pathlib import Path

from vitriol.arch_viz.core import Architecture, Layer
from vitriol.arch_viz.visualizer import ArchitectureVisualizer
from vitriol.arch_viz.renderers.detail import DetailRenderer
from vitriol.core.generator import MinimalWeightGenerator


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def test_load_unknown_model_type_uses_adapter_registration(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    out_dir = tmp_path / "out"
    model_dir.mkdir()
    out_dir.mkdir()

    _write_json(
        model_dir / "config.json",
        {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "vocab_size": 32000,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "intermediate_size": 256,
            "tie_word_embeddings": False,
        },
    )

    g = MinimalWeightGenerator(model_id=str(model_dir), output_dir=str(out_dir))
    cfg = g._load_hf_config()
    assert getattr(cfg, "model_type", None) == "qwen3_5_moe"
    assert getattr(cfg, "model_type", None) != "deepseek_v3"


def test_meta_config_is_raw_source_config_for_local_models(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    out_dir = tmp_path / "out"
    model_dir.mkdir()
    out_dir.mkdir()

    raw = {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "image_token_id": 248056,
        "tie_word_embeddings": False,
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "vocab_size": 1000,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
            "num_experts": 8,
            "num_experts_per_tok": 2,
        },
        "vision_config": {
            "model_type": "qwen3_5_moe",
            "depth": 2,
            "hidden_size": 32,
        },
    }
    _write_json(model_dir / "config.json", raw)

    g = MinimalWeightGenerator(model_id=str(model_dir), output_dir=str(out_dir))
    hf_config = g._load_hf_config()
    g._save_configs(hf_config)

    meta = json.loads((out_dir / "meta-config.json").read_text())
    assert meta == raw

    g._write_manifest()
    assert (out_dir / "vitriol-manifest.json").exists()


def test_arch_viz_html_generation_does_not_crash_on_zero_heads(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    cfg = {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "vocab_size": 1000,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 0,
            "num_key_value_heads": 0,
            "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "layer_types": ["full_attention", "full_attention"],
        },
    }
    _write_json(model_dir / "config.json", cfg)
    _write_json(model_dir / "meta-config.json", cfg)

    out_html = tmp_path / "out.html"
    viz = ArchitectureVisualizer(
        str(model_dir),
        trust_remote_code=False,
        local_files_only=True,
    )
    viz.generate_interactive_html(str(out_html))
    assert out_html.exists()


def test_arch_viz_html_hy3_shows_moe_router_metadata(tmp_path: Path) -> None:
    model_dir = tmp_path / "hy3"
    model_dir.mkdir()

    cfg = {
        "model_type": "hy_v3",
        "architectures": ["HYV3ForCausalLM"],
        "vocab_size": 120832,
        "hidden_size": 4096,
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 13312,
        "moe_intermediate_size": 1536,
        "num_experts": 192,
        "num_experts_per_tok": 8,
        "num_shared_experts": 1,
        "first_k_dense_replace": 1,
        "num_nextn_predict_layers": 1,
        "max_position_embeddings": 262144,
        "qk_norm": True,
        "route_norm": True,
        "moe_router_use_sigmoid": True,
        "moe_router_enable_expert_bias": True,
        "router_scaling_factor": 2.826,
        "tie_word_embeddings": False,
        "rope_parameters": {"rope_type": "default", "rope_theta": 11158840.0},
    }
    _write_json(model_dir / "config.json", cfg)
    _write_json(model_dir / "meta-config.json", cfg)

    out_html = tmp_path / "hy3.html"
    viz = ArchitectureVisualizer(
        str(model_dir),
        trust_remote_code=False,
        local_files_only=True,
    )
    viz.generate_interactive_html(str(out_html))

    html = out_html.read_text(encoding="utf-8")
    assert "HYV3ForCausalLM" in html or "hy_v3" in html
    assert "top-8 of 192 active" in html
    assert "Dense Prefix" in html
    assert "MoE top-8 / 192" in html
    assert "Sigmoid Router" in html
    assert "Router Bias" in html
    assert "Scale · 2.826" in html
    assert "256K Context" in html
    assert "Dense Prefix · 1 layer" in html
    assert "MoE Blocks · 79 layers" in html
    assert "MTP Head · 1 layer" in html


def test_detail_renderer_hy3_role_classification() -> None:
    arch = Architecture(
        model_type="hy_v3",
        arch_type="decoder-only",
        total_layers=5,
        total_params=123,
        memory_fp16_gb=0.0,
        parameters={"dense_prefix_layers": 1, "mtp_layers": 1},
        features=["Hy3", "MoE", "MTP (1)"],
        layers=[
            Layer("Block 0 - FFN", "feedforward", 1, (1, 1), "Dense SwiGLU (Inter: 13312)"),
            Layer("Block 1 - FFN", "feedforward", 1, (1, 1), "MoE (Experts: 192, TopK: 8, Shared: 1, Inter: 1536)"),
            Layer("MTP Head", "adapter", 0, (1,), "Next-N prediction layers: 1"),
        ],
    )

    assert DetailRenderer._role_for_layer(arch, arch.layers[0]) == "Dense Prefix"
    assert DetailRenderer._role_for_layer(arch, arch.layers[1]) == "MoE Block"
    assert DetailRenderer._role_for_layer(arch, arch.layers[2]) == "MTP"
