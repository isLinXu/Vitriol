import json
from pathlib import Path

from vitriol.config.manager import GenerationConfig, SecurityOptions
from vitriol.core.generator import MinimalWeightGenerator


def test_offline_mode_does_not_try_network_for_local_model(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    out_dir = tmp_path / "out"
    model_dir.mkdir()
    out_dir.mkdir()

    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "vocab_size": 1000,
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "intermediate_size": 256,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    cfg = GenerationConfig()
    cfg.security = SecurityOptions(
        trust_remote_code=False,
        allow_network=False,
        local_files_only=True,
    )

    g = MinimalWeightGenerator(model_id=str(model_dir), output_dir=str(out_dir), config=cfg)
    c = g._load_hf_config()
    assert getattr(c, "model_type", None) == "qwen3_5_moe"
