import json
import types
from pathlib import Path

import pytest
import torch

from vitriol.arch_viz.analyzer import ArchitectureAnalyzer
from vitriol.arch_viz.parser import ConfigParser
from vitriol.cli.commands.viz import build_inline_config_model
from vitriol.config.manager import GenerationConfig, SecurityOptions
from vitriol.core.generator import MinimalWeightGenerator


def test_get_original_shard_map_prefers_local_index_without_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_dir = tmp_path / "model"
    out_dir = tmp_path / "out"
    model_dir.mkdir()
    out_dir.mkdir()

    (model_dir / "config.json").write_text(json.dumps({"model_type": "gpt2"}, indent=2))
    (model_dir / "pytorch_model.bin.index.json").write_text(
        json.dumps(
            {"weight_map": {"model.embed_tokens.weight": "pytorch_model-00001-of-00001.bin"}},
            indent=2,
        )
    )

    def _boom(*_a, **_kw):
        raise AssertionError("Network access attempted")

    monkeypatch.setattr("huggingface_hub.list_repo_files", _boom, raising=False)

    cfg = GenerationConfig()
    cfg.security = SecurityOptions(
        trust_remote_code=False,
        allow_network=False,
        local_files_only=True,
    )
    g = MinimalWeightGenerator(model_id=str(model_dir), output_dir=str(out_dir), config=cfg)
    m = g._get_original_shard_map()
    assert m.get("model.embed_tokens.weight") == "pytorch_model-00001-of-00001.bin"


def test_resolve_target_shard_reuses_precomputed_shard_order(monkeypatch: pytest.MonkeyPatch) -> None:
    from vitriol.core import generator as gen_mod

    generator = MinimalWeightGenerator.__new__(MinimalWeightGenerator)
    original_map = {
        "w1": "pytorch_model-00001-of-00002.bin",
        "w2": "pytorch_model-00002-of-00002.bin",
    }
    available = ["pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin"]

    def _boom(*_args, **_kwargs):
        raise AssertionError("sorted() should not be called when available_shards is provided")

    monkeypatch.setattr(gen_mod, "sorted", _boom, raising=False)

    shard = generator._resolve_target_shard(
        "missing.weight",
        original_map,
        None,
        param_seq_idx=1,
        available_shards=available,
    )
    assert shard == available[1]


def test_copy_custom_code_files_respects_security_offline_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_dir = tmp_path / "model"
    out_dir = tmp_path / "out"
    model_dir.mkdir()
    out_dir.mkdir()

    (model_dir / "config.json").write_text(json.dumps({"model_type": "gpt2"}, indent=2))

    called = {"list_repo_files": 0}

    def _list_repo_files(*_a, **_kw):
        called["list_repo_files"] += 1
        return []

    monkeypatch.setattr("huggingface_hub.list_repo_files", _list_repo_files, raising=False)

    cfg = GenerationConfig()
    cfg.security = SecurityOptions(
        trust_remote_code=False,
        allow_network=False,
        local_files_only=True,
    )
    g = MinimalWeightGenerator(model_id=str(model_dir), output_dir=str(out_dir), config=cfg)
    g._copy_custom_code_files()
    assert called["list_repo_files"] == 0


def test_copy_custom_code_files_uses_filename_allowlist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_dir = tmp_path / "out"
    source_dir = tmp_path / "source"
    out_dir.mkdir()
    source_dir.mkdir()

    repo_files = [
        "modeling_demo.py",
        "configuration_demo.py",
        "tokenization_demo.py",
        "evil.py",
        "scripts/post_install.py",
        "tokenizer/tokenizer.json",
        "tokenizer/payload.sh",
        "../modeling_escape.py",
    ]
    downloads: list[str] = []

    def _list_repo_files(_repo_id: str):
        return list(repo_files)

    def _hf_hub_download(*, repo_id: str, filename: str):
        downloads.append(filename)
        source_path = source_dir / filename.replace("/", "__")
        source_path.write_text(f"{repo_id}:{filename}", encoding="utf-8")
        return str(source_path)

    monkeypatch.setattr("huggingface_hub.list_repo_files", _list_repo_files, raising=False)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _hf_hub_download, raising=False)

    cfg = GenerationConfig()
    cfg.security = SecurityOptions(
        trust_remote_code=True,
        allow_network=True,
        local_files_only=False,
    )
    g = MinimalWeightGenerator(model_id="demo/model", output_dir=str(out_dir), config=cfg)
    g._copy_custom_code_files()

    assert (out_dir / "modeling_demo.py").exists()
    assert (out_dir / "configuration_demo.py").exists()
    assert (out_dir / "tokenization_demo.py").exists()
    assert (out_dir / "tokenizer" / "tokenizer.json").exists()
    assert not (out_dir / "evil.py").exists()
    assert not (out_dir / "scripts" / "post_install.py").exists()
    assert not (out_dir / "tokenizer" / "payload.sh").exists()
    assert not (tmp_path / "modeling_escape.py").exists()
    assert downloads == [
        "modeling_demo.py",
        "configuration_demo.py",
        "tokenization_demo.py",
        "tokenizer/tokenizer.json",
    ]


def test_copy_custom_code_files_scopes_python_files_to_auto_map(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "out"
    source_dir = tmp_path / "source"
    out_dir.mkdir()
    source_dir.mkdir()
    (out_dir / "meta-config.json").write_text(
        json.dumps(
            {
                "auto_map": {
                    "AutoConfig": "configuration_demo.DemoConfig",
                    "AutoModel": "modeling_demo.DemoModel",
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    repo_files = [
        "modeling_demo.py",
        "modeling_unused.py",
        "configuration_demo.py",
        "tokenization_unused.py",
        "tokenizer/tokenizer.json",
    ]
    downloads: list[str] = []

    def _list_repo_files(_repo_id: str):
        return list(repo_files)

    def _hf_hub_download(*, repo_id: str, filename: str):
        downloads.append(filename)
        source_path = source_dir / filename.replace("/", "__")
        source_path.write_text(f"{repo_id}:{filename}", encoding="utf-8")
        return str(source_path)

    monkeypatch.setattr("huggingface_hub.list_repo_files", _list_repo_files, raising=False)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _hf_hub_download, raising=False)

    cfg = GenerationConfig()
    cfg.security = SecurityOptions(
        trust_remote_code=True,
        allow_network=True,
        local_files_only=False,
    )
    g = MinimalWeightGenerator(model_id="demo/model", output_dir=str(out_dir), config=cfg)
    g._copy_custom_code_files()

    assert (out_dir / "modeling_demo.py").exists()
    assert (out_dir / "configuration_demo.py").exists()
    assert (out_dir / "tokenizer" / "tokenizer.json").exists()
    assert not (out_dir / "modeling_unused.py").exists()
    assert not (out_dir / "tokenization_unused.py").exists()
    assert downloads == [
        "modeling_demo.py",
        "configuration_demo.py",
        "tokenizer/tokenizer.json",
    ]


def test_copy_custom_code_files_honors_file_count_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "out"
    source_dir = tmp_path / "source"
    out_dir.mkdir()
    source_dir.mkdir()
    monkeypatch.setenv("VITRIOL_CUSTOM_CODE_MAX_FILES", "2")

    repo_files = [
        "modeling_demo.py",
        "configuration_demo.py",
        "tokenization_demo.py",
    ]
    downloads: list[str] = []

    def _list_repo_files(_repo_id: str):
        return list(repo_files)

    def _hf_hub_download(*, repo_id: str, filename: str):
        downloads.append(filename)
        source_path = source_dir / filename
        source_path.write_text(f"{repo_id}:{filename}", encoding="utf-8")
        return str(source_path)

    monkeypatch.setattr("huggingface_hub.list_repo_files", _list_repo_files, raising=False)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _hf_hub_download, raising=False)

    cfg = GenerationConfig()
    cfg.security = SecurityOptions(
        trust_remote_code=True,
        allow_network=True,
        local_files_only=False,
    )
    g = MinimalWeightGenerator(model_id="demo/model", output_dir=str(out_dir), config=cfg)
    g._copy_custom_code_files()

    assert (out_dir / "modeling_demo.py").exists()
    assert (out_dir / "configuration_demo.py").exists()
    assert not (out_dir / "tokenization_demo.py").exists()
    assert downloads == ["modeling_demo.py", "configuration_demo.py"]


def test_copy_custom_code_files_skips_oversized_python_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "out"
    source_dir = tmp_path / "source"
    out_dir.mkdir()
    source_dir.mkdir()
    monkeypatch.setenv("VITRIOL_CUSTOM_CODE_MAX_PY_BYTES", "4")

    def _list_repo_files(_repo_id: str):
        return ["modeling_demo.py"]

    def _hf_hub_download(*, repo_id: str, filename: str):
        source_path = source_dir / filename
        source_path.write_text("12345", encoding="utf-8")
        return str(source_path)

    monkeypatch.setattr("huggingface_hub.list_repo_files", _list_repo_files, raising=False)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _hf_hub_download, raising=False)

    cfg = GenerationConfig()
    cfg.security = SecurityOptions(
        trust_remote_code=True,
        allow_network=True,
        local_files_only=False,
    )
    g = MinimalWeightGenerator(model_id="demo/model", output_dir=str(out_dir), config=cfg)
    g._copy_custom_code_files()

    assert not (out_dir / "modeling_demo.py").exists()


def test_patch_remote_classes_skips_dynamic_module_when_trust_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    calls: list[tuple[str, str]] = []

    def _get_class_from_dynamic_module(class_reference: str, model_id: str, **_kwargs):
        calls.append((class_reference, model_id))
        raise AssertionError("dynamic module loading should not run without trust_remote_code")

    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
        _get_class_from_dynamic_module,
    )

    cfg = GenerationConfig()
    cfg.security = SecurityOptions(
        trust_remote_code=False,
        allow_network=True,
        local_files_only=False,
    )
    g = MinimalWeightGenerator(model_id="demo/model", output_dir=str(out_dir), config=cfg)
    hf_config = types.SimpleNamespace(auto_map={"AutoModel": "modeling_demo.DemoModel"})

    g._patch_remote_classes(hf_config)

    assert calls == []


def test_reconcile_config_never_uses_unsafe_torch_load_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cfg_path = out_dir / "config.json"
    cfg_path.write_text(json.dumps({"model_type": "gpt2"}, indent=2), encoding="utf-8")
    (out_dir / "pytorch_model.bin.index.json").write_text(
        json.dumps(
            {"weight_map": {"model.embed_tokens.weight": "pytorch_model-00001-of-00001.bin"}},
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "pytorch_model-00001-of-00001.bin").write_bytes(b"not a safe weights file")

    calls: list[dict[str, object]] = []

    def _fake_torch_load(_path, **kwargs):
        calls.append(dict(kwargs))
        raise RuntimeError("weights_only blocked")

    monkeypatch.setattr("vitriol.core.generator.torch.load", _fake_torch_load)

    cfg = GenerationConfig()
    g = MinimalWeightGenerator(model_id="demo/model", output_dir=str(out_dir), config=cfg)
    g._reconcile_config_with_weights(str(cfg_path))

    assert calls == [{"map_location": "cpu", "weights_only": True}]


def test_arch_viz_parser_falls_back_to_pretrained_config_for_unknown_type(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    raw = {
        "model_type": "unknown_future_model",
        "architectures": ["SomeFutureArch"],
        "text_config": {"vocab_size": 123, "hidden_size": 456},
    }
    (model_dir / "meta-config.json").write_text(json.dumps(raw, indent=2))

    cfg = ConfigParser.load_config(str(model_dir), trust_remote_code=False, local_files_only=True)
    assert getattr(cfg, "model_type", None) == "unknown_future_model"


def test_build_inline_config_model_prefers_meta_config(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gpt2",
                "vocab_size": 10,
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
            },
            indent=2,
        )
    )

    meta = {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": {
            "vocab_size": 1000,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
        },
    }
    (model_dir / "meta-config.json").write_text(json.dumps(meta, indent=2))

    m = build_inline_config_model(model_dir)
    assert m is not None
    assert m.get("meta") == meta
    assert m.get("raw") == meta
    assert m.get("hidden_size") == 64
    assert m.get("num_layers") == 2
    assert m.get("config_source") == "meta-config.json"
    assert m.get("params_source") in {"analyzer", "config_derived"}


def test_build_inline_config_model_diffusers_does_not_emit_placeholder_params(tmp_path: Path) -> None:
    model_dir = tmp_path / "sd"
    (model_dir / "unet").mkdir(parents=True)
    (model_dir / "vae").mkdir()
    (model_dir / "text_encoder").mkdir()

    (model_dir / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionPipeline"}, indent=2),
        encoding="utf-8",
    )
    (model_dir / "unet" / "config.json").write_text(
        json.dumps({"cross_attention_dim": 1024, "down_block_types": ["A", "B", "C"]}, indent=2),
        encoding="utf-8",
    )

    m = build_inline_config_model(model_dir)
    assert m is not None
    assert m.get("is_diffusion") is True
    assert m.get("total_params") == 0
    assert m.get("params_source") == "unavailable"
    assert m.get("config_source") == "model_index.json"


def test_generator_rope_defaults_do_not_add_unsupported_default_keys() -> None:
    from vitriol.core import generator as gen_mod
    from vitriol.patches import model_family_patches as patch_mod

    assert gen_mod._ROPE_DEFAULTS == {
        "rope_type": "default",
        "rope_theta": 10000.0,
    }
    assert patch_mod._ROPE_DEFAULTS == {
        "rope_type": "default",
        "rope_theta": 10000.0,
    }


def test_generator_skips_non_persistent_buffers_during_export(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    out_dir = tmp_path / "out"
    model_dir.mkdir()
    out_dir.mkdir()

    (model_dir / "config.json").write_text(json.dumps({"model_type": "gpt2"}, indent=2), encoding="utf-8")
    (model_dir / "pytorch_model.bin.index.json").write_text(
        json.dumps({"weight_map": {"x": "pytorch_model-00001-of-00001.bin"}}, indent=2),
        encoding="utf-8",
    )

    cfg = GenerationConfig(strategy="ultra")
    cfg.security = SecurityOptions(trust_remote_code=False, allow_network=False, local_files_only=True)
    _ = MinimalWeightGenerator(model_id=str(model_dir), output_dir=str(out_dir), config=cfg)

    class DummyModel(torch.nn.Module):
        class Inner(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("inner_persist", torch.ones(1), persistent=True)
                self.register_buffer("inner_temp", torch.ones(1), persistent=False)

        def __init__(self) -> None:
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(1))
            self.register_buffer("persist_buf", torch.ones(1), persistent=True)
            self.register_buffer("temp_buf", torch.ones(1), persistent=False)
            self.inner = self.Inner()

    names = []
    model = DummyModel()
    non_persistent_buffers = set()
    for module_prefix, module in model.named_modules():
        local_names = getattr(module, "_non_persistent_buffers_set", set())
        for local_name in local_names:
            qualified = f"{module_prefix}.{local_name}" if module_prefix else local_name
            non_persistent_buffers.add(qualified)

    def _iter_export_tensors():
        for item in model.named_parameters():
            yield item
        for name, buf in model.named_buffers():
            if name in non_persistent_buffers:
                continue
            yield name, buf

    names = [name for name, _ in _iter_export_tensors()]

    assert "param" in names
    assert "persist_buf" in names
    assert "temp_buf" not in names
    assert "inner.inner_persist" in names
    assert "inner.inner_temp" not in names


def test_generator_snapshot_export_tensors_reuses_single_parameter_and_buffer_walk() -> None:
    generator = MinimalWeightGenerator.__new__(MinimalWeightGenerator)

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.param_walks = 0
            self.buffer_walks = 0
            self.param = torch.nn.Parameter(torch.zeros(1))
            self.register_buffer("persist_buf", torch.ones(1), persistent=True)
            self.register_buffer("temp_buf", torch.ones(1), persistent=False)

        def named_parameters(self, *args, **kwargs):
            self.param_walks += 1
            return super().named_parameters(*args, **kwargs)

        def named_buffers(self, *args, **kwargs):
            self.buffer_walks += 1
            return super().named_buffers(*args, **kwargs)

    model = DummyModel()
    export_items = generator._snapshot_export_tensors(model)
    names = [name for name, _ in export_items]

    assert model.param_walks == 1
    assert model.buffer_walks == 1
    assert names == ["param", "persist_buf"]


def test_hy3_architecture_analyzer_exposes_moe_gqa_and_mtp_metadata(tmp_path: Path) -> None:
    model_dir = tmp_path / "hy3"
    model_dir.mkdir()

    raw = {
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
        "expert_hidden_dim": 1536,
        "tie_word_embeddings": False,
        "rope_parameters": {"rope_type": "default", "rope_theta": 11158840.0},
    }
    (model_dir / "config.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    (model_dir / "meta-config.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")

    cfg = ConfigParser.load_config(str(model_dir), trust_remote_code=False, local_files_only=True)
    arch = ArchitectureAnalyzer().analyze(cfg)

    assert arch.model_type == "hy_v3"
    assert "Hy3" in arch.features
    assert "MoE" in arch.features
    assert "GQA" in arch.features
    assert "Long Context" in arch.features
    assert "MTP (1)" in arch.features
    assert "RouteNorm" in arch.features
    assert "Sigmoid Router" in arch.features
    assert "Router Bias" in arch.features
    assert arch.parameters["num_kv_heads"] == 8
    assert arch.parameters["num_experts"] == 192
    assert arch.parameters["top_k_experts"] == 8
    assert arch.parameters["dense_prefix_layers"] == 1
    assert arch.parameters["mtp_layers"] == 1
    assert arch.parameters["route_norm"] is True
    assert arch.parameters["router_sigmoid"] is True
    assert arch.parameters["router_bias"] is True
    assert arch.parameters["router_scaling_factor"] == 2.826

    moe_layer = next(layer for layer in arch.layers if layer.name == "Block 1 - FFN")
    assert "TopK: 8" in moe_layer.description
    assert "sigmoid router" in moe_layer.description


def test_save_tokenizer_preserves_tokenizers_backend_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_dir = tmp_path / "model"
    out_dir = tmp_path / "out"
    model_dir.mkdir()
    out_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "gpt2"}, indent=2), encoding="utf-8")

    class FakeTokenizer:
        def save_pretrained(self, path: str) -> None:
            Path(path, "tokenizer.json").write_text('{"model":{"unk_token":null}}', encoding="utf-8")
            Path(path, "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "backend": "tokenizers",
                        "tokenizer_class": "TokenizersBackend",
                        "bos_token": "<bos>",
                        "eos_token": "<eos>",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    monkeypatch.setattr(
        "vitriol.utils.hf_loading.load_tokenizer",
        lambda *_args, **_kwargs: FakeTokenizer(),
    )

    cfg = GenerationConfig(strategy="ultra")
    cfg.security = SecurityOptions(trust_remote_code=True, allow_network=False, local_files_only=True)
    generator = MinimalWeightGenerator(model_id=str(model_dir), output_dir=str(out_dir), config=cfg)
    generator._save_tokenizer()

    saved_cfg = json.loads((out_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert saved_cfg["tokenizer_class"] == "TokenizersBackend"
    assert saved_cfg["backend"] == "tokenizers"


def test_save_tokenizer_falls_back_to_copying_repo_assets_when_loader_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "repo"
    out_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"
    model_dir.mkdir()
    out_dir.mkdir()
    cache_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "gpt2"}, indent=2), encoding="utf-8")

    token_files = {
        "tokenizer.json": '{"model":{"unk_token":null}}',
        "tokenizer_config.json": json.dumps({"tokenizer_class": "TokenizersBackend", "backend": "tokenizers"}, indent=2),
        "special_tokens_map.json": json.dumps({"bos_token": "<bos>"}, indent=2),
    }
    for name, content in token_files.items():
        (cache_dir / name).write_text(content, encoding="utf-8")

    monkeypatch.setattr(
        "vitriol.utils.hf_loading.load_tokenizer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AttributeError("'PreTrainedConfig' object has no attribute 'max_position_embeddings'")
        ),
    )
    monkeypatch.setattr(
        "huggingface_hub.list_repo_files",
        lambda *_args, **_kwargs: ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
        raising=False,
    )
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda repo_id=None, filename=None, **_kwargs: str(cache_dir / filename),
        raising=False,
    )

    cfg = GenerationConfig(strategy="ultra")
    cfg.security = SecurityOptions(trust_remote_code=True, allow_network=True, local_files_only=False)
    generator = MinimalWeightGenerator(
        model_id="deepseek-ai/DeepSeek-V4-Flash-Base",
        output_dir=str(out_dir),
        config=cfg,
    )
    generator._save_tokenizer()

    assert (out_dir / "tokenizer.json").exists()
    assert (out_dir / "tokenizer_config.json").exists()
    assert (out_dir / "special_tokens_map.json").exists()
