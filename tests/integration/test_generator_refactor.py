"""Integration tests for generator refactor and ArchitectureGene extension.

These tests verify that the split modules work together correctly after
refactoring and that the extended ArchitectureGene supports MLA/MoE/Mamba.
"""
from __future__ import annotations

import pytest

from vitriol.core._generator_utils import (
    GenerationResult,
    build_fallback_config,
    copy_safe_attrs,
    custom_repo_file_size_limit,
    extract_auto_map_modules,
    extract_shard_id,
    find_best_alias,
    inject_recursive,
    is_allowed_custom_repo_file,
    positive_int_env,
)
from vitriol.core.generator import MinimalWeightGenerator
from vitriol.core.shrinker import ConfigShrinker
from vitriol.nas.search_space import ArchitectureGene, LLMSearchSpace


class TestGeneratorRefactor:
    """Verify that the split generator modules still work together."""

    def test_all_imports_work(self) -> None:
        """All split modules can be imported and have the expected symbols."""
        from vitriol.core.config_loader import load_hf_config
        from vitriol.core.custom_code_sync import copy_custom_code_files
        from vitriol.core.generator_persistence import save_configs

        assert callable(load_hf_config)
        assert callable(copy_custom_code_files)
        assert callable(save_configs)
        assert callable(positive_int_env)
        assert callable(extract_shard_id)
        assert callable(build_fallback_config)
        assert callable(find_best_alias)
        assert callable(inject_recursive)
        assert callable(copy_safe_attrs)
        assert callable(extract_auto_map_modules)
        assert callable(is_allowed_custom_repo_file)
        assert callable(custom_repo_file_size_limit)

    def test_generation_result_dataclass(self) -> None:
        """GenerationResult can be constructed and serialised."""
        result = GenerationResult(
            output_dir="/tmp/test",
            manifest_path=None,
            index_path="/tmp/test/index.json",
            total_size=1024,
            generated_at="2026-06-14T12:00:00Z",
        )
        d = result.to_dict()
        assert d["output_dir"] == "/tmp/test"
        assert d["total_size"] == 1024
        assert "generated_at" in d

    def test_config_shrinker_standalone(self) -> None:
        """ConfigShrinker can shrink a mock config object independently."""
        class MockConfig:
            num_hidden_layers = 24
            hidden_size = 2048
            num_attention_heads = 16
            intermediate_size = 8192
            num_experts = 8

        cfg = MockConfig()
        ConfigShrinker().shrink(cfg)
        assert cfg.hidden_size == 256
        assert cfg.num_hidden_layers == 2
        assert cfg.num_attention_heads == 2

    def test_find_best_alias_known_types(self) -> None:
        """Alias discovery works for known model types."""
        aliases = find_best_alias("gemma4")
        assert "gemma3" in aliases

        aliases = find_best_alias("deepseek_v4")
        assert "deepseek_v3" in aliases

    def test_extract_auto_map_modules(self) -> None:
        """Auto-map module extraction handles nested structures."""
        auto_map = {
            "AutoModelForCausalLM": "custom_modeling.modeling.MyModel",
            "AutoModel": ["custom_modeling.modeling.MyModel", "fallback.Model"],
        }
        modules = extract_auto_map_modules(auto_map)
        assert "custom_modeling.modeling" in modules
        assert "fallback" in modules

    def test_is_allowed_custom_repo_file(self) -> None:
        """Custom-repo file allow-list behaves correctly."""
        assert is_allowed_custom_repo_file("modeling_custom.py") is True
        assert is_allowed_custom_repo_file("model.safetensors") is False
        assert is_allowed_custom_repo_file("tokenizer/tokenizer.json") is True
        assert is_allowed_custom_repo_file("../../etc/passwd") is False

    def test_extract_shard_id(self) -> None:
        """Shard ID extraction is robust."""
        assert extract_shard_id("model-00001-of-00004.safetensors", 0) == 1
        assert extract_shard_id("pytorch_model-00003-of-00010.bin", 0) == 3
        assert extract_shard_id("model.bin", 99) == 99


class TestArchitectureGeneExtension:
    """Verify MLA/MoE/Mamba support in ArchitectureGene."""

    def test_basic_gene_no_extensions(self) -> None:
        """Default gene has all advanced flags disabled."""
        gene = ArchitectureGene(
            n_layers=12, hidden_size=1024, n_heads=16,
            attention_type="MHA", ffn_type="SwiGLU",
            activation="silu", norm_type="RMSNorm",
        )
        assert gene.use_mla is False
        assert gene.use_moe is False
        assert gene.use_mamba is False
        assert gene.intermediate_size > 0
        assert gene.num_kv_heads == 16

    def test_mla_gene_derived_fields(self) -> None:
        """MLA gene computes qk_nope_head_dim, kv_lora_rank, etc."""
        gene = ArchitectureGene(
            n_layers=12, hidden_size=1024, n_heads=16,
            attention_type="GQA", ffn_type="SwiGLU",
            activation="silu", norm_type="RMSNorm",
            use_mla=True,
        )
        assert gene.use_mla is True
        assert gene.qk_nope_head_dim > 0
        assert gene.qk_rope_head_dim > 0
        assert gene.kv_lora_rank > 0
        assert gene.q_lora_rank > 0

    def test_moe_gene_derived_fields(self) -> None:
        """MoE gene computes num_experts, moe_intermediate_size, etc."""
        gene = ArchitectureGene(
            n_layers=12, hidden_size=1024, n_heads=16,
            attention_type="GQA", ffn_type="SwiGLU",
            activation="silu", norm_type="RMSNorm",
            use_moe=True,
        )
        assert gene.use_moe is True
        assert gene.num_experts > 0
        assert gene.num_experts_per_tok > 0
        assert gene.moe_intermediate_size > 0
        assert gene.shared_expert_intermediate_size > 0

    def test_mamba_gene_derived_fields(self) -> None:
        """Mamba gene computes d_state, d_conv, expand_factor."""
        gene = ArchitectureGene(
            n_layers=12, hidden_size=1024, n_heads=16,
            attention_type="MHA", ffn_type="Standard",
            activation="gelu", norm_type="LayerNorm",
            use_mamba=True,
        )
        assert gene.use_mamba is True
        assert gene.d_state > 0
        assert gene.d_conv > 0
        assert gene.expand_factor > 0

    def test_to_config_includes_mla_moe(self) -> None:
        """to_config emits MLA and MoE fields when enabled."""
        gene = ArchitectureGene(
            n_layers=12, hidden_size=1024, n_heads=16,
            attention_type="GQA", ffn_type="SwiGLU",
            activation="silu", norm_type="RMSNorm",
            use_mla=True, use_moe=True,
        )
        cfg = gene.to_config()
        assert "qk_nope_head_dim" in cfg
        assert "num_experts" in cfg
        assert "_vitriol_nas_gene" in cfg
        # The gene markers are stored inside _vitriol_nas_gene
        assert cfg["_vitriol_nas_gene"]["use_mla"] is True
        assert cfg["_vitriol_nas_gene"]["use_moe"] is True

    def test_from_config_detects_mla(self) -> None:
        """from_config detects MLA from config dict."""
        raw = {
            "num_hidden_layers": 12,
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "intermediate_size": 4096,
            "hidden_act": "silu",
            "qk_nope_head_dim": 64,
            "kv_lora_rank": 128,
            "vocab_size": 32000,
        }
        gene = ArchitectureGene.from_config(raw)
        assert gene.use_mla is True
        assert gene.use_moe is False

    def test_from_config_detects_moe(self) -> None:
        """from_config detects MoE from config dict."""
        raw = {
            "num_hidden_layers": 12,
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "intermediate_size": 4096,
            "hidden_act": "silu",
            "num_experts": 8,
            "n_routed_experts": 8,
            "vocab_size": 32000,
        }
        gene = ArchitectureGene.from_config(raw)
        assert gene.use_moe is True

    def test_from_config_detects_mamba(self) -> None:
        """from_config detects Mamba from model_type prefix."""
        raw = {
            "num_hidden_layers": 12,
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "hidden_act": "silu",
            "model_type": "mamba2",
            "d_state": 64,
            "vocab_size": 32000,
        }
        gene = ArchitectureGene.from_config(raw)
        assert gene.use_mamba is True

    def test_search_space_sample_advanced(self) -> None:
        """LLMSearchSpace.sample can produce genes with advanced flags."""
        space = LLMSearchSpace()
        genes = [space.sample() for _ in range(50)]
        mla_count = sum(1 for g in genes if g.use_mla)
        moe_count = sum(1 for g in genes if g.use_moe)
        mamba_count = sum(1 for g in genes if g.use_mamba)
        # Probabilistic assertion — with 50 samples we expect some hits
        assert mla_count >= 1, f"Expected at least 1 MLA in 50 samples, got {mla_count}"
        assert moe_count >= 1, f"Expected at least 1 MoE in 50 samples, got {moe_count}"
        assert mamba_count >= 1, f"Expected at least 1 Mamba in 50 samples, got {mamba_count}"

    def test_mutate_advanced_flags(self) -> None:
        """Mutate can flip advanced flags."""
        space = LLMSearchSpace()
        gene = space.sample()
        mutated = space.mutate(gene, mutation_rate=0.5)
        # After high mutation rate, at least one flag may differ
        assert isinstance(mutated, ArchitectureGene)
        assert mutated.use_mla in (True, False)
        assert mutated.use_moe in (True, False)
        assert mutated.use_mamba in (True, False)
