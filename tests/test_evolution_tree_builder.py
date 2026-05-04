"""Tests for evolution/tree_builder module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from vitriol.evolution.tree_builder import (
    ArchNode,
    ArchInnovation,
    ArchitectureMetrics,
    EvolutionTree,
    DEFAULT_FAMILIES,
    FALLBACK_PARAMS,
)


# Prevent any network calls during tree building by mocking hf_load_config globally
@pytest.fixture(autouse=True)
def mock_hf_load_config():
    with patch("vitriol.evolution.tree_builder.hf_load_config") as mock_hf:
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {"hidden_size": 128, "model_type": "test"}
        mock_hf.return_value = mock_config
        yield mock_hf


# ─────────────────────────────────────────────────────────────────────────────
# ArchInnovation Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestArchInnovation:
    """Tests for ArchInnovation dataclass."""

    def test_creation(self):
        innov = ArchInnovation(
            name="GQA",
            description="Grouped Query Attention",
            introduced_in="Llama-2-70B",
            year=2023,
        )
        assert innov.name == "GQA"
        assert innov.year == 2023

    def test_defaults(self):
        innov = ArchInnovation(name="Test", description="Desc", introduced_in="Model", year=2024)
        assert innov.description == "Desc"


# ─────────────────────────────────────────────────────────────────────────────
# ArchNode Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestArchNode:
    """Tests for ArchNode dataclass."""

    def test_model_name_with_org(self):
        node = ArchNode(model_id="org/model-name", config={})
        assert node.model_name == "model-name"

    def test_model_name_no_slash(self):
        node = ArchNode(model_id="model-name", config={})
        assert node.model_name == "model-name"

    def test_family_qwen(self):
        node = ArchNode(model_id="qwen/Qwen2-7B", config={})
        assert node.family == "Qwen"

    def test_family_llama_meta(self):
        node = ArchNode(model_id="meta-llama/Llama-2-7b", config={})
        assert node.family == "LLaMA"

    def test_family_llama_keyword(self):
        node = ArchNode(model_id="some-org/llama-custom", config={})
        assert node.family == "LLaMA"

    def test_family_mistral(self):
        node = ArchNode(model_id="mistralai/Mistral-7B", config={})
        assert node.family == "LLaMA"

    def test_family_deepseek_org(self):
        node = ArchNode(model_id="deepseek-ai/DeepSeek-V3", config={})
        assert node.family == "DeepSeek"

    def test_family_deepseek_keyword(self):
        node = ArchNode(model_id="org/deepseek-custom", config={})
        assert node.family == "DeepSeek"

    def test_family_glm(self):
        node = ArchNode(model_id="THUDM/glm-4-9b", config={})
        assert node.family == "GLM"

    def test_family_gpt(self):
        node = ArchNode(model_id="openai/gpt-4", config={})
        assert node.family == "GPT"

    def test_family_kimi(self):
        node = ArchNode(model_id="moonshotai/kimi-v1", config={})
        assert node.family == "Kimi"

    def test_family_kimi_keyword(self):
        node = ArchNode(model_id="org/kimi-custom", config={})
        assert node.family == "Kimi"

    def test_family_unknown_org(self):
        node = ArchNode(model_id="unknown-org/model", config={})
        assert node.family == "unknown-org"

    def test_family_other(self):
        node = ArchNode(model_id="random", config={})
        assert node.family == "Other"

    def test_family_phi(self):
        node = ArchNode(model_id="microsoft/Phi-3-mini", config={})
        # Phi is not in the keyword map but org is 'microsoft' -> returns org
        assert node.family == "microsoft"

    def test_get_key_params(self):
        node = ArchNode(model_id="test", config={
            "hidden_size": 128,
            "num_hidden_layers": 12,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "intermediate_size": 512,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "model_type": "llama",
            "num_local_experts": 8,
        })
        params = node.get_key_params()
        assert params["hidden_size"] == 128
        assert params["is_moe"] is True
        assert params["num_experts"] == 8
        assert params["rope_type"] == "default"

    def test_children_default(self):
        node = ArchNode(model_id="test", config={})
        assert node.children == []

    def test_innovations_default(self):
        node = ArchNode(model_id="test", config={})
        assert node.innovations == []

    def test_similarity_score_default(self):
        node = ArchNode(model_id="test", config={})
        assert node.similarity_score == 1.0

    def test_metadata_default(self):
        node = ArchNode(model_id="test", config={})
        assert node.metadata == {}


# ─────────────────────────────────────────────────────────────────────────────
# ArchitectureMetrics Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestArchitectureMetrics:
    """Tests for ArchitectureMetrics dataclass."""

    def test_to_dict(self):
        m = ArchitectureMetrics(
            total_params=1e9,
            trainable_params=1e9,
            flops_per_token=1e12,
            vram_estimate_gb=16.0,
            inference_latency_ms=50.0,
            memory_bandwidth_gbs=100.0,
        )
        d = m.to_dict()
        assert d["total_params"] == 1e9
        assert d["vram_estimate_gb"] == 16.0
        assert d["inference_latency_ms"] == 50.0


# ─────────────────────────────────────────────────────────────────────────────
# EvolutionTree Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEvolutionTreeInit:
    """Tests for EvolutionTree initialization."""

    def test_default_init(self):
        tree = EvolutionTree()
        assert tree.nodes == {}
        assert len(tree.families) > 0

    def test_init_with_custom_families(self):
        custom = {"Custom": {"root": "custom/model", "members": {}}}
        tree = EvolutionTree(custom_families=custom)
        assert "Custom" in tree.families
        assert len(tree.families) >= len(DEFAULT_FAMILIES)

    def test_init_merge_families(self):
        custom = {"LLaMA": {"members": {"new-model": {}}}}
        tree = EvolutionTree(custom_families=custom)
        assert "LLaMA" in tree.families


class TestEvolutionTreeAddModel:
    """Tests for add_model method."""

    def test_add_model(self):
        tree = EvolutionTree()
        node = tree.add_model("test/model", config={"hidden_size": 128})
        assert "test/model" in tree.nodes
        assert tree.nodes["test/model"].config["hidden_size"] == 128

    def test_add_model_duplicate(self):
        tree = EvolutionTree()
        tree.add_model("test/model", config={"a": 1})
        node2 = tree.add_model("test/model", config={"b": 2})
        # Should return existing node, not overwrite
        assert tree.nodes["test/model"].config == {"a": 1}

    def test_add_model_with_parent(self):
        tree = EvolutionTree()
        tree.add_model("parent", config={})
        tree.add_model("child", config={}, parent="parent")
        assert tree.nodes["child"].parent == "parent"
        assert "child" in tree.nodes["parent"].children

    def test_add_model_with_innovations(self):
        tree = EvolutionTree()
        innov = ArchInnovation(name="GQA", description="Grouped Query", introduced_in="Test", year=2024)
        tree.add_model("test", config={}, innovations=[innov])
        assert len(tree.nodes["test"].innovations) == 1

    def test_add_model_infer_parent(self):
        tree = EvolutionTree()
        tree.add_model("Qwen/Qwen2-7B", config={})
        # Adding a known family member should infer parent
        node = tree.add_model("Qwen/Qwen1.5-7B", config={})
        assert node.parent is not None

    @patch("vitriol.evolution.tree_builder.hf_load_config")
    def test_add_model_fetch_config(self, mock_hf_load):
        tree = EvolutionTree()
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {"hidden_size": 256}
        mock_hf_load.return_value = mock_config

        node = tree.add_model("org/model")
        assert node.config["hidden_size"] == 256
        mock_hf_load.assert_called_once()

    @patch("vitriol.evolution.tree_builder.hf_load_config")
    def test_add_model_fallback_params(self, mock_hf_load):
        tree = EvolutionTree()
        mock_hf_load.side_effect = Exception("Load failed")

        # Use a model that has fallback params
        node = tree.add_model("meta-llama/Llama-2-7b")
        assert "hidden_size" in node.config
        assert node.config["hidden_size"] == 4096

    @patch("vitriol.evolution.tree_builder.hf_load_config")
    def test_add_model_no_fallback(self, mock_hf_load):
        tree = EvolutionTree()
        mock_hf_load.side_effect = Exception("Load failed")

        node = tree.add_model("unknown/model-with-no-fallback")
        assert node.config == {}


class TestEvolutionTreeBuild:
    """Tests for build method."""

    def test_build_empty(self):
        tree = EvolutionTree()
        tree.build()
        # Should build default families
        assert len(tree.nodes) > 0

    def test_build_adds_members(self):
        tree = EvolutionTree()
        tree.build()
        # All members from DEFAULT_FAMILIES should be added
        for family_data in DEFAULT_FAMILIES.values():
            for model_id in family_data.get("members", {}):
                assert model_id in tree.nodes

    def test_build_sets_children(self):
        tree = EvolutionTree()
        tree.build()
        # Children relationships should be set
        llama_family = DEFAULT_FAMILIES["LLaMA"]
        for model_id, info in llama_family["members"].items():
            if model_id in tree.nodes:
                for child in info.get("children", []):
                    if child in tree.nodes:
                        assert tree.nodes[child].parent == model_id

    def test_build_adds_innovations(self):
        tree = EvolutionTree()
        tree.build()
        # Innovations are only added for models in the 'members' dict
        for family_data in DEFAULT_FAMILIES.values():
            members = family_data.get("members", {})
            for model_id, innovations in family_data.get("innovations", {}).items():
                if model_id in tree.nodes and model_id in members:
                    assert len(tree.nodes[model_id].innovations) >= len(innovations)


class TestEvolutionTreeSubtree:
    """Tests for get_subtree."""

    def test_get_subtree(self):
        # Use a manually built tree to avoid build() complexity
        tree = EvolutionTree()
        tree.add_model("parent", config={"hidden_size": 128})
        tree.add_model("child1", config={"hidden_size": 128}, parent="parent")
        tree.add_model("grandchild", config={"hidden_size": 128}, parent="child1")
        subtree = tree.get_subtree("parent")
        assert isinstance(subtree, EvolutionTree)
        assert "parent" in subtree.nodes
        assert "child1" in subtree.nodes
        assert "grandchild" in subtree.nodes

    def test_get_subtree_missing(self):
        tree = EvolutionTree()
        subtree = tree.get_subtree("nonexistent")
        assert isinstance(subtree, EvolutionTree)


class TestEvolutionTreeToDict:
    """Tests for to_dict method."""

    def test_to_dict(self):
        tree = EvolutionTree()
        tree.add_model("test", config={"hidden_size": 128})
        data = tree.to_dict()
        assert "nodes" in data
        assert "families" in data
        assert "total_nodes" in data
        assert data["total_nodes"] == 1
        assert "test" in data["nodes"]

    def test_to_dict_empty(self):
        tree = EvolutionTree()
        data = tree.to_dict()
        assert data["total_nodes"] == 0


class TestEvolutionTreeGetFamily:
    """Tests for get_family method."""

    def test_get_family(self):
        tree = EvolutionTree()
        tree.add_model("qwen/model", config={})
        assert tree.get_family("qwen/model") == "Qwen"

    def test_get_family_missing(self):
        tree = EvolutionTree()
        assert tree.get_family("missing") is None


class TestEvolutionTreeInnovationTimeline:
    """Tests for get_innovation_timeline."""

    def test_empty_timeline(self):
        tree = EvolutionTree()
        timeline = tree.get_innovation_timeline()
        assert timeline == []

    def test_timeline_sorted(self):
        tree = EvolutionTree()
        innov1 = ArchInnovation(name="A", description="Desc", introduced_in="M", year=2023)
        innov2 = ArchInnovation(name="B", description="Desc", introduced_in="M", year=2024)
        tree.add_model("model1", config={}, innovations=[innov2])
        tree.add_model("model2", config={}, innovations=[innov1])
        timeline = tree.get_innovation_timeline()
        years = [item["year"] for item in timeline]
        assert years == sorted(years)


class TestEvolutionTreeFindCommonAncestor:
    """Tests for find_common_ancestor."""

    def test_same_model(self):
        tree = EvolutionTree()
        tree.add_model("a", config={}, parent=None)
        assert tree.find_common_ancestor("a", "a") == "a"

    def test_parent_child(self):
        tree = EvolutionTree()
        tree.add_model("parent", config={})
        tree.add_model("child", config={}, parent="parent")
        assert tree.find_common_ancestor("parent", "child") == "parent"

    def test_no_common(self):
        tree = EvolutionTree()
        tree.add_model("a", config={})
        tree.add_model("b", config={})
        assert tree.find_common_ancestor("a", "b") is None

    def test_missing_model(self):
        tree = EvolutionTree()
        assert tree.find_common_ancestor("missing1", "missing2") is None

    def test_cycle_detection(self):
        tree = EvolutionTree()
        tree.add_model("a", config={})
        tree.add_model("b", config={}, parent="a")
        # Manually create cycle
        tree.nodes["a"].parent = "b"
        result = tree.find_common_ancestor("a", "b")
        # Should not infinite loop
        assert result is not None


class TestEvolutionTreeComputeSimilarity:
    """Tests for compute_similarity."""

    def test_identical(self):
        tree = EvolutionTree()
        config = {"hidden_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4}
        tree.add_model("a", config=config)
        tree.add_model("b", config=config)
        sim = tree.compute_similarity("a", "b")
        assert sim == 1.0

    def test_different(self):
        tree = EvolutionTree()
        tree.add_model("a", config={"hidden_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4})
        tree.add_model("b", config={"hidden_size": 4096, "num_hidden_layers": 80, "num_attention_heads": 64})
        sim = tree.compute_similarity("a", "b")
        assert 0 <= sim < 1.0

    def test_missing_model(self):
        tree = EvolutionTree()
        assert tree.compute_similarity("missing1", "missing2") == 0.0

    def test_same_model(self):
        tree = EvolutionTree()
        tree.add_model("a", config={})
        assert tree.compute_similarity("a", "a") == 1.0

    def test_zero_norm(self):
        tree = EvolutionTree()
        tree.add_model("a", config={})
        tree.add_model("b", config={})
        sim = tree.compute_similarity("a", "b")
        # Both have zero features after log1p(0)
        assert sim == 0.0


class TestEvolutionTreeSaveLoad:
    """Tests for save and load methods."""

    def test_save_and_load(self):
        tree = EvolutionTree()
        tree.add_model("test", config={"hidden_size": 128})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tree.json"
            tree.save(str(path))
            assert path.exists()

            tree2 = EvolutionTree()
            tree2.load(str(path))
            assert "test" in tree2.nodes

    def test_save_content(self):
        tree = EvolutionTree()
        tree.add_model("test", config={"hidden_size": 128})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tree.json"
            tree.save(str(path))
            data = json.loads(path.read_text())
            assert data["total_nodes"] == 1
            assert "test" in data["nodes"]


class TestEvolutionTreeLoadBuiltin:
    """Tests for load_builtin_families."""

    def test_load_builtin(self):
        tree = EvolutionTree()
        tree.load_builtin_families()
        assert len(tree.families) > 0


class TestFallbackParams:
    """Tests for FALLBACK_PARAMS constant."""

    def test_fallback_exists(self):
        assert isinstance(FALLBACK_PARAMS, dict)
        assert len(FALLBACK_PARAMS) > 0

    def test_llama_fallback(self):
        assert "meta-llama/Llama-2-7b" in FALLBACK_PARAMS
        params = FALLBACK_PARAMS["meta-llama/Llama-2-7b"]
        assert params["hidden_size"] == 4096

    def test_deepseek_fallback(self):
        assert "deepseek-ai/DeepSeek-LLM-7B" in FALLBACK_PARAMS

    def test_all_have_hidden_size(self):
        for model_id, params in FALLBACK_PARAMS.items():
            assert "hidden_size" in params, f"{model_id} missing hidden_size"


class TestDefaultFamilies:
    """Tests for DEFAULT_FAMILIES constant."""

    def test_families_exist(self):
        assert isinstance(DEFAULT_FAMILIES, dict)
        assert len(DEFAULT_FAMILIES) > 0

    def test_llama_family(self):
        assert "LLaMA" in DEFAULT_FAMILIES
        family = DEFAULT_FAMILIES["LLaMA"]
        assert "root" in family
        assert "members" in family
        assert "innovations" in family

    def test_qwen_family(self):
        assert "Qwen" in DEFAULT_FAMILIES

    def test_deepseek_family(self):
        assert "DeepSeek" in DEFAULT_FAMILIES

    def test_all_families_have_root(self):
        for name, family in DEFAULT_FAMILIES.items():
            assert "root" in family, f"{name} missing root"
            assert "members" in family, f"{name} missing members"
