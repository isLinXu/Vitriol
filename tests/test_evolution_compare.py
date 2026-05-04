"""Tests for vitriol.evolution.compare module."""

import pytest
from unittest.mock import Mock
from dataclasses import dataclass, field
from typing import List, Dict, Any

from vitriol.evolution.compare import (
    ComparisonResult,
    ArchComparator,
    ComparisonReport,
    ATTENTION_TYPES,
    FFN_TYPES,
    POSITION_ENCODING,
)


# Mock ArchNode for testing
@dataclass
class MockInnovation:
    name: str


@dataclass
class MockArchNode:
    model_id: str
    model_name: str
    config: Dict[str, Any]
    innovations: List[MockInnovation] = field(default_factory=list)

    def get_key_params(self) -> Dict[str, Any]:
        return {
            "hidden_size": self.config.get("hidden_size", 0),
            "num_hidden_layers": self.config.get("num_hidden_layers", 0),
            "num_attention_heads": self.config.get("num_attention_heads", 0),
            "model_type": self.config.get("model_type", "unknown"),
            "is_moe": self.config.get("num_local_experts", 0) > 1,
            "num_key_value_heads": self.config.get("num_key_value_heads", 0),
        }


class TestComparisonResult:
    """Tests for ComparisonResult dataclass."""

    def test_creation(self):
        """Test ComparisonResult creation."""
        result = ComparisonResult(
            model1_id="org/model1",
            model2_id="org/model2",
            similarity_score=75.0
        )
        assert result.model1_id == "org/model1"
        assert result.model2_id == "org/model2"
        assert result.similarity_score == 75.0
        assert result.param_differences == {}
        assert result.shared_features == []

    def test_to_dict(self):
        """Test to_dict method."""
        result = ComparisonResult(
            model1_id="m1",
            model2_id="m2",
            similarity_score=80.0,
            shared_features=["feature1"]
        )
        d = result.to_dict()
        assert d["model1"] == "m1"
        assert d["model2"] == "m2"
        assert d["similarity_score"] == 80.0
        assert d["shared_features"] == ["feature1"]


class TestArchComparator:
    """Tests for ArchComparator class."""

    def test_init(self):
        """Test initialization."""
        comparator = ArchComparator()
        assert comparator is not None

    def test_compare_identical_configs(self):
        """Test comparing identical configs."""
        config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "model_type": "llama"
        }
        node1 = MockArchNode("org/a", "Model A", config)
        node2 = MockArchNode("org/b", "Model B", config)

        comparator = ArchComparator()
        result = comparator.compare(node1, node2)

        assert result.similarity_score > 90
        assert len(result.param_differences) == 0

    def test_compare_different_configs(self):
        """Test comparing different configs."""
        config1 = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "model_type": "llama"
        }
        config2 = {
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "model_type": "gpt"
        }
        node1 = MockArchNode("org/a", "Model A", config1)
        node2 = MockArchNode("org/b", "Model B", config2)

        comparator = ArchComparator()
        result = comparator.compare(node1, node2)

        assert result.similarity_score < 80
        assert len(result.param_differences) > 0

    def test_compare_with_innovations(self):
        """Test comparing with innovations."""
        config = {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32}
        node1 = MockArchNode("org/a", "Model A", config, [MockInnovation("MLA")])
        node2 = MockArchNode("org/b", "Model B", config, [MockInnovation("MoE")])

        comparator = ArchComparator()
        result = comparator.compare(node1, node2)

        assert "MLA" in result.model1_innovations
        assert "MoE" in result.model2_innovations

    def test_compare_params(self):
        """Test compare_params public API."""
        config1 = {"hidden_size": 4096, "num_hidden_layers": 32}
        config2 = {"hidden_size": 2048, "num_hidden_layers": 32}

        comparator = ArchComparator()
        result = comparator.compare_params(config1, config2)

        assert result["params_match"] is False
        assert "hidden_size" in result["differences"]

    def test_compare_params_identical(self):
        """Test compare_params with identical configs."""
        config = {"hidden_size": 4096, "num_hidden_layers": 32}

        comparator = ArchComparator()
        result = comparator.compare_params(config, config)

        assert result["params_match"] is True
        assert len(result["differences"]) == 0

    def test_compare_attention_mha(self):
        """Test attention comparison for MHA."""
        config1 = {"num_attention_heads": 32, "num_key_value_heads": 32}
        config2 = {"num_attention_heads": 32, "num_key_value_heads": 1}

        comparator = ArchComparator()
        result = comparator.compare_attention(config1, config2)

        assert result["attention_type_1"] == "MHA"
        assert result["attention_type_2"] == "MQA"

    def test_compare_attention_gqa(self):
        """Test attention comparison for GQA."""
        config1 = {"num_attention_heads": 32, "num_key_value_heads": 8}

        comparator = ArchComparator()
        result = comparator.compare_attention(config1, config1)

        assert result["attention_type_1"] == "GQA"

    def test_compare_attention_unknown(self):
        """Test attention comparison with unknown config."""
        config = {}

        comparator = ArchComparator()
        result = comparator.compare_attention(config, config)

        assert result["attention_type_1"] == "unknown"

    def test_calculate_similarity_identical(self):
        """Test similarity for identical models."""
        config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "model_type": "llama",
            "num_key_value_heads": 32
        }
        node1 = MockArchNode("org/a", "A", config)
        node2 = MockArchNode("org/b", "B", config)

        comparator = ArchComparator()
        similarity = comparator._calculate_similarity(node1, node2)

        assert similarity == 1.0

    def test_calculate_similarity_different(self):
        """Test similarity for very different models."""
        config1 = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "model_type": "llama",
            "num_key_value_heads": 32
        }
        config2 = {
            "hidden_size": 1024,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "model_type": "gpt",
            "num_key_value_heads": 12
        }
        node1 = MockArchNode("org/a", "A", config1)
        node2 = MockArchNode("org/b", "B", config2)

        comparator = ArchComparator()
        similarity = comparator._calculate_similarity(node1, node2)

        assert 0 < similarity < 1.0

    def test_get_attention_type(self):
        """Test attention type detection."""
        comparator = ArchComparator()

        assert comparator._get_attention_type({"num_key_value_heads": 0}) == "multi_head"
        assert comparator._get_attention_type({"num_key_value_heads": 1}) == "multi_query"
        assert comparator._get_attention_type({"num_key_value_heads": 8, "num_attention_heads": 32}) == "grouped_query"
        assert comparator._get_attention_type({"num_key_value_heads": 32, "num_attention_heads": 32}) == "multi_head"

    def test_extract_features_mha(self):
        """Test feature extraction for MHA."""
        config = {"num_attention_heads": 32, "num_key_value_heads": 32}

        comparator = ArchComparator()
        features = comparator._extract_features(config)

        assert "Multi-Head Attention" in features
        assert "RoPE" in features

    def test_extract_features_gqa(self):
        """Test feature extraction for GQA."""
        config = {"num_attention_heads": 32, "num_key_value_heads": 8}

        comparator = ArchComparator()
        features = comparator._extract_features(config)

        assert "Grouped Query Attention" in features

    def test_extract_features_mqa(self):
        """Test feature extraction for MQA."""
        config = {"num_attention_heads": 32, "num_key_value_heads": 1}

        comparator = ArchComparator()
        features = comparator._extract_features(config)

        assert "Multi-Query Attention" in features

    def test_extract_features_moe(self):
        """Test feature extraction for MoE."""
        config = {"num_attention_heads": 32, "num_local_experts": 8}

        comparator = ArchComparator()
        features = comparator._extract_features(config)

        assert any("Mixture of Experts" in f for f in features)

    def test_extract_features_swiglu(self):
        """Test feature extraction for SwiGLU."""
        config = {"activation_function": "swiglu"}

        comparator = ArchComparator()
        features = comparator._extract_features(config)

        assert "SwiGLU Activation" in features

    def test_extract_features_sliding_window(self):
        """Test feature extraction for sliding window."""
        config = {"sliding_window": 4096}

        comparator = ArchComparator()
        features = comparator._extract_features(config)

        assert any("Sliding Window" in f for f in features)

    def test_generate_summary_high_similarity(self):
        """Test summary for high similarity."""
        node1 = MockArchNode("org/a", "Model A", {})
        node2 = MockArchNode("org/b", "Model B", {})

        comparator = ArchComparator()
        summary = comparator._generate_summary(
            node1, node2, 0.9, {}, set(), set(), set(), [], []
        )

        assert "highly similar" in summary.lower()

    def test_generate_summary_low_similarity(self):
        """Test summary for low similarity."""
        node1 = MockArchNode("org/a", "Model A", {})
        node2 = MockArchNode("org/b", "Model B", {})

        comparator = ArchComparator()
        summary = comparator._generate_summary(
            node1, node2, 0.3, {}, set(), set(), set(), [], []
        )

        assert "distinct" in summary.lower()

    def test_identify_pros_gqa(self):
        """Test pros identification for GQA."""
        node = MockArchNode("org/a", "Model A", {"num_experts": 0})

        comparator = ArchComparator()
        pros = comparator._identify_pros(node, node, {"Grouped Query Attention"}, [])

        assert any("GQA reduces KV cache" in p for p in pros)

    def test_identify_pros_mla(self):
        """Test pros identification for MLA."""
        node = MockArchNode("org/a", "Model A", {"num_experts": 0})

        comparator = ArchComparator()
        pros = comparator._identify_pros(node, node, {"Multi-head Latent Attention"}, [])

        assert any("MLA" in p for p in pros)

    def test_identify_pros_larger_model(self):
        """Test pros for larger hidden size."""
        node1 = MockArchNode("org/a", "Model A", {"hidden_size": 8192})
        node2 = MockArchNode("org/b", "Model B", {"hidden_size": 4096})

        comparator = ArchComparator()
        pros = comparator._identify_pros(node1, node2, set(), [])

        assert any("Larger hidden dimension" in p for p in pros)

    def test_compare_with_moe_config(self):
        """Test comparison with MoE config."""
        config1 = {"hidden_size": 4096, "num_local_experts": 8, "n_routed_experts": 8}
        config2 = {"hidden_size": 4096, "num_local_experts": 1}
        node1 = MockArchNode("org/a", "A", config1)
        node2 = MockArchNode("org/b", "B", config2)

        comparator = ArchComparator()
        result = comparator.compare(node1, node2)

        assert any("Mixture of Experts" in f for f in result.unique_to_model1)


class TestComparisonReport:
    """Tests for ComparisonReport class."""

    def test_to_markdown(self):
        """Test markdown formatting."""
        result = ComparisonResult(
            model1_id="org/model1",
            model2_id="org/model2",
            similarity_score=75.0,
            param_differences={
                "hidden_size": {"model1": 4096, "model2": 2048, "difference": "+100.0%"}
            },
            shared_features=["Feature1"],
            unique_to_model1=["Unique1"],
            unique_to_model2=["Unique2"]
        )

        markdown = ComparisonReport.to_markdown(result)
        assert "# Architecture Comparison Report" in markdown
        assert "75.0%" in markdown
        assert "hidden_size" in markdown
        assert "Feature1" in markdown
        assert "Unique1" in markdown

    def test_to_markdown_empty(self):
        """Test markdown with empty result."""
        result = ComparisonResult(
            model1_id="org/model1",
            model2_id="org/model2",
            similarity_score=0.0
        )

        markdown = ComparisonReport.to_markdown(result)
        assert "# Architecture Comparison Report" in markdown
        assert "_No shared architectural features_" in markdown

    def test_to_json(self):
        """Test JSON formatting."""
        result = ComparisonResult(
            model1_id="org/model1",
            model2_id="org/model2",
            similarity_score=80.0
        )

        json_str = ComparisonReport.to_json(result)
        assert '"model1": "org/model1"' in json_str
        assert "80.0" in json_str

    def test_to_html(self):
        """Test HTML formatting."""
        result = ComparisonResult(
            model1_id="org/model1",
            model2_id="org/model2",
            similarity_score=85.0,
            pros_model1=["Pro1"],
            pros_model2=["Pro2"]
        )

        html = ComparisonReport.to_html(result)
        assert "<!DOCTYPE html>" in html
        assert "85.0%" in html
        assert "Pro1" in html
        assert "Pro2" in html

    def test_to_html_empty(self):
        """Test HTML with empty features."""
        result = ComparisonResult(
            model1_id="org/model1",
            model2_id="org/model2",
            similarity_score=50.0
        )

        html = ComparisonReport.to_html(result)
        assert "<!DOCTYPE html>" in html
        assert "<p>None</p>" in html


class TestConstants:
    """Tests for module constants."""

    def test_attention_types(self):
        """Test attention type definitions."""
        assert "multi_head" in ATTENTION_TYPES
        assert "multi_query" in ATTENTION_TYPES
        assert "grouped_query" in ATTENTION_TYPES

    def test_ffn_types(self):
        """Test FFN type definitions."""
        assert "standard" in FFN_TYPES
        assert "swiglu" in FFN_TYPES
        assert "moe" in FFN_TYPES

    def test_position_encoding(self):
        """Test position encoding definitions."""
        assert "rope" in POSITION_ENCODING
        assert "alibi" in POSITION_ENCODING
