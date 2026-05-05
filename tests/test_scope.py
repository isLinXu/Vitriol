"""Tests for Scope.model_info() — mocked, no network required."""
import pytest
from unittest.mock import patch, MagicMock
from vitriol.core.scope import Scope, ModelInfo


def _make_mock_config():
    """Create a mock config that mimics a Qwen2.5-0.5B config."""
    cfg = MagicMock()
    cfg._name_or_path = "Qwen/Qwen2.5-0.5B"
    cfg.model_type = "qwen2"
    cfg.architectures = ["Qwen2ForCausalLM"]
    cfg.vocab_size = 151936
    cfg.hidden_size = 896
    cfg.num_hidden_layers = 24
    cfg.num_attention_heads = 14
    cfg.num_key_value_heads = 2
    cfg.intermediate_size = 4864
    cfg.max_position_embeddings = 131072
    cfg.rope_theta = 1000000.0
    cfg.rope_scaling = None
    cfg.num_experts = 0
    cfg.num_experts_per_tok = 0
    cfg.n_routed_experts = 0
    cfg.n_shared_experts = 0
    cfg.moe_intermediate_size = 0
    cfg.shared_expert_intermediate_size = 0
    return cfg


def _make_mock_tokenizer():
    """Create a mock tokenizer."""
    tok = MagicMock()
    tok.__class__.__name__ = "AutoTokenizer"
    tok.name_or_path = "Qwen/Qwen2.5-0.5B"
    return tok


@pytest.fixture
def scope():
    """Create a Scope instance with mocked model loading."""
    mock_config = _make_mock_config()
    mock_model = MagicMock()
    mock_model.config = mock_config
    mock_tokenizer = _make_mock_tokenizer()

    with patch.object(Scope, '_load_model_and_tokenizer'):
        s = Scope.__new__(Scope)
        s.model_id_or_path = "Qwen/Qwen2.5-0.5B"
        s.trust_remote_code = False
        s.allow_network = False
        s.local_files_only = True
        s._model = mock_model
        s._tokenizer = mock_tokenizer
        s._config = mock_config
    return s


class TestScopeModelInfo:
    """Test suite for Scope.model_info()."""

    def test_model_info_returns_model_info(self, scope: Scope):
        info = scope.model_info()
        assert isinstance(info, ModelInfo), f"Expected ModelInfo, got {type(info)}"

    def test_model_info_model_name(self, scope: Scope):
        info = scope.model_info()
        assert isinstance(info.model_name, str)

    def test_model_info_model_type(self, scope: Scope):
        info = scope.model_info()
        assert isinstance(info.model_type, str)

    def test_model_info_architecture(self, scope: Scope):
        info = scope.model_info()
        assert isinstance(info.architecture, str)

    def test_model_info_vocab_size(self, scope: Scope):
        info = scope.model_info()
        assert isinstance(info.vocab_size, int)
        assert info.vocab_size >= 0

    def test_model_info_hidden_size(self, scope: Scope):
        info = scope.model_info()
        assert isinstance(info.hidden_size, int)
        assert info.hidden_size > 0

    def test_model_info_num_hidden_layers(self, scope: Scope):
        info = scope.model_info()
        assert isinstance(info.num_hidden_layers, int)
        assert info.num_hidden_layers > 0

    def test_model_info_num_attention_heads(self, scope: Scope):
        info = scope.model_info()
        assert isinstance(info.num_attention_heads, int)
        assert info.num_attention_heads > 0

    def test_model_info_num_key_value_heads(self, scope: Scope):
        info = scope.model_info()
        assert isinstance(info.num_key_value_heads, int)
        assert info.num_key_value_heads > 0

    def test_model_info_intermediate_size(self, scope: Scope):
        info = scope.model_info()
        assert isinstance(info.intermediate_size, int)
        assert info.intermediate_size > 0

    def test_model_info_max_position_embeddings(self, scope: Scope):
        info = scope.model_info()
        assert isinstance(info.max_position_embeddings, int)
        assert info.max_position_embeddings > 0

    def test_model_info_rope_theta(self, scope: Scope):
        info = scope.model_info()
        assert isinstance(info.rope_theta, float)
        assert info.rope_theta > 0

    def test_model_info_rope_scaling(self, scope: Scope):
        info = scope.model_info()
        assert hasattr(info, 'rope_scaling')

    def test_model_info_num_experts(self, scope: Scope):
        info = scope.model_info()
        assert isinstance(info.num_experts, int)
        assert info.num_experts >= 0

    def test_model_info_num_experts_per_tok(self, scope: Scope):
        info = scope.model_info()
        assert isinstance(info.num_experts_per_tok, int)
        assert info.num_experts_per_tok >= 0

    def test_model_info_n_routed_experts(self, scope: Scope):
        info = scope.model_info()
        assert isinstance(info.n_routed_experts, int)
        assert info.n_routed_experts >= 0

    def test_model_info_n_shared_experts(self, scope: Scope):
        info = scope.model_info()
        assert isinstance(info.n_shared_experts, int)
        assert info.n_shared_experts >= 0

    def test_model_info_moe_intermediate_size(self, scope: Scope):
        info = scope.model_info()
        assert isinstance(info.moe_intermediate_size, int)
        assert info.moe_intermediate_size >= 0

    def test_model_info_shared_expert_intermediate_size(self, scope: Scope):
        info = scope.model_info()
        assert isinstance(info.shared_expert_intermediate_size, int)
        assert info.shared_expert_intermediate_size >= 0

    def test_model_info_head_dim(self, scope: Scope):
        info = scope.model_info()
        assert hasattr(info, 'head_dim')
        if info.num_attention_heads > 0:
            assert info.head_dim == info.hidden_size // info.num_attention_heads

    def test_model_info_vitriol_score(self, scope: Scope):
        info = scope.model_info()
        assert hasattr(info, 'vitriol_score')
        assert isinstance(info.vitriol_score, float)
        assert info.vitriol_score >= 0

    def test_model_info_attention_diversity_score(self, scope: Scope):
        info = scope.model_info()
        assert hasattr(info, 'attention_diversity_score')
        assert isinstance(info.attention_diversity_score, float)
        assert info.attention_diversity_score >= 0

    def test_model_info_tokenizer_class(self, scope: Scope):
        info = scope.model_info()
        assert isinstance(info.tokenizer_class, str)

    def test_model_info_tokenizer_model_path(self, scope: Scope):
        info = scope.model_info()
        assert isinstance(info.tokenizer_model_path, str)

    def test_model_info_tokenizer_config_file(self, scope: Scope):
        info = scope.model_info()
        assert isinstance(info.tokenizer_config_file, str)

    def test_model_info_raises_without_load(self):
        """Test that model_info() raises RuntimeError when model is not loaded."""
        with patch.object(Scope, '_load_model_and_tokenizer'):
            s = Scope.__new__(Scope)
            s.model_id_or_path = "test"
            s.trust_remote_code = False
            s.allow_network = False
            s.local_files_only = True
            s._model = None
            s._tokenizer = None
            s._config = None
        with pytest.raises(RuntimeError, match="Model not loaded"):
            s.model_info()
