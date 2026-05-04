"""Tests for additional adapters (deepseek, gemma, glm, mistral, phi, cohere, stablelm, minimax)."""

from unittest.mock import MagicMock

from vitriol.adapters.deepseek import DeepSeekAdapter
from vitriol.adapters.gemma import GemmaAdapter
from vitriol.adapters.glm import GLMAdapter
from vitriol.adapters.mistral import MistralAdapter
from vitriol.adapters.phi import PhiAdapter
from vitriol.adapters.cohere import CohereAdapter
from vitriol.adapters.stablelm import StableLMAdapter
from vitriol.adapters.minimax import MiniMaxAdapter


# ─────────────────────────────────────────────────────────────────────────────
# DeepSeek adapters
# ─────────────────────────────────────────────────────────────────────────────

class TestDeepSeekAdapter:
    def test_match_deepseek(self):
        config = MagicMock()
        config.model_type = "deepseek"
        assert DeepSeekAdapter.match("test", config) is True

    def test_match_deepseek_v2(self):
        config = MagicMock()
        config.model_type = "deepseek_v2"
        assert DeepSeekAdapter.match("test", config) is True

    def test_match_false(self):
        config = MagicMock()
        config.model_type = "llama"
        assert DeepSeekAdapter.match("test", config) is False

    def test_patch_config(self):
        adapter = DeepSeekAdapter()
        config = MagicMock()
        config.is_encoder_decoder = True
        result = adapter.patch_config(config)
        assert result.is_encoder_decoder is False



# ─────────────────────────────────────────────────────────────────────────────
# Gemma adapter
# ─────────────────────────────────────────────────────────────────────────────

class TestGemmaAdapter:
    def test_match_gemma(self):
        config = MagicMock()
        config.model_type = "gemma"
        assert GemmaAdapter.match("test", config) is True

    def test_match_gemma2(self):
        config = MagicMock()
        config.model_type = "gemma2"
        assert GemmaAdapter.match("test", config) is True

    def test_match_false(self):
        config = MagicMock()
        config.model_type = "llama"
        assert GemmaAdapter.match("test", config) is False


# ─────────────────────────────────────────────────────────────────────────────
# GLM adapter
# ─────────────────────────────────────────────────────────────────────────────

class TestGLMAdapter:
    def test_match_chatglm(self):
        config = MagicMock()
        config.model_type = "chatglm"
        assert GLMAdapter.match("test", config) is True

    def test_match_glm(self):
        config = MagicMock()
        config.model_type = "glm"
        assert GLMAdapter.match("test", config) is True

    def test_match_by_architecture(self):
        config = MagicMock()
        config.model_type = "other"
        config.architectures = ["ChatGLMModel"]
        assert GLMAdapter.match("test", config) is True

    def test_match_false(self):
        config = MagicMock()
        config.model_type = "llama"
        config.architectures = []
        assert GLMAdapter.match("test", config) is False


# ─────────────────────────────────────────────────────────────────────────────
# Mistral adapter
# ─────────────────────────────────────────────────────────────────────────────

class TestMistralAdapter:
    def test_match_mistral(self):
        config = MagicMock()
        config.model_type = "mistral"
        assert MistralAdapter.match("test", config) is True

    def test_match_false(self):
        config = MagicMock()
        config.model_type = "llama"
        assert MistralAdapter.match("test", config) is False


# ─────────────────────────────────────────────────────────────────────────────
# Phi adapter
# ─────────────────────────────────────────────────────────────────────────────

class TestPhiAdapter:
    def test_match_phi(self):
        config = MagicMock()
        config.model_type = "phi"
        assert PhiAdapter.match("test", config) is True

    def test_match_phi3(self):
        config = MagicMock()
        config.model_type = "phi3"
        assert PhiAdapter.match("test", config) is True

    def test_match_false(self):
        config = MagicMock()
        config.model_type = "llama"
        assert PhiAdapter.match("test", config) is False


# ─────────────────────────────────────────────────────────────────────────────
# Cohere adapter
# ─────────────────────────────────────────────────────────────────────────────

class TestCohereAdapter:
    def test_match_cohere(self):
        config = MagicMock()
        config.model_type = "cohere"
        assert CohereAdapter.match("test", config) is True

    def test_match_false(self):
        config = MagicMock()
        config.model_type = "llama"
        assert CohereAdapter.match("test", config) is False


# ─────────────────────────────────────────────────────────────────────────────
# StableLM adapter
# ─────────────────────────────────────────────────────────────────────────────

class TestStableLMAdapter:
    def test_match_stablelm(self):
        config = MagicMock()
        config.model_type = "stablelm"
        assert StableLMAdapter.match("test", config) is True

    def test_match_false(self):
        config = MagicMock()
        config.model_type = "llama"
        assert StableLMAdapter.match("test", config) is False


# ─────────────────────────────────────────────────────────────────────────────
# MiniMax adapter
# ─────────────────────────────────────────────────────────────────────────────

class TestMiniMaxAdapter:
    def test_match_minimax(self):
        config = MagicMock()
        config.model_type = "minimax"
        assert MiniMaxAdapter.match("test", config) is True

    def test_match_false(self):
        config = MagicMock()
        config.model_type = "llama"
        assert MiniMaxAdapter.match("test", config) is False

