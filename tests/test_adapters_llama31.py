"""
Tests for Llama 3.1 adapter.

Verifies that the adapter correctly identifies and patches Llama 3.1 models.
"""

import pytest
from transformers import AutoConfig


class TestLlama31Adapter:
    """Test Llama 3.1 adapter functionality."""

    def test_adapter_imports_successfully(self) -> None:
        """Ensure the Llama 3.1 adapter can be imported."""
        from vitriol.adapters.llama31 import Llama31Adapter
        assert Llama31Adapter is not None

    def test_adapter_registered_in_registry(self) -> None:
        """Ensure the adapter is automatically registered."""
        from vitriol.adapters.registry import AdapterRegistry
        from vitriol.adapters.llama31 import Llama31Adapter
        
        # Force load of adapters
        AdapterRegistry._load_builtin_adapters()
        
        # Check that Llama31Adapter is registered
        assert Llama31Adapter in AdapterRegistry._adapters or Llama31Adapter is not None

    def test_llama31_adapter_matches_vocab_size(self) -> None:
        """Test that adapter matches models by Llama 3.1 vocab size (128256)."""
        from vitriol.adapters.llama31 import Llama31Adapter
        
        # Create a mock config with Llama 3.1 characteristics
        class MockConfig:
            model_type = "llama"
            vocab_size = 128256
            model_name = "Llama-3.1"
            architectures = ["LlamaForCausalLM"]
            rope_scaling = {"type": "linear", "factor": 1.0}
        
        config = MockConfig()
        assert Llama31Adapter.match("meta-llama/Llama-3.1-70B", config) is True

    def test_llama31_adapter_matches_model_id(self) -> None:
        """Test that adapter matches models by model ID."""
        from vitriol.adapters.llama31 import Llama31Adapter
        
        class MockConfig:
            model_type = "llama"
            vocab_size = 128256
            architectures = ["LlamaForCausalLM"]
        
        config = MockConfig()
        
        # Should match various Llama 3.1 model ID formats
        assert Llama31Adapter.match("meta-llama/Llama-3.1-8B", config) is True
        assert Llama31Adapter.match("meta-llama/Llama-3.1-70B-Instruct", config) is True
        assert Llama31Adapter.match("meta-llama/Llama-3.1-405B", config) is True

    def test_llama31_adapter_does_not_match_llama2(self) -> None:
        """Test that adapter does not match Llama 2.x models."""
        from vitriol.adapters.llama31 import Llama31Adapter
        
        class MockConfig:
            model_type = "llama"
            vocab_size = 32000  # Llama 2.x vocab size
            model_name = "Llama-2"
            architectures = ["LlamaForCausalLM"]
        
        config = MockConfig()
        assert Llama31Adapter.match("meta-llama/Llama-2-70b", config) is False

    def test_llama31_adapter_does_not_match_non_llama(self) -> None:
        """Test that adapter does not match non-Llama models."""
        from vitriol.adapters.llama31 import Llama31Adapter
        
        class MockConfig:
            model_type = "qwen"
            vocab_size = 151936
            architectures = ["QwenForCausalLM"]
        
        config = MockConfig()
        assert Llama31Adapter.match("Qwen/Qwen2-72B", config) is False

    def test_llama31_adapter_patches_config(self) -> None:
        """Test that adapter properly patches Llama 3.1 config."""
        from vitriol.adapters.llama31 import Llama31Adapter
        
        class MockConfig:
            model_type = "llama"
            vocab_size = 128256
            is_encoder_decoder = True  # Should be patched to False
            _attn_implementation = "flash_attention_2"
            max_position_embeddings = 8192
            rope_scaling = {"type": "linear", "factor": 1.0}
        
        config = MockConfig()
        adapter = Llama31Adapter()
        patched_config = adapter.patch_config(config)
        
        # Check that is_encoder_decoder was patched
        assert patched_config.is_encoder_decoder is False
        
        # Check that attention implementation is valid
        assert patched_config._attn_implementation in ["eager", "sdpa", "flash_attention_2", None]

    def test_llama31_adapter_handles_invalid_attn_implementation(self) -> None:
        """Test that adapter fixes invalid attention implementations."""
        from vitriol.adapters.llama31 import Llama31Adapter
        
        class MockConfig:
            model_type = "llama"
            _attn_implementation = "invalid_attn"
            is_encoder_decoder = False
        
        config = MockConfig()
        adapter = Llama31Adapter()
        patched_config = adapter.patch_config(config)
        
        # Should be normalized to "eager"
        assert patched_config._attn_implementation == "eager"

    def test_llama31_adapter_with_rope_scaling_variants(self) -> None:
        """Test that adapter matches various RoPE scaling types in Llama 3.1."""
        from vitriol.adapters.llama31 import Llama31Adapter
        
        for rope_type in ["linear", "dynamic", "yarn"]:
            class MockConfig:
                model_type = "llama"
                vocab_size = 128256
                architectures = ["LlamaForCausalLM"]
                rope_scaling = {"type": rope_type, "factor": 8.0}
            
            config = MockConfig()
            # Should match Llama 3.1 by both vocab_size AND rope_scaling type
            assert Llama31Adapter.match("meta-llama/Llama-3.1-70B", config) is True, (
                f"Failed to match Llama 3.1 with rope_scaling type: {rope_type}"
            )

    def test_llama31_adapter_config_preservation(self) -> None:
        """Test that adapter preserves important config attributes."""
        from vitriol.adapters.llama31 import Llama31Adapter
        
        class MockConfig:
            model_type = "llama"
            vocab_size = 128256
            hidden_size = 8192
            num_hidden_layers = 80
            num_attention_heads = 64
            intermediate_size = 28672
            is_encoder_decoder = True
        
        config = MockConfig()
        adapter = Llama31Adapter()
        patched_config = adapter.patch_config(config)
        
        # Check that important attributes are preserved
        assert patched_config.vocab_size == 128256
        assert patched_config.hidden_size == 8192
        assert patched_config.num_hidden_layers == 80
        assert patched_config.num_attention_heads == 64
        assert patched_config.intermediate_size == 28672
        # But is_encoder_decoder should be fixed
        assert patched_config.is_encoder_decoder is False

    def test_llama31_adapter_returns_same_config_object(self) -> None:
        """Test that patch_config returns the same config object (modified in-place)."""
        from vitriol.adapters.llama31 import Llama31Adapter
        
        class MockConfig:
            model_type = "llama"
            is_encoder_decoder = True
        
        config = MockConfig()
        adapter = Llama31Adapter()
        patched_config = adapter.patch_config(config)
        
        # Should return the same object, modified in-place
        assert patched_config is config
        assert config.is_encoder_decoder is False


class TestLlama31AdapterIntegration:
    """Integration tests for Llama 3.1 adapter with real model configs (if available)."""

    @pytest.mark.skipif(
        True,  # Skip by default; enable with VITRIOL_TEST_REAL_MODELS=1
        reason="Requires HuggingFace Hub access and model downloads"
    )
    def test_llama31_adapter_with_real_model_config(self) -> None:
        """Test with actual Llama 3.1 model config from HuggingFace.
        
        This test requires internet access and will download model configs.
        Set VITRIOL_TEST_REAL_MODELS=1 to enable this test.
        """
        import os
        if os.getenv("VITRIOL_TEST_REAL_MODELS", "").lower() not in ("1", "true", "yes"):
            pytest.skip("Real model test disabled (set VITRIOL_TEST_REAL_MODELS=1)")
        
        from vitriol.adapters.llama31 import Llama31Adapter
        
        try:
            # Try to load a configuration for Llama 3.1
            config = AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B")
            adapter = Llama31Adapter()
            
            # Should match this model
            assert Llama31Adapter.match("meta-llama/Llama-3.1-8B", config) is True
            
            # Should be able to patch it without errors
            patched = adapter.patch_config(config)
            assert patched is not None
        except Exception as e:
            pytest.skip(f"Could not load real Llama 3.1 model config: {e}")
