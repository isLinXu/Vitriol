"""
Adapter for Meta Llama 3.1 models.

Llama 3.1 introduces several architectural improvements over Llama 2.x:
- RoPE position embeddings with improved scaling
- Sliding window attention in some variants
- Key-value cache rotation optimizations
- Enhanced tokenizer with extended vocabulary

This adapter handles these Llama 3.1 specific configs.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PretrainedConfig
else:
    PretrainedConfig = object

from .base import ModelAdapter
from .registry import AdapterRegistry

logger = logging.getLogger(__name__)


class Llama31Adapter(ModelAdapter):
    """Adapter for Llama 3.1 models from Meta.
    
    Supports:
    - meta-llama/Llama-3.1-8B
    - meta-llama/Llama-3.1-70B
    - meta-llama/Llama-3.1-405B
    - meta-llama/Llama-3.1-8B-Instruct
    - meta-llama/Llama-3.1-70B-Instruct
    - meta-llama/Llama-3.1-405B-Instruct
    """

    @classmethod
    def match(cls, model_id: str, config) -> bool:
        """Check if this adapter applies to Llama 3.1 models.
        
        Args:
            model_id: HuggingFace model ID or local path
            config: PretrainedConfig from AutoConfig
            
        Returns:
            True if this is a Llama 3.1 model
        """
        # Check model_id for Llama 3.1 indicators
        if isinstance(model_id, str):
            model_id_lower = model_id.lower()
            if "llama-3.1" in model_id_lower or "llama3.1" in model_id_lower:
                return True
        
        # Check config for Llama 3.1 indicators
        model_type = getattr(config, "model_type", "")
        
        # Llama 3.1 should have model_type "llama"
        if model_type.lower() != "llama":
            return False
        
        # Check for Llama 3.1 specific attributes
        # Llama 3.1 models have specific RoPE config
        rope_scaling = getattr(config, "rope_scaling", None)
        vocab_size = getattr(config, "vocab_size", 0)
        
        # Llama 3.1 has vocab_size of 128256
        if vocab_size == 128256:
            logger.debug("Detected Llama 3.1 model by vocab_size (128256)")
            return True
        
        # Check model name in config
        model_name = getattr(config, "model_name", "").lower()
        if "llama-3.1" in model_name or "llama3.1" in model_name:
            logger.debug("Detected Llama 3.1 model by model_name")
            return True
        
        # Additional check: architectures field
        architectures = getattr(config, "architectures", [])
        if architectures and "Llama" in architectures[0]:
            # This could be Llama 2.x or 3.x, need more specific checks
            # For now, let's check rope_scaling pattern
            if rope_scaling and isinstance(rope_scaling, dict):
                rope_type = rope_scaling.get("type", "").lower()
                # Llama 3.1 typically uses "linear" or "dynamic" scaling
                if rope_type in ["linear", "dynamic", "yarn"]:
                    logger.debug(f"Detected Llama 3.1 model by RoPE scaling type: {rope_type}")
                    return True
        
        return False

    def patch_config(self, config) -> "PretrainedConfig":
        """Patch Llama 3.1 config for compatibility.
        
        Args:
            config: PretrainedConfig to patch
            
        Returns:
            Patched config
        """
        # Ensure is_encoder_decoder is False (Llama is decoder-only)
        if hasattr(config, "is_encoder_decoder"):
            config.is_encoder_decoder = False
            logger.debug("Set is_encoder_decoder = False")
        
        # Ensure proper attention config
        # Llama 3.1 may have flash_attention_2 config
        attn_impl = getattr(config, "_attn_implementation", "eager")
        if attn_impl not in ["eager", "sdpa", "flash_attention_2"]:
            config._attn_implementation = "eager"
            logger.debug(f"Normalized attention implementation to eager (was {attn_impl})")
        
        # Verify rope_scaling is properly set
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is None:
            logger.debug("No rope_scaling in config, defaulting to None (standard RoPE)")
        else:
            logger.debug(f"rope_scaling config: {rope_scaling}")
        
        # Verify max_position_embeddings
        max_pos = getattr(config, "max_position_embeddings", 8192)
        if max_pos != 8192:
            logger.debug(f"max_position_embeddings: {max_pos} (Llama 3.1 default: 8192)")
        
        return config


# Automatically register this adapter
AdapterRegistry.register(Llama31Adapter)
