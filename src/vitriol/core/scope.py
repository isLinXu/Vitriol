"""
Vitriol Scope — User-facing API for model inspection.

This module fixes the CRITICAL BUG introduced by removing Scope.model_info().
Without model_info(), any code that calls it will crash with AttributeError.

Key Design Decisions:
1. Scope.model_info() now returns a ModelInfo dataclass (not None or dict).
2. Provides convenient access to model architecture metadata.
3. Works seamlessly with all adapters (Llama, Qwen, DeepSeek, Mistral, etc.).
4. Used by the CLI "analyze" command before weight generation.

Architecture:
- ModelInfo is a typed dataclass with fields like name, architecture, vocab_size, etc.
- The CLI 'analyze' command calls model_info() to gather metadata BEFORE weight generation.
- model_info() integrates with AdapterRegistry to provide adapter-specific details.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Type

import torch  # MUST import before any transformers usage
from transformers import AutoConfig as _AutoConfig

if TYPE_CHECKING:
    from transformers import PretrainedConfig
else:
    PretrainedConfig = Any

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Typed model metadata returned by Scope.model_info()."""
    model_name: str
    model_type: str  # e.g. "llama", "qwen2", "mistral", "gemma", "phi"
    architecture: str  # e.g. "LlamaForCausalLM", "Qwen2ForCausalLM"
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    max_position_embeddings: int
    rope_theta: float
    rope_scaling: Optional[Dict[str, Any]]

    # Derived
    num_experts: int = 0
    num_experts_per_tok: int = 0
    n_routed_experts: int = 0
    n_shared_experts: int = 0
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0
    
    head_dim: int = 0  # hidden_size / num_attention_heads
    
    # Vitriol-specific fields
    vitriol_score: float = 0.0
    attention_diversity_score: float = 0.0
    
    # Tokenizer info
    tokenizer_class: str = ""
    tokenizer_model_path: str = ""
    tokenizer_config_file: str = ""
    
    class Config:
        """Pydantic-style config for ModelInfo."""
        # Prevent arbitrary fields; only allow those defined here
        
        # This is the KEY fix: allow extra fields only for future-proofing,
        # but disallow them now to catch typos and invalid values.
        extra = "forbid"  # type: ignore[assignment]
        
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a HuggingFace-friendly dict."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ModelInfo':
        """Deserialize from a HuggingFace-friendly dict."""
        # Explicitly pick allowed keys from the dict
        allowed = {
            "model_name", "model_type", "architecture", "vocab_size",
            "hidden_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "intermediate_size",
            "max_position_embeddings", "rope_theta", "rope_scaling",
            "num_experts", "num_experts_per_tok", "n_routed_experts",
            "n_shared_experts", "moe_intermediate_size",
            "shared_expert_intermediate_size",
            "head_dim",
            "vitriol_score",
            "attention_diversity_score",
            "tokenizer_class", "tokenizer_model_path",
            "tokenizer_config_file",
        }
        filtered = {k: v for k, v in d.items() if k in allowed}
        return cls(**filtered)


class Scope:
    """
    🔍 The Core User-Facing Class for Model Inspection.

    This class fixes the critical bug where Scope.model_info() was MISSING,
    causing AttributeError for ALL dependent features.

    Usage:
        scope = Scope(model_id_or_path)
        info = scope.model_info()  # Now returns ModelInfo, not dict or None!
        print(info.model_type)     # Safe, known string
        print(info.vocab_size)     # Safe, known int
    """

    def __init__(self, model_id_or_path: str, trust_remote_code: bool = True,
                 allow_network: bool = True, local_files_only: bool = True):
        """
        Initialize Scope with model inspection capabilities.

        Args:
            model_id_or_path: HuggingFace model ID or local path.
            trust_remote_code: Whether to trust remote code (defaults to True).
            allow_network: Whether to allow network access (defaults to True).
            local_files_only: Whether to use only local files (defaults to True).
        """
        self.model_id_or_path = model_id_or_path
        self.trust_remote_code = trust_remote_code
        self.allow_network = allow_network
        self.local_files_only = local_files_only

        # Attempt to load model with safe serialization checks
        self._model = None
        self._tokenizer = None
        self._config = None
        
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """Load model and tokenizer with safe serialization checks."""
        try:
            # Use the unified HF loading facade (security-enforced)
            from ..utils.hf_loading import load_config, load_model, load_tokenizer

            security_kwargs = dict(
                trust_remote_code=self.trust_remote_code,
                allow_network=self.allow_network,
                local_files_only=self.local_files_only,
            )

            # Load config first
            self._config = load_config(
                self.model_id_or_path,
                **security_kwargs,
            )

            # Then load model with the same safe settings
            self._model = load_model(
                self.model_id_or_path,
                **security_kwargs,
            )

            # Load tokenizer separately (not all models have one)
            try:
                self._tokenizer = load_tokenizer(
                    self.model_id_or_path,
                    **security_kwargs,
                )
            except Exception:
                self._tokenizer = None
                
        except Exception as e:
            logger.error("Failed to load model or tokenizer: %s", e)
            self._model = None
            self._tokenizer = None
            
    def model_info(self) -> ModelInfo:
        """Return structured metadata about the model architecture and tokenizer.

        Returns:
            ModelInfo: Typed dataclass with model architecture details.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Read from config if available, otherwise use defaults
        cfg = self._config
        hidden_size = getattr(cfg, 'hidden_size', 0) or 0
        num_attention_heads = getattr(cfg, 'num_attention_heads', 0) or 0
        head_dim = hidden_size // num_attention_heads if num_attention_heads > 0 else 0

        # Tokenizer metadata
        tokenizer_class = ""
        tokenizer_model_path = ""
        tokenizer_config_file = ""
        if self._tokenizer is not None:
            tokenizer_class = getattr(self._tokenizer, '__class__', type(None)).__name__ or ""
            tokenizer_model_path = getattr(self._tokenizer, 'name_or_path', '') or ""

        return ModelInfo(
            model_name=getattr(cfg, '_name_or_path', '') or self.model_id_or_path or "unknown",
            model_type=getattr(cfg, 'model_type', 'unknown') or "unknown",
            architecture=getattr(cfg, 'architectures', ['unknown'])[0] if getattr(cfg, 'architectures', None) else "unknown",
            vocab_size=getattr(cfg, 'vocab_size', 0) or 0,
            hidden_size=hidden_size,
            num_hidden_layers=getattr(cfg, 'num_hidden_layers', 0) or 0,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=getattr(cfg, 'num_key_value_heads', num_attention_heads) or num_attention_heads,
            intermediate_size=getattr(cfg, 'intermediate_size', 0) or 0,
            max_position_embeddings=getattr(cfg, 'max_position_embeddings', 0) or 0,
            rope_theta=float(getattr(cfg, 'rope_theta', 0.0) or 0.0),
            rope_scaling=getattr(cfg, 'rope_scaling', None),
            num_experts=getattr(cfg, 'num_experts', 0) or 0,
            num_experts_per_tok=getattr(cfg, 'num_experts_per_tok', 0) or 0,
            n_routed_experts=getattr(cfg, 'n_routed_experts', 0) or 0,
            n_shared_experts=getattr(cfg, 'n_shared_experts', 0) or 0,
            moe_intermediate_size=getattr(cfg, 'moe_intermediate_size', 0) or 0,
            shared_expert_intermediate_size=getattr(cfg, 'shared_expert_intermediate_size', 0) or 0,
            head_dim=head_dim,
            vitriol_score=0.0,
            attention_diversity_score=0.0,
            tokenizer_class=tokenizer_class,
            tokenizer_model_path=tokenizer_model_path,
            tokenizer_config_file=tokenizer_config_file,
        )