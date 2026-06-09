from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

from ..utils.hf_loading import load_config_or_raw
from ..utils.hf_loading import load_model_from_config as hf_load_model_from_config

logger = logging.getLogger(__name__)


@dataclass
class ModelAnalysis:
    """Model analysis report"""
    model_id: str
    architecture: str
    total_params: int
    trainable_params: int
    memory_footprint_gb: float
    layer_count: int
    hidden_size: int
    attention_heads: int
    vocab_size: int
    sequence_length: int
    special_features: List[str]
    estimated_file_size: Dict[str, float]  # GB

class ModelAnalyzer:
    """Analyze model architecture and estimate sizes"""

    def __init__(
        self,
        model_id: str,
        trust_remote_code: bool = False,
        allow_network: bool = True,
        local_files_only: bool = False,
    ):
        self.model_id = model_id
        self.trust_remote_code = trust_remote_code
        self.allow_network = bool(allow_network)
        self.local_files_only = bool(local_files_only)
        self.config = None

    def analyze(self) -> ModelAnalysis:
        """Execute full analysis"""
        self.config = load_config_or_raw(
            self.model_id,
            security={
                "trust_remote_code": self.trust_remote_code,
                "allow_network": self.allow_network,
                "local_files_only": self.local_files_only,
            },
        )

        params = self._estimate_params()
        memory_gb = self._estimate_memory()

        return ModelAnalysis(
            model_id=self.model_id,
            architecture=getattr(self.config, 'model_type', 'unknown'),
            total_params=params,
            trainable_params=params, # Assuming all trainable for simplicity
            memory_footprint_gb=memory_gb,
            layer_count=getattr(self.config, 'num_hidden_layers', getattr(self.config, 'n_layer', 0)),
            hidden_size=getattr(self.config, 'hidden_size', getattr(self.config, 'n_embd', 0)),
            attention_heads=getattr(self.config, 'num_attention_heads', getattr(self.config, 'n_head', 0)),
            vocab_size=getattr(self.config, 'vocab_size', 0),
            sequence_length=getattr(self.config, 'max_position_embeddings', getattr(self.config, 'n_positions', 0)),
            special_features=self._detect_features(),
            estimated_file_size=self._estimate_file_sizes(memory_gb)
        )

    def _estimate_params(self) -> int:
        """Estimate parameter count based on config"""
        # This is a heuristic estimation.
        # Ideally we would init a meta model, but that might be slow/heavy.
        # We can try to use `num_parameters()` if available on empty model?
        try:
            # Try to init on meta device to get exact count
            from accelerate import init_empty_weights
            with init_empty_weights():
                model = hf_load_model_from_config(
                    self.config,
                    security={
                        "trust_remote_code": self.trust_remote_code,
                        "allow_network": self.allow_network,
                        "local_files_only": self.local_files_only,
                    },
                )
            return model.num_parameters()
        except Exception as exc:
            logger.debug("Model parameter count failed, falling back to heuristic: %s", exc)
            hidden_size = getattr(self.config, 'hidden_size', 4096)
            num_layers = getattr(self.config, 'num_hidden_layers', 32)
            vocab_size = getattr(self.config, 'vocab_size', 32000)

            # Simple Transformer estimation
            # Embedding: V * H
            # Layers: L * (4 * H^2 + 2 * H * 4H) approx (Attn + FFN) -> 12 * H^2
            # This is very rough.
            params = vocab_size * hidden_size
            params += num_layers * 12 * (hidden_size ** 2)
            return params

    def _estimate_memory(self) -> float:
        """Estimate memory footprint in GB (assuming bfloat16/float16)"""
        params = self._estimate_params()
        return (params * 2) / (1024**3)

    def _detect_features(self) -> List[str]:
        features = []
        if hasattr(self.config, 'rope_scaling') and self.config.rope_scaling:
            features.append("RoPE")
        if hasattr(self.config, 'sliding_window') and self.config.sliding_window:
            features.append("Sliding Window Attention")
        if 'moe' in getattr(self.config, 'model_type', '').lower():
            features.append("MoE")
        if hasattr(self.config, 'num_key_value_heads') and self.config.num_key_value_heads != getattr(self.config, 'num_attention_heads', -1):
            features.append("GQA")
        return features

    def _estimate_file_sizes(self, base_size_gb: float) -> Dict[str, float]:
        return {
            'random': base_size_gb,
            'sparse': base_size_gb * 0.001,  # Metadata only
            'compact': base_size_gb * 0.2,  # Rough guess for shared weights
            'ultra': base_size_gb * 0.0001  # Minimal
        }
