
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Any
import json

@dataclass
class Layer:
    """Represents a single layer in the model architecture."""
    name: str
    type: str  # embedding, attention, feedforward, normalization, output, block_start, block_end
    params: int
    shape: Tuple[Any, ...]
    description: str = ""
    
    def __repr__(self) -> str:
        return f"Layer({self.name}, {self.params:,} params, {self.type})"

@dataclass
class Architecture:
    """Represents the complete model architecture."""
    model_type: str
    arch_type: str
    total_layers: int
    total_params: int
    memory_fp16_gb: float
    parameters: Dict[str, Any]  # vocab_size, hidden_size, etc.
    features: List[str]
    special_features: List[str] = field(default_factory=list) # Added for compatibility
    layers: List[Layer] = field(default_factory=list)
    encoder_layers: int = 0
    decoder_layers: int = 0

    def __post_init__(self) -> None:
        """Compatibility shim for feature fields.

        Legacy renderers/tests read special_features, while newer analyzers populate features.
        If special_features is empty, fall back to features to avoid losing information.
        """
        if not self.special_features and self.features:
            self.special_features = list(self.features)
        if self.encoder_layers and "encoder_layers" not in self.parameters:
            self.parameters["encoder_layers"] = self.encoder_layers
        if self.decoder_layers and "decoder_layers" not in self.parameters:
            self.parameters["decoder_layers"] = self.decoder_layers
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
        
    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
