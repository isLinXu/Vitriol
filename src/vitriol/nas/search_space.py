import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict, fields

@dataclass
class ArchitectureGene:
    """Represents a specific architecture configuration encoded as a gene."""
    # Macro Architecture
    n_layers: int
    hidden_size: int
    n_heads: int
    
    # Micro Architecture
    attention_type: str  # "MHA", "GQA", "MQA"
    ffn_type: str        # "Standard", "SwiGLU", "GeGLU"
    activation: str      # "gelu", "silu", "relu"
    norm_type: str       # "LayerNorm", "RMSNorm"
    
    vocab_size: int = 32000  # Default to 32k, but can be larger
    
    # Derived/Optional
    intermediate_size: int = field(init=False)
    num_kv_heads: int = field(init=False)

    def __post_init__(self):
        # Constraints and derived values
        if self.hidden_size % self.n_heads != 0:
            # Adjust hidden size to be divisible by n_heads
            self.hidden_size = (self.hidden_size // self.n_heads) * self.n_heads
            
        # Default FFN ratio (typically 4x or 8/3x for SwiGLU)
        mult = 8/3 if self.ffn_type in ["SwiGLU", "GeGLU"] else 4.0
        self.intermediate_size = int(self.hidden_size * mult)
        
        # KV Heads logic
        if self.attention_type == "MHA":
            self.num_kv_heads = self.n_heads
        elif self.attention_type == "MQA":
            self.num_kv_heads = 1
        elif self.attention_type == "GQA":
            # Default to 1/4 or 1/8 heads
            self.num_kv_heads = max(1, self.n_heads // 4)
        else:
            self.num_kv_heads = self.n_heads

    def to_dict(self) -> Dict[str, Any]:
        """Convert gene to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ArchitectureGene':
        """Create gene from dictionary."""
        # Filter out derived fields that are not in __init__
        valid_keys = {f.name for f in fields(cls) if f.init}
        init_kwargs = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**init_kwargs)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ArchitectureGene':
        """Create a gene from a Hugging Face-style config dictionary."""
        source = dict(config.get("_vitriol_nas_gene") or {})

        hidden_size = int(config.get("hidden_size", source.get("hidden_size", 1024)))
        n_heads = int(config.get("num_attention_heads", source.get("n_heads", 8)))
        num_kv_heads = int(config.get("num_key_value_heads", source.get("num_kv_heads", n_heads)))

        if num_kv_heads == n_heads:
            attention_type = "MHA"
        elif num_kv_heads == 1:
            attention_type = "MQA"
        else:
            attention_type = "GQA"

        intermediate_size = int(config.get("intermediate_size", source.get("intermediate_size", hidden_size * 4)))
        swiglu_size = int(hidden_size * 8 / 3)
        ffn_type = source.get("ffn_type")
        if ffn_type not in {"Standard", "SwiGLU", "GeGLU"}:
            ffn_type = "SwiGLU" if abs(intermediate_size - swiglu_size) < abs(intermediate_size - hidden_size * 4) else "Standard"

        return cls(
            n_layers=int(config.get("num_hidden_layers", source.get("n_layers", 12))),
            hidden_size=hidden_size,
            n_heads=n_heads,
            attention_type=source.get("attention_type", attention_type),
            ffn_type=ffn_type,
            activation=str(config.get("hidden_act", source.get("activation", "gelu"))),
            norm_type=source.get("norm_type", "RMSNorm" if "rms_norm_eps" in config else "LayerNorm"),
            vocab_size=int(config.get("vocab_size", source.get("vocab_size", 32000))),
        )

    def to_config(self) -> Dict[str, Any]:
        """Convert gene to Hugging Face config dictionary."""
        return {
            "model_type": "qwen2",  # Default to qwen2 structure for broad compatibility
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.n_layers,
            "num_attention_heads": self.n_heads,
            "num_key_value_heads": self.num_kv_heads,
            "hidden_act": self.activation,
            "rms_norm_eps": 1e-6,
            "attention_dropout": 0.0,
            "hidden_dropout_prob": 0.0,
            "use_bias": True,
            "rope_theta": 10000.0,
            "use_cache": True,
            # Vitriol specific markers if needed
            "_vitriol_nas_gene": asdict(self)
        }

class SearchSpace:
    """Base class for search spaces."""
    def sample(self) -> ArchitectureGene:
        raise NotImplementedError

class LLMSearchSpace(SearchSpace):
    """Defines the search space for LLM architectures."""
    
    def __init__(self, vocab_sizes: Optional[List[int]] = None):
        # Macro Dimensions
        self.n_layers_range = list(range(6, 33, 2))  # 6 to 32, step 2
        self.hidden_size_choices = [512, 768, 1024, 1536, 2048, 4096]
        self.n_heads_choices = [4, 8, 12, 16, 24, 32]
        self.vocab_size_choices = vocab_sizes if vocab_sizes else [32000, 151936] # Standard + Qwen
        
        # Micro Dimensions
        self.attention_types = ["MHA", "GQA", "MQA"]
        self.ffn_types = ["Standard", "SwiGLU"]
        self.activations = ["gelu", "silu"]
        self.norm_types = ["RMSNorm", "LayerNorm"]
        self.default_config = {
            "num_hidden_layers": self.n_layers_range,
            "hidden_size": self.hidden_size_choices,
            "num_attention_heads": self.n_heads_choices,
            "intermediate_size": sorted({
                int(hidden * 4.0)
                for hidden in self.hidden_size_choices
            } | {
                int(hidden * 8 / 3)
                for hidden in self.hidden_size_choices
            }),
        }

    def sample(self) -> ArchitectureGene:
        """Randomly sample an architecture from the search space."""
        # Sample hidden size first
        hidden = random.choice(self.hidden_size_choices)
        
        # Filter valid n_heads (hidden % heads == 0)
        valid_heads = [h for h in self.n_heads_choices if hidden % h == 0]
        if not valid_heads:
            # Fallback
            valid_heads = [4]
            hidden = (hidden // 4) * 4
            
        return ArchitectureGene(
            n_layers=random.choice(self.n_layers_range),
            hidden_size=hidden,
            n_heads=random.choice(valid_heads),
            vocab_size=random.choice(self.vocab_size_choices),
            attention_type=random.choice(self.attention_types),
            ffn_type=random.choice(self.ffn_types),
            activation=random.choice(self.activations),
            norm_type=random.choice(self.norm_types)
        )

    def sample_random(self) -> ArchitectureGene:
        """Backward-compatible alias used by the RL searcher."""
        return self.sample()

    def validate_gene(self, gene: ArchitectureGene) -> bool:
        """Validate that a gene stays inside this search space."""
        if gene.n_layers not in self.n_layers_range:
            return False
        if gene.hidden_size not in self.hidden_size_choices:
            return False
        if gene.n_heads not in self.n_heads_choices:
            return False
        if gene.hidden_size % gene.n_heads != 0:
            return False
        if gene.attention_type not in self.attention_types:
            return False
        if gene.ffn_type not in self.ffn_types:
            return False
        if gene.activation not in self.activations:
            return False
        if gene.norm_type not in self.norm_types:
            return False
        return gene.vocab_size in self.vocab_size_choices

    def mutate(self, gene: ArchitectureGene, mutation_rate: float = 0.1) -> ArchitectureGene:
        """Mutate a gene with given probability."""
        new_gene_dict = asdict(gene)
        
        if random.random() < mutation_rate:
            new_gene_dict['n_layers'] = random.choice(self.n_layers_range)
            
        if random.random() < mutation_rate:
            hidden = random.choice(self.hidden_size_choices)
            new_gene_dict['hidden_size'] = hidden
            # Re-validate heads
            valid_heads = [h for h in self.n_heads_choices if hidden % h == 0]
            if new_gene_dict['n_heads'] not in valid_heads:
                new_gene_dict['n_heads'] = random.choice(valid_heads)
                
        if random.random() < mutation_rate:
            # Mutate micro architecture
            new_gene_dict['attention_type'] = random.choice(self.attention_types)

        if random.random() < mutation_rate:
            new_gene_dict['vocab_size'] = random.choice(self.vocab_size_choices)
        
        if random.random() < mutation_rate:
            new_gene_dict['ffn_type'] = random.choice(self.ffn_types)

        return ArchitectureGene(**{k: v for k, v in new_gene_dict.items() if k not in ['intermediate_size', 'num_kv_heads']})
