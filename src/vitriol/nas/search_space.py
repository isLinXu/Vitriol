import random
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

from ..types import HFConfigDict


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

    # Derived/Optional fields (computed in __post_init__)
    intermediate_size: int = field(init=False)
    num_kv_heads: int = field(init=False)

    # MLA (Multi-head Latent Attention) parameters
    use_mla: bool = False
    qk_nope_head_dim: int = field(init=False)
    qk_rope_head_dim: int = field(init=False)
    kv_lora_rank: int = field(init=False)
    q_lora_rank: int = field(init=False)

    # MoE (Mixture of Experts) parameters
    use_moe: bool = False
    num_experts: int = field(init=False)
    num_experts_per_tok: int = field(init=False)
    moe_intermediate_size: int = field(init=False)
    shared_expert_intermediate_size: int = field(init=False)

    # Mamba / State Space Model parameters
    use_mamba: bool = False
    d_state: int = field(init=False)
    d_conv: int = field(init=False)
    expand_factor: int = field(init=False)

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

        # MLA derived values
        head_dim = self.hidden_size // self.n_heads
        if self.use_mla:
            self.qk_nope_head_dim = max(head_dim // 2, 1)
            self.qk_rope_head_dim = max(head_dim // 4, 4)
            self.kv_lora_rank = max(head_dim * 2, 64)
            self.q_lora_rank = max(head_dim * 2, 64)
        else:
            self.qk_nope_head_dim = head_dim
            self.qk_rope_head_dim = max(head_dim // 4, 4)
            self.kv_lora_rank = 0
            self.q_lora_rank = 0

        # MoE derived values
        if self.use_moe:
            self.num_experts = max(8, self.n_layers * 2)
            self.num_experts_per_tok = max(2, self.num_experts // 4)
            self.moe_intermediate_size = max(self.intermediate_size // 2, 256)
            self.shared_expert_intermediate_size = max(self.intermediate_size // 4, 128)
        else:
            self.num_experts = 0
            self.num_experts_per_tok = 0
            self.moe_intermediate_size = 0
            self.shared_expert_intermediate_size = 0

        # Mamba derived values
        if self.use_mamba:
            self.d_state = max(16, self.hidden_size // 64)
            self.d_conv = 4
            self.expand_factor = 2
        else:
            self.d_state = 0
            self.d_conv = 0
            self.expand_factor = 0

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
    def from_config(cls, config: HFConfigDict) -> 'ArchitectureGene':
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

        # Detect MLA
        use_mla = bool(
            source.get("use_mla")
            or config.get("qk_nope_head_dim")
            or config.get("kv_lora_rank")
        )

        # Detect MoE
        num_experts = int(config.get("num_experts", config.get("n_routed_experts", 0)) or 0)
        use_moe = bool(source.get("use_moe") or num_experts > 0)

        # Detect Mamba
        use_mamba = bool(
            source.get("use_mamba")
            or config.get("d_state")
            or config.get("model_type", "").startswith("mamba")
        )

        return cls(
            n_layers=int(config.get("num_hidden_layers", source.get("n_layers", 12))),
            hidden_size=hidden_size,
            n_heads=n_heads,
            attention_type=source.get("attention_type", attention_type),
            ffn_type=ffn_type,
            activation=str(config.get("hidden_act", source.get("activation", "gelu"))),
            norm_type=source.get("norm_type", "RMSNorm" if "rms_norm_eps" in config else "LayerNorm"),
            vocab_size=int(config.get("vocab_size", source.get("vocab_size", 32000))),
            use_mla=use_mla,
            use_moe=use_moe,
            use_mamba=use_mamba,
        )

    def to_config(self) -> HFConfigDict:
        """Convert gene to Hugging Face config dictionary."""
        cfg: Dict[str, Any] = {
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
        if self.use_mla:
            cfg["qk_nope_head_dim"] = self.qk_nope_head_dim
            cfg["qk_rope_head_dim"] = self.qk_rope_head_dim
            cfg["kv_lora_rank"] = self.kv_lora_rank
            cfg["q_lora_rank"] = self.q_lora_rank
        if self.use_moe:
            cfg["num_experts"] = self.num_experts
            cfg["num_experts_per_tok"] = self.num_experts_per_tok
            cfg["moe_intermediate_size"] = self.moe_intermediate_size
            cfg["shared_expert_intermediate_size"] = self.shared_expert_intermediate_size
        if self.use_mamba:
            cfg["d_state"] = self.d_state
            cfg["d_conv"] = self.d_conv
            cfg["expand_factor"] = self.expand_factor
        return cfg

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

        # Sample advanced architecture variants
        use_mla = random.random() < 0.15  # 15% chance for MLA
        use_moe = random.random() < 0.20  # 20% chance for MoE
        use_mamba = random.random() < 0.10  # 10% chance for Mamba

        return ArchitectureGene(
            n_layers=random.choice(self.n_layers_range),
            hidden_size=hidden,
            n_heads=random.choice(valid_heads),
            vocab_size=random.choice(self.vocab_size_choices),
            attention_type=random.choice(self.attention_types),
            ffn_type=random.choice(self.ffn_types),
            activation=random.choice(self.activations),
            norm_type=random.choice(self.norm_types),
            use_mla=use_mla,
            use_moe=use_moe,
            use_mamba=use_mamba,
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

        # Mutate advanced flags
        if random.random() < mutation_rate:
            new_gene_dict['use_mla'] = random.random() < 0.15
        if random.random() < mutation_rate:
            new_gene_dict['use_moe'] = random.random() < 0.20
        if random.random() < mutation_rate:
            new_gene_dict['use_mamba'] = random.random() < 0.10

        # Exclude derived fields that are not in __init__
        derived = {'intermediate_size', 'num_kv_heads', 'qk_nope_head_dim',
                   'qk_rope_head_dim', 'kv_lora_rank', 'q_lora_rank',
                   'num_experts', 'num_experts_per_tok', 'moe_intermediate_size',
                   'shared_expert_intermediate_size', 'd_state', 'd_conv', 'expand_factor'}
        return ArchitectureGene(**{k: v for k, v in new_gene_dict.items() if k not in derived})
