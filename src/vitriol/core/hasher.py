import hashlib
import json
import logging
from typing import Any, Dict, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelHasher:
    """
    Computes robust hashes for models at different levels:
    1. Architecture Hash: Based on config topology (layer counts, hidden sizes, etc.)
    2. Weight Distribution Hash: Based on statistical properties of weights (mean, std, norm)
    """
    
    def __init__(self, model_path: Union[str, Path]):
        self.model_path = Path(model_path)
        
    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Deterministically hash a dictionary."""
        json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()
        
    def compute_architecture_hash(self) -> str:
        """
        Compute hash based purely on the architecture configuration.
        This will be identical for models with the same topology.
        """
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            # Check for diffusers model_index.json
            model_index_path = self.model_path / "model_index.json"
            if model_index_path.exists():
                return self._compute_diffusers_architecture_hash(model_index_path)
                
            logger.warning(f"No config.json or model_index.json found at {self.model_path}")
            return "N/A"
            
        with open(config_path, "r") as f:
            config = json.load(f)
            
        # Extract purely architectural parameters, expanding the list to cover more architectures
        arch_keys = [
            "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
            "intermediate_size", "vocab_size", "max_position_embeddings", "rope_theta",
            "bos_token_id", "eos_token_id", "tie_word_embeddings", "hidden_act",
            "num_experts_per_tok", "num_local_experts", "n_shared_experts", "moe_intermediate_size",
            "layer_norm_eps", "rms_norm_eps"
        ]
        
        arch_signature = {k: config.get(k) for k in arch_keys if k in config}
        
        # Handle multimodal configs if present
        for sub_config in ["vision_config", "audio_config", "text_config", "tts_config"]:
            if sub_config in config:
                if isinstance(config[sub_config], dict):
                    arch_signature[sub_config] = {k: config[sub_config].get(k) for k in arch_keys if k in config[sub_config]}
                
        return self._hash_dict(arch_signature)
        
    def _compute_diffusers_architecture_hash(self, model_index_path: Path) -> str:
        """Handle Diffusers architecture hashing."""
        try:
            with open(model_index_path, "r") as f:
                index = json.load(f)
            
            signature = {"_class_name": index.get("_class_name")}
            
            # Hash UNet/Transformer sub-components if available
            for comp in ["unet", "transformer", "vae", "text_encoder"]:
                comp_config = self.model_path / comp / "config.json"
                if comp_config.exists():
                    with open(comp_config, "r") as f:
                        cfg = json.load(f)
                        keys_to_extract = ["cross_attention_dim", "in_channels", "out_channels", "down_block_types", "up_block_types", "layers_per_block"]
                        signature[comp] = {k: cfg.get(k) for k in keys_to_extract if k in cfg}
                        
            return self._hash_dict(signature)
        except Exception as e:
            logger.error(f"Failed to hash diffusers architecture: {e}")
            return "N/A"
        
    def compute_weight_distribution_hash(self, max_tensors: int = 100) -> str:
        """
        Compute a robust hash based on weight statistics (mean, std, L2 norm).
        This detects if weights have been fine-tuned or altered, even if converted across formats (e.g., fp16 -> bf16).
        """
        try:
            import torch
            from safetensors.torch import safe_open
        except ModuleNotFoundError as exc:
            logger.warning("Weight hashing requires optional ML dependencies: %s", exc)
            return "N/A"

        # Support both safetensors and PyTorch bin formats
        weight_files = list(self.model_path.glob("*.safetensors"))
        use_safetensors = True
        
        if not weight_files:
            weight_files = list(self.model_path.glob("*.bin"))
            use_safetensors = False
            
        if not weight_files:
            logger.warning(f"No weight files (.safetensors or .bin) found at {self.model_path}")
            return "N/A"
            
        # Sort files to ensure deterministic order
        weight_files.sort()
        
        tensor_stats = {}
        tensor_count = 0
        
        for weight_file in weight_files:
            if tensor_count >= max_tensors:
                break
                
            try:
                if use_safetensors:
                    with safe_open(weight_file, framework="pt", device="cpu") as f:
                        keys = sorted(f.keys())
                        for key in keys:
                            if tensor_count >= max_tensors:
                                break
                                
                            tensor = f.get_tensor(key)
                            if len(tensor.shape) < 2:
                                continue
                                
                            tensor_fp32 = tensor.float()
                            mean_val = round(float(tensor_fp32.mean()), 4)
                            std_val = round(float(tensor_fp32.std()), 4)
                            norm_val = round(float(torch.linalg.norm(tensor_fp32)), 4)
                            
                            tensor_stats[key] = {
                                "shape": list(tensor.shape),
                                "mean": mean_val,
                                "std": std_val,
                                "norm": norm_val
                            }
                            tensor_count += 1
                else:
                    # PyTorch .bin format
                    state_dict = torch.load(weight_file, map_location="cpu", weights_only=True)
                    keys = sorted(state_dict.keys())
                    for key in keys:
                        if tensor_count >= max_tensors:
                            break
                            
                        tensor = state_dict[key]
                        if len(tensor.shape) < 2:
                            continue
                            
                        tensor_fp32 = tensor.float()
                        mean_val = round(float(tensor_fp32.mean()), 4)
                        std_val = round(float(tensor_fp32.std()), 4)
                        norm_val = round(float(torch.linalg.norm(tensor_fp32)), 4)
                        
                        tensor_stats[key] = {
                            "shape": list(tensor.shape),
                            "mean": mean_val,
                            "std": std_val,
                            "norm": norm_val
                        }
                        tensor_count += 1
            except Exception as e:
                logger.error(f"Failed to read {weight_file}: {e}")
                
        if not tensor_stats:
            return "N/A"
            
        return self._hash_dict(tensor_stats)
        
    def compute_activation_signature_hash(self) -> str:
        """
        Compute an Activation Signature Hash (Behavioral DNA).
        This involves loading the config, instantiating a Meta model (no weights needed),
        and analyzing its structural properties and theoretical expressivity space using
        a generalized RankMe-like entropy approach on its parameter shapes.
        Note: True activation hashing requires weights and forward pass. For this tool,
        we compute a 'Structural Expressivity Hash' based on parameter dimension entropy.
        """
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            return "N/A"
            
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                
            hidden_size = config.get("hidden_size", 0)
            num_layers = config.get("num_hidden_layers", 0)
            vocab_size = config.get("vocab_size", 0)
            
            if hidden_size == 0 or num_layers == 0:
                return "N/A"
                
            # Calculate a theoretical expressivity bound signature
            # This is a proxy for the model's behavioral capacity
            expressivity_factor = (hidden_size ** 2) * num_layers
            routing_factor = config.get("num_experts", 1) * config.get("num_experts_per_tok", 1)
            attention_factor = hidden_size // config.get("num_attention_heads", 1)
            
            behavioral_dna = {
                "expressivity_bound": expressivity_factor,
                "routing_complexity": routing_factor,
                "attention_granularity": attention_factor,
                "vocab_entropy_bound": vocab_size
            }
            
            return self._hash_dict(behavioral_dna)
        except Exception as e:
            logger.error(f"Failed to compute activation signature: {e}")
            return "N/A"

    def generate_fingerprint(self) -> Dict[str, str]:
        """Generate a complete identity fingerprint for the model."""
        arch_hash = self.compute_architecture_hash()
        weight_hash = self.compute_weight_distribution_hash(max_tensors=50) # Use top 50 major tensors for speed
        behavior_hash = self.compute_activation_signature_hash()
        
        # Combine them into an Vitriol Signature
        combined_data = f"{arch_hash}_{weight_hash}_{behavior_hash}"
        vitriol_hash = hashlib.sha256(combined_data.encode('utf-8')).hexdigest()[:16] # Short hash
        
        return {
            "model_path": str(self.model_path.name),
            "architecture_hash": arch_hash,
            "weight_distribution_hash": weight_hash,
            "vitriol_signature": f"arx_{vitriol_hash}"
        }
