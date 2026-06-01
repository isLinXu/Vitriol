"""GLM family analyzer."""

from typing import Any

from ..core import Architecture, Layer
from ._helpers import _finalize_architecture
from .base import TransformerAnalyzer


class GLMAnalyzer(TransformerAnalyzer):
    """GLM-5 / glm_moe_dsa specific analyzer."""
    def analyze(self, config: Any) -> Architecture:
        # GLM-5 has hybrid layers (dense + sparse MoE) defined by mlp_layer_types
        # We need to iterate carefully

        vocab_size = getattr(config, 'vocab_size', 0)
        hidden_size = getattr(config, 'hidden_size', 0)
        num_layers = getattr(config, 'num_hidden_layers', 0)
        num_heads = getattr(config, 'num_attention_heads', 0)
        num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)

        # GLM specific fields
        mlp_layer_types = getattr(config, 'mlp_layer_types', [])
        moe_intermediate_size = getattr(config, 'moe_intermediate_size', 0)
        intermediate_size = getattr(config, 'intermediate_size', 0)
        num_experts = getattr(config, 'n_routed_experts', 0)
        active_experts = getattr(config, 'num_experts_per_tok', 0)
        shared_experts = getattr(config, 'n_shared_experts', 0)

        features = ["GLM-5", "DSA"]
        if "sparse" in mlp_layer_types:
            features.append("MoE")

        layers = []
        total_params = 0

        # 1. Embedding
        emb_params = vocab_size * hidden_size
        layers.append(Layer("Token Embedding", "embedding", emb_params, (vocab_size, hidden_size), f"Vocab: {vocab_size}"))
        total_params += emb_params

        # 2. Layers
        for i in range(num_layers):
            layers.append(Layer(f"Block {i}", "block_start", 0, (), ""))

            # Norm 1
            ln_params = hidden_size
            layers.append(Layer(f"Block {i} - Norm 1", "normalization", ln_params, (hidden_size,)))
            total_params += ln_params

            # Attention (DSA?)
            # Simplified calculation for standard attention, adjust for DSA/MLA if needed
            # GLM-5 uses standard GQA + RoPE usually, DSA is sparsity pattern
            head_dim = hidden_size // num_heads
            # Q, K, V, O
            # Check for qk_nope_head_dim etc if needed for precise count
            # Standard GQA:
            q_params = hidden_size * hidden_size
            k_params = hidden_size * (num_kv_heads * head_dim)
            v_params = hidden_size * (num_kv_heads * head_dim)
            o_params = hidden_size * hidden_size
            attn_params = q_params + k_params + v_params + o_params

            layers.append(Layer(f"Block {i} - Attention", "attention", attn_params, (hidden_size, hidden_size), "DSA"))
            total_params += attn_params

            # Norm 2
            layers.append(Layer(f"Block {i} - Norm 2", "normalization", ln_params, (hidden_size,)))
            total_params += ln_params

            # FFN / MoE
            layer_type = mlp_layer_types[i] if i < len(mlp_layer_types) else "dense"

            if layer_type == "sparse":
                # MoE Layer
                # Gate
                gate_params = hidden_size * num_experts
                # Experts: SwiGLU 3 matrices
                # Routed Experts
                routed_params = num_experts * (3 * hidden_size * moe_intermediate_size)
                # Shared Experts
                shared_params = shared_experts * (3 * hidden_size * moe_intermediate_size)

                ffn_params = gate_params + routed_params + shared_params
                desc = f"MoE (Routed: {num_experts}, Active: {active_experts}, Shared: {shared_experts})"
            else:
                # Dense Layer
                # SwiGLU
                ffn_params = 3 * hidden_size * intermediate_size
                desc = f"Dense (Inter: {intermediate_size})"

            layers.append(Layer(f"Block {i} - FFN", "feedforward", ffn_params, (hidden_size, intermediate_size), desc))
            total_params += ffn_params

            layers.append(Layer(f"Block {i}", "block_end", 0, (), ""))

        # 3. Final Norm
        layers.append(Layer("Final Norm", "normalization", hidden_size, (hidden_size,)))
        total_params += hidden_size

        # 4. Head
        if getattr(config, 'tie_word_embeddings', False):
            head_params = 0
            desc = "Tied"
        else:
            head_params = vocab_size * hidden_size
            desc = "Untied"
        layers.append(Layer("LM Head", "output", head_params, (hidden_size, vocab_size), desc))
        total_params += head_params

        return _finalize_architecture(
            Architecture(
                model_type=getattr(config, 'model_type', 'unknown'),
                arch_type="decoder-only",
                total_layers=num_layers,
                total_params=total_params,
                memory_fp16_gb=total_params * 2 / (1024**3),
                parameters={
                    "vocab_size": vocab_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "num_heads": num_heads
                },
                features=features,
                layers=layers,
            ),
            total_layers=num_layers,
        )
