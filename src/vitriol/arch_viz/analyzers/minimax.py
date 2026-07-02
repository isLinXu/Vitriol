"""MiniMax family analyzer (M2, M3, etc.)."""

from typing import Any

from ..core import Architecture, Layer
from ._helpers import _finalize_architecture
from .base import TransformerAnalyzer


class MiniMaxAnalyzer(TransformerAnalyzer):
    """MiniMax family model analyzer (supports M2, M3, etc.)."""

    def analyze(self, config: Any) -> Architecture:
        model_type = getattr(config, "model_type", "")
        is_multimodal = "vl" in model_type or hasattr(config, "text_config") or hasattr(config, "vision_config")

        layers = []
        total_params = 0
        features = []

        if is_multimodal:
            # Handle multimodal model (MiniMax-M3)
            text_config = getattr(config, "text_config", config)
            vision_config = getattr(config, "vision_config", None)

            # Add Vision Encoder if present
            if vision_config is not None:
                layers.append(Layer("Vision Encoder", "encoder", 0, (), "Vision Encoder"))
                vision_hidden_size = getattr(vision_config, "hidden_size", 0)
                vision_layers = getattr(vision_config, "num_hidden_layers", 0)

                # Add projector hidden size for vision projector
                proj_hidden_size = getattr(config, "projector_hidden_size", 0)

                features = ["MiniMax-M3", "Multimodal", "MoE"]

                # Estimate vision encoder params (simplified)
                # Vision encoder layers
                for i in range(vision_layers):
                    layers.append(Layer(f"Vision Block {i}", "block_start", 0, (), ""))

                    # Vision self-attention
                    v_ln1 = vision_hidden_size
                    layers.append(Layer(f"Vision Block {i} - LN1", "normalization", v_ln1, (vision_hidden_size,)))
                    total_params += v_ln1

                    v_attn = 4 * vision_hidden_size * vision_hidden_size
                    layers.append(Layer(f"Vision Block {i} - Attention", "attention", v_attn, (vision_hidden_size, vision_hidden_size), "Vision Attention"))
                    total_params += v_attn

                    v_ln2 = vision_hidden_size
                    layers.append(Layer(f"Vision Block {i} - LN2", "normalization", v_ln2, (vision_hidden_size,)))
                    total_params += v_ln2

                    # Vision FFN
                    v_inter = getattr(vision_config, "intermediate_size", 0)
                    v_ffn = 3 * vision_hidden_size * v_inter
                    layers.append(Layer(f"Vision Block {i} - FFN", "feedforward", v_ffn, (vision_hidden_size, v_inter), "Vision FFN"))
                    total_params += v_ffn

                    layers.append(Layer(f"Vision Block {i}", "block_end", 0, (), ""))

                # Add multimodal projector
                if proj_hidden_size > 0 and vision_hidden_size > 0:
                    proj_params = vision_hidden_size * proj_hidden_size + proj_hidden_size * proj_hidden_size
                    layers.append(Layer("Multimodal Projector", "adapter", proj_params, (vision_hidden_size, proj_hidden_size), "Vision-to-Text Projection"))
                    total_params += proj_params

            # Use text config for LLM part
            config = text_config

        else:
            features = ["MiniMax-M2.5", "MoE"]

        # Now analyze the LLM part
        vocab_size = getattr(config, 'vocab_size', 0)
        hidden_size = getattr(config, 'hidden_size', 0)
        num_layers = getattr(config, 'num_hidden_layers', 0)
        num_heads = getattr(config, 'num_attention_heads', 0)
        num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)

        # MoE Params
        n_local_experts = getattr(config, 'num_local_experts', 0)
        active_experts = getattr(config, 'num_experts_per_tok', 0)

        # Check for sparse attention or other features
        num_mtp = getattr(config, 'num_mtp_modules', 0)
        if num_mtp > 0:
            features.append(f"MTP (Modules: {num_mtp})")

        # Dense intermediate size (for non-MoE layers)
        dense_inter = getattr(config, 'dense_intermediate_size', 0)
        shared_inter = getattr(config, 'shared_intermediate_size', 0)
        n_shared_experts = getattr(config, 'n_shared_experts', 0)

        if n_shared_experts > 0:
            features.append(f"Shared MoE with {n_shared_experts} shared experts")

        if getattr(config, 'sparse_attention_config', None) is not None:
            features.append("Sparse Attention")

        # 1. Embedding
        emb_params = vocab_size * hidden_size
        layers.append(Layer("Token Embedding", "embedding", emb_params, (vocab_size, hidden_size), f"Vocab: {vocab_size}"))
        total_params += emb_params

        # 2. Layers
        # Check MoE layer frequency
        moe_freq = getattr(config, 'moe_layer_freq', None)

        for i in range(num_layers):
            is_moe_layer = False
            if moe_freq is not None and i < len(moe_freq):
                is_moe_layer = moe_freq[i] > 0

            layers.append(Layer(f"Block {i}", "block_start", 0, (), ""))

            # Norm 1
            ln_params = hidden_size
            layers.append(Layer(f"Block {i} - Norm 1", "normalization", ln_params, (hidden_size,)))
            total_params += ln_params

            # Attention
            head_dim = getattr(config, 'head_dim', hidden_size // num_heads)
            q_params = hidden_size * (num_heads * head_dim)
            k_params = hidden_size * (num_kv_heads * head_dim)
            v_params = hidden_size * (num_kv_heads * head_dim)
            o_params = (num_heads * head_dim) * hidden_size

            attn_params = q_params + k_params + v_params + o_params
            layers.append(Layer(f"Block {i} - Attention", "attention", attn_params, (hidden_size, hidden_size), "GQA"))
            total_params += attn_params

            # Norm 2
            layers.append(Layer(f"Block {i} - Norm 2", "normalization", ln_params, (hidden_size,)))
            total_params += ln_params

            if is_moe_layer:
                # MoE FFN
                gate_params = hidden_size * n_local_experts
                # Shared expert params
                shared_expert_params = 0
                if n_shared_experts > 0 and shared_inter > 0:
                    shared_expert_params = n_shared_experts * 3 * hidden_size * shared_inter
                # Routed expert params
                intermediate_size = getattr(config, 'intermediate_size', 0)
                routed_expert_params = n_local_experts * 3 * hidden_size * intermediate_size

                ffn_params = gate_params + shared_expert_params + routed_expert_params
                desc = f"MoE (Experts: {n_local_experts}, Active: {active_experts})"
            else:
                # Dense FFN
                if dense_inter > 0:
                    ffn_params = 3 * hidden_size * dense_inter
                    desc = "Dense FFN"
                else:
                    intermediate_size = getattr(config, 'intermediate_size', 0)
                    ffn_params = 3 * hidden_size * intermediate_size
                    desc = "Dense FFN"

            layers.append(Layer(f"Block {i} - FFN", "feedforward", ffn_params, (hidden_size, intermediate_size or dense_inter), desc))
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

        # 5. MTP Modules (Multi-Token Prediction)
        if num_mtp > 0:
            for m in range(num_mtp):
                 layers.append(Layer(f"MTP Module {m}", "adapter", 0, (), f"Speculative Head {m}"))

        return _finalize_architecture(
            Architecture(
                model_type=model_type,
                arch_type="decoder-only" if not is_multimodal else "multimodal",
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
