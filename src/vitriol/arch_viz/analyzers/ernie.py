"""ERNIE 4.5 VL family analyzer."""

from typing import Any

from ..core import Architecture, Layer
from ._helpers import _finalize_architecture
from .base import TransformerAnalyzer


class ErnieAnalyzer(TransformerAnalyzer):
    """ERNIE 4.5 VL (MoE + Vision) Analyzer."""

    def analyze(self, config: Any) -> Architecture:
        # Check for vision config
        vision_config = getattr(config, 'vision_config', None)

        # Base Transformer Analysis (Language Part)
        vocab_size = getattr(config, 'vocab_size', 0)
        hidden_size = getattr(config, 'hidden_size', 0)
        num_layers = getattr(config, 'num_hidden_layers', 0)
        num_heads = getattr(config, 'num_attention_heads', 0)

        # MoE Config
        moe_num_experts = getattr(config, 'moe_num_experts', [0])
        # config.moe_num_experts is list [64, 64] ? Or maybe different per layer?
        # Based on config: "moe_num_experts": [64, 64]
        # "moe_num_shared_experts": 2
        n_routed = moe_num_experts[0] if isinstance(moe_num_experts, list) else moe_num_experts
        n_shared = getattr(config, 'moe_num_shared_experts', 0)
        active_experts = getattr(config, 'moe_k', 0)

        moe_intermediate_list = getattr(config, 'moe_intermediate_size', [0, 0])
        # [1536, 512] ?
        # Usually [routed_inter, shared_inter] or similar
        routed_inter = moe_intermediate_list[0] if isinstance(moe_intermediate_list, list) else moe_intermediate_list
        shared_inter = moe_intermediate_list[1] if isinstance(moe_intermediate_list, list) and len(moe_intermediate_list) > 1 else routed_inter

        features = ["ERNIE-4.5", "MoE", "Vision-Language"]
        if getattr(config, 'rope_3d', False):
            features.append("3D-RoPE")

        layers = []
        total_params = 0

        # --- Vision Encoder ---
        if vision_config:
            v_hidden = getattr(vision_config, 'hidden_size', 0)
            v_layers = getattr(vision_config, 'depth', getattr(vision_config, 'num_hidden_layers', 0))
            v_patch = getattr(vision_config, 'patch_size', 14)

            # Simple Vision Encoder Estimate
            # ViT-like: PatchEmbed + Layers * (Attn + MLP)
            v_params = 0
            # Patch Embed: (3, hidden, patch, patch)
            v_params += 3 * v_hidden * v_patch * v_patch

            # Transformer Layers
            # Attn: 4 * hidden^2 (Q,K,V,O)
            # MLP: 2 * hidden * (4*hidden) = 8 * hidden^2
            # Norms: 2 * hidden
            layer_p = 12 * v_hidden * v_hidden
            v_params += v_layers * layer_p

            layers.append(Layer("Vision Encoder", "encoder", v_params, (v_layers, v_hidden), f"ViT-L/{v_layers}"))
            total_params += v_params

            # Projector (if any)
            mm_hidden = getattr(vision_config, 'mm_hidden_size', v_hidden)
            proj_params = v_hidden * mm_hidden # Linear
            layers.append(Layer("Vision Projector", "adapter", proj_params, (v_hidden, mm_hidden), "Linear"))
            total_params += proj_params

        # --- Language Model ---

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

            # Attention
            attn_params = 4 * hidden_size * hidden_size # Standard Approx
            layers.append(Layer(f"Block {i} - Attention", "attention", attn_params, (hidden_size, hidden_size), "FlashAttn"))
            total_params += attn_params

            # Norm 2
            layers.append(Layer(f"Block {i} - Norm 2", "normalization", ln_params, (hidden_size,)))
            total_params += ln_params

            # MoE FFN
            # Gate
            gate_params = hidden_size * n_routed
            # Routed Experts (SwiGLU 3 matrices)
            routed_params = n_routed * (3 * hidden_size * routed_inter)
            # Shared Experts (SwiGLU 3 matrices)
            shared_params = n_shared * (3 * hidden_size * shared_inter)

            ffn_params = gate_params + routed_params + shared_params
            desc = f"MoE (Routed: {n_routed}, Active: {active_experts}, Shared: {n_shared})"

            layers.append(Layer(f"Block {i} - FFN", "feedforward", ffn_params, (hidden_size, routed_inter), desc))
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
                arch_type="encoder-decoder", # VL is effectively hybrid
                total_layers=num_layers,
                total_params=total_params,
                memory_fp16_gb=total_params * 2 / (1024**3),
                parameters={
                    "vocab_size": vocab_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "vision_layers": int(getattr(vision_config, "num_hidden_layers", getattr(vision_config, "depth", 0)) or 0),
                },
                features=features,
                layers=layers,
            ),
            total_layers=num_layers,
        )
