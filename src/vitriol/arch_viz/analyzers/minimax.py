"""MiniMax-M2 family analyzer."""

from typing import Any

from ..core import Architecture, Layer
from ._helpers import _finalize_architecture
from .base import TransformerAnalyzer


class MiniMaxAnalyzer(TransformerAnalyzer):
    """MiniMax-M2.5 Analyzer."""

    def analyze(self, config: Any) -> Architecture:
        # MiniMax M2.5 has MTP modules (Multi-Token Prediction?) or similar auxiliary heads
        # config.use_mtp = True, config.num_mtp_modules = 3
        # It is an MoE model

        vocab_size = getattr(config, 'vocab_size', 0)
        hidden_size = getattr(config, 'hidden_size', 0)
        num_layers = getattr(config, 'num_hidden_layers', 0)
        num_heads = getattr(config, 'num_attention_heads', 0)
        num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)

        # MoE Params
        n_local_experts = getattr(config, 'num_local_experts', 0)
        active_experts = getattr(config, 'num_experts_per_tok', 0)

        features = ["MiniMax-M2.5", "MoE"]
        if getattr(config, 'use_mtp', False):
            features.append(f"MTP (Modules: {getattr(config, 'num_mtp_modules', 0)})")
        if getattr(config, 'attn_type_list', None):
            features.append("Hybrid Attention")

        layers = []
        total_params = 0

        # 1. Embedding
        emb_params = vocab_size * hidden_size
        layers.append(Layer("Token Embedding", "embedding", emb_params, (vocab_size, hidden_size), f"Vocab: {vocab_size}"))
        total_params += emb_params

        # 2. Layers
        # MiniMax uses attn_type_list to switch between different attention types?
        # Based on config: "attn_type_list": [1, 1, ..., 1] (all 1s)

        for i in range(num_layers):
            layers.append(Layer(f"Block {i}", "block_start", 0, (), ""))

            # Norm 1
            ln_params = hidden_size
            layers.append(Layer(f"Block {i} - Norm 1", "normalization", ln_params, (hidden_size,)))
            total_params += ln_params

            # Attention
            head_dim = getattr(config, 'head_dim', hidden_size // num_heads)
            # Standard GQA
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

            # MoE FFN
            # Gate
            gate_params = hidden_size * n_local_experts
            # Experts (SwiGLU)
            intermediate_size = getattr(config, 'intermediate_size', 0)
            expert_params = n_local_experts * (3 * hidden_size * intermediate_size)

            ffn_params = gate_params + expert_params
            desc = f"MoE (Experts: {n_local_experts}, Active: {active_experts})"

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

        # 5. MTP Modules (Multi-Token Prediction)
        # Assuming MTP modules are extra small decoder blocks or heads
        if getattr(config, 'use_mtp', False):
            num_mtp = getattr(config, 'num_mtp_modules', 0)

            # MTP usually shares embedding/head but has own transformer layers?
            # Or just extra heads?
            # MiniMax M2 paper/docs might specify. Usually it's like speculative decoding heads.
            # Let's approximate as extra lightweight blocks.

            # For visualization, we list them as special layers
            for m in range(num_mtp):
                 # MTP Block
                 # Assume similar structure but maybe smaller or fewer layers
                 # We'll just add a placeholder block for now
                 layers.append(Layer(f"MTP Module {m}", "adapter", 0, (), f"Speculative Head {m}"))

        return _finalize_architecture(
            Architecture(
                model_type=getattr(config, 'model_type', 'minimax_m2'),
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
