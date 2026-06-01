"""Qwen3.5 MoE family analyzer."""

from typing import Any

from ..core import Architecture, Layer
from ._helpers import _finalize_architecture
from .base import TransformerAnalyzer


class Qwen35Analyzer(TransformerAnalyzer):
    """Qwen3.5 MoE (A3B) Analyzer."""

    def analyze(self, config: Any) -> Architecture:
        def _get(obj: Any, key: str, default: Any = 0) -> Any:
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        # Qwen3.5 MoE has "text_config" and "vision_config"
        text_config = getattr(config, 'text_config', config)
        vision_config = getattr(config, 'vision_config', None)

        # Base
        vocab_size = int(_get(text_config, 'vocab_size', 0) or 0)
        hidden_size = int(_get(text_config, 'hidden_size', 0) or 0)
        num_layers = int(_get(text_config, 'num_hidden_layers', 0) or 0)
        num_heads = int(_get(text_config, 'num_attention_heads', 0) or 0)
        if num_heads <= 0 and hidden_size:
            num_heads = 1
        num_kv_heads = int(_get(text_config, 'num_key_value_heads', num_heads) or 0) or num_heads

        # MoE
        n_experts = int(_get(text_config, 'num_experts', 0) or 0)
        active_experts = int(_get(text_config, 'num_experts_per_tok', 0) or 0)

        # Layer Types (Linear vs Full Attention)
        layer_types = _get(text_config, 'layer_types', []) or []

        features = ["Qwen3.5", "MoE"]
        if vision_config:
            features.append("Vision")
        if "linear_attention" in layer_types:
            features.append("Linear Attention (Mamba/SSM?)")

        layers = []
        total_params = 0

        # --- Vision ---
        if vision_config:
            v_hidden = getattr(vision_config, 'hidden_size', 0)
            v_layers = getattr(vision_config, 'depth', 0)
            v_params = 12 * v_layers * v_hidden * v_hidden # approx
            layers.append(Layer("Vision Encoder", "encoder", v_params, (v_layers, v_hidden), "Qwen-Vision"))
            total_params += v_params

        # --- Text ---
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

            # Attention (Linear or Full)
            l_type = layer_types[i] if i < len(layer_types) else "full_attention"

            head_dim = _get(text_config, 'head_dim', None)
            head_dim = int(head_dim or 0)
            if head_dim <= 0:
                head_dim = (hidden_size // num_heads) if num_heads else 0
            q_params = hidden_size * (num_heads * head_dim)
            k_params = hidden_size * (num_kv_heads * head_dim)
            v_params = hidden_size * (num_kv_heads * head_dim)
            o_params = (num_heads * head_dim) * hidden_size
            attn_params = q_params + k_params + v_params + o_params

            attn_desc = "Full Attn" if l_type == "full_attention" else "Linear Attn"
            layers.append(Layer(f"Block {i} - Attention", "attention", attn_params, (hidden_size, hidden_size), attn_desc))
            total_params += attn_params

            # Norm 2
            layers.append(Layer(f"Block {i} - Norm 2", "normalization", ln_params, (hidden_size,)))
            total_params += ln_params

            # MoE FFN
            moe_inter = int(_get(text_config, 'moe_intermediate_size', 0) or 0)
            shared_inter = int(_get(text_config, 'shared_expert_intermediate_size', 0) or 0)

            # Gate
            gate_params = hidden_size * n_experts
            # Routed (SwiGLU)
            routed_params = n_experts * (3 * hidden_size * moe_inter)
            # Shared (SwiGLU)
            # Is shared expert enabled? usually yes if size > 0
            shared_params = 3 * hidden_size * shared_inter

            ffn_params = gate_params + routed_params + shared_params
            desc = f"MoE (Routed: {n_experts}, Active: {active_experts}, Shared Inter: {shared_inter})"

            layers.append(Layer(f"Block {i} - FFN", "feedforward", ffn_params, (hidden_size, moe_inter), desc))
            total_params += ffn_params

            layers.append(Layer(f"Block {i}", "block_end", 0, (), ""))

        # 3. Final Norm
        layers.append(Layer("Final Norm", "normalization", hidden_size, (hidden_size,)))
        total_params += hidden_size

        # 4. Head
        if getattr(text_config, 'tie_word_embeddings', False):
            head_params = 0
            desc = "Tied"
        else:
            head_params = vocab_size * hidden_size
            desc = "Untied"
        layers.append(Layer("LM Head", "output", head_params, (hidden_size, vocab_size), desc))
        total_params += head_params

        return _finalize_architecture(
            Architecture(
                model_type="qwen3_5_moe",
                arch_type="decoder-only",
                total_layers=num_layers,
                total_params=total_params,
                memory_fp16_gb=total_params * 2 / (1024**3),
                parameters={
                    "vocab_size": vocab_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "vision_layers": int(getattr(vision_config, "depth", 0) or 0),
                },
                features=features,
                layers=layers,
            ),
            total_layers=num_layers,
        )
