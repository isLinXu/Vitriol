"""Intern-S1 family analyzer."""

from typing import Any

from ..core import Architecture, Layer
from ._helpers import _finalize_architecture
from .base import TransformerAnalyzer


class InternS1Analyzer(TransformerAnalyzer):
    """Intern-S1-Pro Analyzer (Multimodal: Text + Vision + TimeSeries)."""

    def analyze(self, config: Any) -> Architecture:
        # Intern-S1-Pro is a composite model
        # Main config has "vision_config", "ts_config", "text_config"

        # We need to extract the "text_config" (LLM) as the base
        # And attach Vision/TS encoders

        text_config = getattr(config, 'text_config', config) # Fallback if direct
        vision_config = getattr(config, 'vision_config', None)
        ts_config = getattr(config, 'ts_config', None)

        # Base LLM Analysis (using text_config)
        vocab_size = getattr(text_config, 'vocab_size', 0)
        hidden_size = getattr(text_config, 'hidden_size', 0)
        num_layers = getattr(text_config, 'num_hidden_layers', 0)
        num_heads = getattr(text_config, 'num_attention_heads', 0)
        num_kv_heads = getattr(text_config, 'num_key_value_heads', num_heads)

        # MoE?
        # "num_experts": 512, "num_experts_per_tok": 8
        n_experts = getattr(text_config, 'num_experts', 0)
        active_experts = getattr(text_config, 'num_experts_per_tok', 0)

        features = ["Intern-S1-Pro"]
        if n_experts > 0:
            features.append("MoE")
        if vision_config:
            features.append("Vision")
        if ts_config:
            features.append("TimeSeries")

        layers = []
        total_params = 0

        # --- Vision Encoder ---
        if vision_config:
            v_hidden = getattr(vision_config, 'hidden_size', 1024)
            v_layers = getattr(vision_config, 'depth', 24)
            # ViT-Large/24 approx
            v_params = 12 * v_layers * v_hidden * v_hidden
            layers.append(Layer("Vision Encoder", "encoder", v_params, (v_layers, v_hidden), f"ViT-L/{v_layers}"))
            total_params += v_params

            # Projector? Usually a linear or MLP
            mm_hidden = getattr(vision_config, 'out_hidden_size', v_hidden)
            proj_params = v_hidden * mm_hidden
            layers.append(Layer("Vision Projector", "adapter", proj_params, (v_hidden, mm_hidden), "Projection"))
            total_params += proj_params

        # --- Time Series Encoder ---
        if ts_config:
            ts_hidden = getattr(ts_config, 'd_model', 768)
            ts_layers = getattr(ts_config, 'encoder_layers', 17)
            # Encoder-Decoder structure in TS config?
            # "encoder_layers": 17, "decoder_layers": 4
            ts_dec_layers = getattr(ts_config, 'decoder_layers', 4)

            ts_enc_params = 12 * ts_layers * ts_hidden * ts_hidden
            ts_dec_params = 12 * ts_dec_layers * ts_hidden * ts_hidden

            ts_params = ts_enc_params + ts_dec_params
            layers.append(Layer("TimeSeries Model", "encoder", ts_params, (ts_layers + ts_dec_layers, ts_hidden), "TS-Transformer"))
            total_params += ts_params

            # TS Adapter/Projector
            ts_out = getattr(ts_config, 'ts_adapt_out_dim', 1024)
            ts_proj = ts_hidden * ts_out
            layers.append(Layer("TS Projector", "adapter", ts_proj, (ts_hidden, ts_out), "Adapter"))
            total_params += ts_proj

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
            head_dim = getattr(text_config, 'head_dim', hidden_size // num_heads)
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

            # FFN / MoE
            # InternLM MoE: Gate + Experts
            intermediate_size = getattr(text_config, 'intermediate_size', 0)
            moe_inter_size = getattr(text_config, 'moe_intermediate_size', intermediate_size)

            # Check if this layer is MoE?
            # "mlp_only_layers": [] -> all MoE? or implied by num_experts > 0
            # Assuming all layers MoE if num_experts > 0

            if n_experts > 0:
                gate_params = hidden_size * n_experts
                expert_params = n_experts * (3 * hidden_size * moe_inter_size)
                ffn_params = gate_params + expert_params
                desc = f"MoE (Experts: {n_experts}, Active: {active_experts})"
            else:
                ffn_params = 3 * hidden_size * intermediate_size
                desc = f"Dense (Inter: {intermediate_size})"

            layers.append(Layer(f"Block {i} - FFN", "feedforward", ffn_params, (hidden_size, moe_inter_size), desc))
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
                model_type="interns1_pro", # Composite
                arch_type="encoder-decoder", # Multimodal
                total_layers=num_layers,
                total_params=total_params,
                memory_fp16_gb=total_params * 2 / (1024**3),
                parameters={
                    "vocab_size": vocab_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "vision_layers": int(getattr(vision_config, "depth", 0) or 0),
                    "ts_encoder_layers": int(getattr(ts_config, "encoder_layers", 0) or 0),
                    "ts_decoder_layers": int(getattr(ts_config, "decoder_layers", 0) or 0),
                },
                features=features,
                layers=layers,
            ),
            total_layers=num_layers,
        )
