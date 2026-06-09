"""Main content and per-column architecture rendering."""

from typing import Any

from ..core import Architecture


class _HtmlColumnsMixin:
    """Mixin providing HTML column rendering for architecture visualization."""
    def _render_main_content(self, arch: Architecture) -> str:
        svg_defs = """
        <svg id="svg-connections">
            <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="10"
                        refX="9" refY="5" orient="auto" markerUnits="strokeWidth">
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--text-tertiary)" />
                </marker>
                <marker id="arrowhead-up" markerWidth="9" markerHeight="9"
                        refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth">
                    <path d="M 0 0 L 9 4.5 L 0 9 z" fill="var(--color-layer)" />
                </marker>
                <marker id="arrowhead-down" markerWidth="9" markerHeight="9"
                        refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth">
                    <path d="M 0 0 L 9 4.5 L 0 9 z" fill="var(--color-tensor)" />
                </marker>
                <marker id="arrowhead-cross" markerWidth="9" markerHeight="9"
                        refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth">
                    <path d="M 0 0 L 9 4.5 L 0 9 z" fill="var(--color-activation)" />
                </marker>
            </defs>
        </svg>"""

        at = arch.arch_type
        col_backbone  = self._render_column_backbone(arch)
        col_dna       = self._render_column_arch_dna(arch)

        if at == "encoder-only":
            inner = (col_backbone
                     + self._render_column_encoder(arch)
                     + self._render_column_encoder_detail(arch)
                     + col_dna)
        elif at == "encoder-decoder":
            inner = (col_backbone
                     + self._render_column_encoder(arch)
                     + self._render_column_decoder(arch)
                     + self._render_column_cross_attention(arch)
                     + col_dna)
        else:  # decoder-only (default)
            inner = (col_backbone
                     + self._render_column_decoder(arch)
                     + self._render_column_components(arch)
                     + col_dna)

        return f"""
    <div class="main-container">
        {svg_defs}
        {inner}
    </div>"""

    def _render_column_backbone(self, arch: Architecture) -> str:
        at = arch.arch_type
        strip_cls = {"decoder-only":"arch-strip-decoder",
                     "encoder-only":"arch-strip-encoder",
                     "encoder-decoder":"arch-strip-encdec"}.get(at,"arch-strip-decoder")
        strip_lbl = {"decoder-only":"Decoder-Only","encoder-only":"Encoder-Only",
                     "encoder-decoder":"Encoder–Decoder"}.get(at,"Decoder-Only")
        badge_row = self._feature_badges(arch.special_features)
        model_type = self._html(arch.model_type)
        return f"""
        <div class="column col-1">
            <div class="column-header">
                <div class="column-title">Model Backbone</div>
                <div class="column-subtitle">End-to-End Pipeline</div>
            </div>
            <div class="arch-type-strip {strip_cls}">{strip_lbl}</div>

            <div id="node-text-input" data-node-id="node-text-input"
                 class="box box-input interactive">
                <span class="box-label">Text Input</span>
            </div>
            <div class="connector-vertical connector-arrow"></div>

            <div id="node-tokenizer" data-node-id="node-tokenizer"
                 class="box box-activation interactive">
                <span class="box-label">Tokenizer</span>
                <span class="box-sublabel">Vocab: {arch.parameters.get('vocab_size', 'N/A')}</span>
            </div>
            <div class="connector-vertical connector-arrow"></div>

            <div class="group-box" id="backbone-main">
                <div class="group-label">{model_type} Core</div>
                {badge_row}

                <div id="node-embedding" data-node-id="node-embedding"
                     class="box box-activation interactive">
                    <span class="box-label">Embedding</span>
                    <span class="box-sublabel">d={arch.parameters.get('hidden_size', 'N/A')}</span>
                </div>
                <div class="connector-vertical connector-arrow"></div>

                <div id="node-hidden-entry" data-node-id="node-hidden-entry"
                     class="box box-tensor">
                    <span class="box-label">Hidden States</span>
                    <span class="box-sublabel">[B, L, {arch.parameters.get('hidden_size', 'D')}]</span>
                </div>
                <div class="connector-vertical connector-arrow"></div>

                <div class="group-box group-layer"
                     id="layers-container" data-node-id="layers-container"
                     style="background: rgba(230,124,115,0.05); border-color: var(--color-layer);">
                    <div class="group-label" style="color: var(--color-layer);">
                        Layers (×{arch.total_layers})
                    </div>
                    {self._render_hy3_layer_groups(arch)}
                </div>

                <div class="connector-vertical connector-arrow"></div>
                <div id="node-rms-final" data-node-id="node-rms-final"
                     class="box box-norm interactive">
                    <span class="box-label">RMSNorm</span>
                </div>
            </div>

            <div class="connector-vertical connector-arrow"></div>
            <div id="node-linear-logits" data-node-id="node-linear-logits"
                 class="box box-input interactive">
                <span class="box-label">Linear</span>
                <span class="box-sublabel">Logits</span>
            </div>
            <div class="connector-vertical connector-arrow"></div>
            <div id="node-loss" data-node-id="node-loss"
                 class="box box-input interactive">
                <span class="box-label">Loss</span>
            </div>
            <div class="connector-vertical connector-arrow"></div>
            <div id="node-output" data-node-id="node-output"
                 class="box box-input interactive"
                 style="background:linear-gradient(135deg,#1967d2 0%,#174ea6 100%);">
                <span class="box-label">Output</span>
            </div>
        </div>"""

    def _render_column_decoder(self, arch: Architecture) -> str:
        residual_bg  = "var(--residual-copy-bg)"
        num_heads    = arch.parameters.get("num_heads", 32)
        arch.parameters.get("hidden_size", 4096)
        num_kv       = arch.parameters.get("num_kv_heads", num_heads)
        feats        = arch.special_features
        # Attention type label
        if "Sequence Mixer" in feats or "No KV Cache" in feats:
            attn_badge_label = "SSM / recurrent mixer"
        elif "GQA" in feats:
            attn_badge_label = f"GQA · {num_kv} kv-heads"
        elif "MQA" in feats:
            attn_badge_label = "MQA · 1 kv-head"
        else:
            attn_badge_label = f"MHA · {num_heads} heads"
        # Special indicators (SWA or MoE)
        swa_win = arch.parameters.get("sliding_window", 0)
        if "Sequence Mixer" in feats or "No KV Cache" in feats:
            attn_special_indicator = (
                '<div class="swa-indicator">No KV cache path · compression disabled</div>'
            )
        elif "SWA" in feats and swa_win:
            attn_special_indicator = (
                f'<div class="swa-indicator">⧉ Sliding Window {swa_win} tok</div>'
            )
        elif "CSA/HCA" in feats:
            attn_special_indicator = (
                f'<div class="swa-indicator">CSA/HCA · '
                f'{arch.parameters.get("compressed_attention_layers", 0)} compressed · '
                f'{arch.parameters.get("num_hash_layers", 0)} hash</div>'
            )
        elif "MoE" in feats:
            n_exp   = arch.parameters.get("num_experts", 8)
            top_k   = arch.parameters.get("top_k_experts", 2)
            experts = "".join(
                f'<div class="moe-expert {"active" if i < top_k else "passive"}">E{i+1}</div>'
                for i in range(min(n_exp, 8))
            )
            attn_special_indicator = (
                f'<div class="moe-router">{experts}'
                f'<span style="font-size:10px;color:var(--text-secondary);'
                f'width:100%;text-align:center;">top-{top_k} of {n_exp} active</span></div>'
            )
        else:
            attn_special_indicator = ""
        norm_type = "RMSNorm" if "RMSNorm" in feats else "LayerNorm"
        mlp_label, act_type = self._render_hy3_mlp_label(arch)
        mixer_label = "Sequence Mixer" if ("Sequence Mixer" in feats or "No KV Cache" in feats) else "Self-Attention"
        decoder_group_label = "MoE Transformer Block" if self._is_hy3(arch) else "Transformer Block"
        hy3_hint = self._render_hy3_decoder_hint(arch)
        router_bits = []
        if arch.parameters.get("router_sigmoid"):
            router_bits.append("sigmoid")
        if arch.parameters.get("router_bias"):
            router_bits.append("expert-bias")
        if arch.parameters.get("route_norm"):
            router_bits.append("RouteNorm")
        router_suffix = f" Router: {', '.join(router_bits)}." if router_bits else ""
        hy3_router_caption = (
            f'<div class="hy3-router-caption">Dense Prefix uses dense SwiGLU, later blocks route experts.{router_suffix}</div>'
            if self._is_hy3(arch)
            else ""
        )
        return f"""
        <div class="column col-2">
            <div class="column-header">
                <div class="column-title">Decoder Layer</div>
                <div class="column-subtitle">Single Transformer Block</div>
            </div>

            <div class="group-box" id="decoder-layer-detail" style="min-height:600px;">
                <div class="group-label">{decoder_group_label}</div>
                {hy3_hint}

                <div class="box box-tensor"
                     id="decoder-input" data-node-id="decoder-input">
                    <span class="box-label">Hidden_states</span>
                    <span class="box-sublabel">[B, L, D]</span>
                </div>

                <!-- ── Attention sub-block ── -->
                <div style="display:flex;width:100%;position:relative;padding-right:20px;">
                    <div class="flex-column">
                        <div class="connector-vertical connector-arrow"></div>
                        <div class="box box-norm interactive"
                             id="node-rms-attn" data-node-id="node-rms-attn">
                            <span class="box-label">{norm_type}</span>
                            <span class="box-sublabel">Pre-Attn</span>
                        </div>
                        <div class="connector-vertical connector-arrow"></div>
                        <div class="box box-attention interactive"
                             id="attention-module" data-node-id="attention-module">
                            <span class="box-label">{mixer_label}</span>
                            <span class="box-sublabel">{attn_badge_label}</span>
                            {attn_special_indicator}
                        </div>
                        <div class="connector-vertical connector-arrow"></div>
                        <div class="box box-tensor"
                             id="node-hidden-post-attn" data-node-id="node-hidden-post-attn">
                            <span class="box-label">Hidden_states</span>
                        </div>
                    </div>
                    <!-- Residual bracket -->
                    <div style="position:absolute;top:10px;bottom:10px;right:0;width:40px;">
                        <div style="position:absolute;top:0;right:0;width:100%;height:50%;
                             border-right:2px solid var(--color-residual);
                             border-top:2px solid var(--color-residual);
                             border-top-right-radius:8px;"></div>
                        <div style="position:absolute;top:50%;right:0;width:100%;height:50%;
                             border-right:2px solid var(--color-residual);
                             border-bottom:2px solid var(--color-residual);
                             border-bottom-right-radius:8px;"></div>
                        <span style="position:absolute;top:50%;right:-25px;
                              transform:translateY(-50%) rotate(90deg);
                              font-size:10px;color:var(--text-tertiary);
                              background:{residual_bg};">copy</span>
                    </div>
                </div>

                <div class="spacer"></div>
                <div class="box box-operator"
                     id="node-add-attn" data-node-id="node-add-attn">+</div>
                <div class="spacer"></div>
                <div class="box box-tensor"
                     id="node-hidden-mid" data-node-id="node-hidden-mid">
                    <span class="box-label">Hidden_states</span>
                </div>

                <!-- ── MLP sub-block ── -->
                <div style="display:flex;width:100%;position:relative;padding-right:20px;">
                    <div class="flex-column">
                        <div class="connector-vertical connector-arrow"></div>
                        <div class="box box-norm interactive"
                             id="node-rms-mlp" data-node-id="node-rms-mlp">
                            <span class="box-label">{norm_type}</span>
                            <span class="box-sublabel">Pre-MLP</span>
                        </div>
                        <div class="connector-vertical connector-arrow"></div>
                        <div class="box box-mlp interactive"
                             id="mlp-module" data-node-id="mlp-module">
                            <span class="box-label">{mlp_label}</span>
                            <span class="box-sublabel">{act_type}</span>
                            {hy3_router_caption}
                        </div>
                        <div class="connector-vertical connector-arrow"></div>
                        <div class="box box-tensor"
                             id="node-hidden-post-mlp" data-node-id="node-hidden-post-mlp">
                            <span class="box-label">Hidden_states</span>
                        </div>
                    </div>
                    <div style="position:absolute;top:10px;bottom:10px;right:0;width:40px;">
                        <div style="position:absolute;top:0;right:0;width:100%;height:50%;
                             border-right:2px solid var(--color-residual);
                             border-top:2px solid var(--color-residual);
                             border-top-right-radius:8px;"></div>
                        <div style="position:absolute;top:50%;right:0;width:100%;height:50%;
                             border-right:2px solid var(--color-residual);
                             border-bottom:2px solid var(--color-residual);
                             border-bottom-right-radius:8px;"></div>
                        <span style="position:absolute;top:50%;right:-25px;
                              transform:translateY(-50%) rotate(90deg);
                              font-size:10px;color:var(--text-tertiary);
                              background:{residual_bg};">copy</span>
                    </div>
                </div>

                <div class="spacer"></div>
                <div class="box box-operator"
                     id="node-add-mlp" data-node-id="node-add-mlp">+</div>
                <div class="connector-vertical connector-arrow"></div>
                <div class="box box-tensor"
                     id="decoder-output" data-node-id="decoder-output">
                    <span class="box-label">Hidden_states</span>
                    <span class="box-sublabel">[B, L, D]</span>
                </div>
            </div>
        </div>"""


    # ── Encoder column (shared by encoder-only and encoder-decoder) ──────────

    def _render_column_encoder(self, arch: Architecture) -> str:
        num_heads   = arch.parameters.get("num_heads", 16)
        hidden_size = arch.parameters.get("hidden_size", 1024)
        feats       = arch.special_features
        norm_type   = "RMSNorm" if "RMSNorm" in feats else "LayerNorm"
        act_type    = "SwiGLU" if "SwiGLU" in feats else "GELU" if "GELU" in feats else "ReLU"
        bidir_ind   = '<div class="bidir-indicator">↔ Bidirectional Attention</div>'
        enc_layers  = getattr(arch, "encoder_layers", 0) or arch.total_layers
        at          = arch.arch_type
        col_title   = "Encoder Stack" if at == "encoder-decoder" else "Encoder Block"
        return f"""
        <div class="column col-2">
            <div class="column-header">
                <div class="column-title">{col_title}</div>
                <div class="column-subtitle">Layer Detail (×{enc_layers})</div>
            </div>

            <!-- Arch strip -->
            <div class="arch-type-strip arch-strip-encoder">Encoder · Bidirectional</div>

            <div class="group-box group-layer"
                 id="encoder-layer-detail" data-node-id="encoder-layer-detail"
                 style="background:rgba(92,151,245,.04);border-color:var(--color-tensor);">
                <div class="group-label" style="color:var(--color-tensor);">
                    Encoder Layer
                </div>

                <div id="enc-input" data-node-id="enc-input" class="box box-tensor">
                    <span class="box-label">Hidden States</span>
                    <span class="box-sublabel">[B, L, {hidden_size}]</span>
                </div>
                <div class="connector-vertical connector-arrow"></div>

                <!-- Pre-norm -->
                <div class="box box-norm interactive"
                     id="enc-norm-attn" data-node-id="enc-norm-attn">
                    <span class="box-label">{norm_type}</span>
                    <span class="box-sublabel">Pre-Attn</span>
                </div>
                <div class="connector-vertical connector-arrow"></div>

                <!-- Bidirectional self-attention -->
                <div class="box box-attention interactive"
                     id="enc-attn" data-node-id="enc-attn">
                    <span class="box-label">Self-Attention</span>
                    <span class="box-sublabel">MHA · {num_heads} heads</span>
                    {bidir_ind}
                </div>
                <div class="connector-vertical connector-arrow"></div>

                <div class="box box-tensor"
                     id="enc-post-attn" data-node-id="enc-post-attn">
                    <span class="box-label">Attn Output</span>
                </div>
                <div class="spacer"></div>
                <div class="box box-operator"
                     id="enc-add-attn" data-node-id="enc-add-attn">+</div>
                <div class="connector-vertical connector-arrow"></div>

                <div class="box box-tensor"
                     id="enc-hidden-mid" data-node-id="enc-hidden-mid">
                    <span class="box-label">Hidden States</span>
                </div>
                <div class="connector-vertical connector-arrow"></div>

                <div class="box box-norm interactive"
                     id="enc-norm-ffn" data-node-id="enc-norm-ffn">
                    <span class="box-label">{norm_type}</span>
                    <span class="box-sublabel">Pre-FFN</span>
                </div>
                <div class="connector-vertical connector-arrow"></div>

                <div class="box box-mlp interactive"
                     id="enc-ffn" data-node-id="enc-ffn">
                    <span class="box-label">FFN</span>
                    <span class="box-sublabel">{act_type}</span>
                </div>
                <div class="connector-vertical connector-arrow"></div>

                <div class="box box-tensor"
                     id="enc-post-ffn" data-node-id="enc-post-ffn">
                    <span class="box-label">FFN Output</span>
                </div>
                <div class="spacer"></div>
                <div class="box box-operator"
                     id="enc-add-ffn" data-node-id="enc-add-ffn">+</div>
                <div class="connector-vertical connector-arrow"></div>

                <div class="box box-tensor"
                     id="enc-output" data-node-id="enc-output">
                    <span class="box-label">Encoder Output</span>
                    <span class="box-sublabel">[B, L, {hidden_size}]</span>
                </div>
            </div>
        </div>"""

    # ── Encoder-only detail panel (replaces component column for BERT) ───────

    def _render_column_encoder_detail(self, arch: Architecture) -> str:
        num_heads   = arch.parameters.get("num_heads", 16)
        hidden_size = arch.parameters.get("hidden_size", 1024)
        num_heads   = int(num_heads or 0) or 1
        head_dim    = hidden_size // num_heads
        vocab_size  = arch.parameters.get("vocab_size", 30522)
        return f"""
        <div class="column col-3">
            <div class="column-header">
                <div class="column-title">Attention Detail</div>
                <div class="column-subtitle">Bidirectional Self-Attention</div>
            </div>

            <div class="group-box group-attention"
                 id="enc-attn-detail" data-node-id="enc-attn-detail"
                 style="border-style:dashed; border-color:var(--color-tensor);">
                <div class="group-label" style="color:var(--color-tensor);">
                    Scaled Dot-Product (Bidir)
                </div>

                <!-- Shows ALL tokens attending to ALL tokens -->
                <div class="box box-tensor"><span class="box-label">Hidden States</span></div>
                <div class="connector-vertical connector-arrow"></div>
                <div class="flex-row">
                    <div class="box box-weight interactive"><span class="box-label">Q</span></div>
                    <div class="box box-weight interactive"><span class="box-label">K</span></div>
                    <div class="box box-weight interactive"><span class="box-label">V</span></div>
                </div>
                <div class="connector-vertical connector-arrow"></div>
                <div class="box box-tensor">
                    <span class="box-label">QKᵀ / √d</span>
                    <span class="box-sublabel">d={head_dim} · No Mask</span>
                </div>
                <div class="connector-vertical connector-arrow"></div>
                <div class="box box-norm">
                    <span class="box-label">Softmax</span>
                    <span class="box-sublabel">Full Attention</span>
                </div>
                <div class="connector-vertical connector-arrow"></div>
                <div class="box box-tensor">
                    <span class="box-label">Attn × V</span>
                </div>
                <div class="connector-vertical connector-arrow"></div>
                <div class="box box-weight">
                    <span class="box-label">O-Proj</span>
                </div>
            </div>

            <div class="spacer-lg"></div>

            <!-- Output heads (MLM / NSP / Classification) -->
            <div class="group-box"
                 style="border-style:dashed; border-color:var(--color-activation);">
                <div class="group-label" style="color:var(--color-activation);">
                    Output Heads
                </div>
                <div class="flex-row">
                    <div class="flex-column">
                        <div class="box box-activation interactive">
                            <span class="box-label">[CLS]</span>
                            <span class="box-sublabel">Classification</span>
                        </div>
                    </div>
                    <div class="flex-column">
                        <div class="box box-activation interactive">
                            <span class="box-label">MLM Head</span>
                            <span class="box-sublabel">Vocab: {vocab_size}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>"""

    # ── Cross-attention column (encoder-decoder only) ────────────────────────

    def _render_column_cross_attention(self, arch: Architecture) -> str:
        num_heads   = arch.parameters.get("num_heads", 16)
        hidden_size = arch.parameters.get("hidden_size", 1024)
        num_heads   = int(num_heads or 0) or 1
        head_dim    = hidden_size // num_heads
        return f"""
        <div class="column col-4">
            <div class="column-header">
                <div class="column-title">Cross-Attention</div>
                <div class="column-subtitle">Decoder ↔ Encoder</div>
            </div>

            <div class="arch-type-strip arch-strip-encdec">
                Encoder → Decoder Bridge
            </div>

            <div class="group-box group-cross"
                 id="cross-attn-detail" data-node-id="cross-attn-detail"
                 style="border-style:dashed;">
                <div class="group-label" style="color:var(--color-activation);">
                    Cross-Attention
                </div>

                <div class="flex-row">
                    <div class="flex-column">
                        <div class="box box-tensor" style="background:rgba(52,168,83,.2);
                                color:var(--color-activation);border:1px solid rgba(52,168,83,.5);">
                            <span class="box-label">Encoder Out</span>
                            <span class="box-sublabel">K, V source</span>
                        </div>
                    </div>
                    <div class="flex-column">
                        <div class="box box-tensor">
                            <span class="box-label">Decoder State</span>
                            <span class="box-sublabel">Q source</span>
                        </div>
                    </div>
                </div>
                <div class="connector-vertical connector-arrow"></div>

                <div class="flex-row">
                    <div class="box box-weight interactive"
                         style="background:var(--color-activation);color:white;">
                        <span class="box-label">K (enc)</span>
                    </div>
                    <div class="box box-weight interactive"
                         style="background:var(--color-activation);color:white;">
                        <span class="box-label">V (enc)</span>
                    </div>
                    <div class="box box-weight interactive">
                        <span class="box-label">Q (dec)</span>
                    </div>
                </div>
                <div class="connector-vertical connector-arrow"></div>

                <div class="box box-cross-attn"
                     id="cross-attn-compute" data-node-id="cross-attn-compute">
                    <span class="box-label">QKᵀ / √d</span>
                    <span class="box-sublabel">d={head_dim} · Cross-Modal</span>
                </div>
                <div class="connector-vertical connector-arrow"></div>

                <div class="box box-norm">
                    <span class="box-label">Softmax</span>
                    <span class="box-sublabel">Attn over Enc Seq</span>
                </div>
                <div class="connector-vertical connector-arrow"></div>

                <div class="box box-cross-attn"
                     id="cross-attn-out" data-node-id="cross-attn-out">
                    <span class="box-label">Context Vector</span>
                </div>
                <div class="connector-vertical connector-arrow"></div>

                <div class="box box-weight">
                    <span class="box-label">O-Proj</span>
                </div>
                <div class="connector-vertical connector-arrow"></div>

                <div class="box box-tensor">
                    <span class="box-label">Decoder State</span>
                    <span class="box-sublabel">Enriched by Enc ctx</span>
                </div>
            </div>
        </div>"""

    # ── Architecture DNA comparison column ───────────────────────────────────

    def _render_column_arch_dna(self, arch: Architecture) -> str:
        feats  = set(arch.special_features)
        at     = arch.arch_type
        hs     = arch.parameters.get("hidden_size", 4096)
        nh     = arch.parameters.get("num_heads", 32)
        nkv    = arch.parameters.get("num_kv_heads", nh)
        vs     = arch.parameters.get("vocab_size", 0)
        mlen   = arch.parameters.get("max_position", 0)
        n_exp  = arch.parameters.get("num_experts", 0)
        sw     = arch.parameters.get("sliding_window", 0)
        head_d = hs // nh if nh else 0

        def _h(v): return f'<span class="dna-val highlight">{v}</span>'
        def _w(v): return f'<span class="dna-val warn">{v}</span>'
        def _o(v): return f'<span class="dna-val ok">{v}</span>'
        def _n(v): return f'<span class="dna-val">{v}</span>'

        def row(k, v_html) -> Any:
            return f'<div class="dna-row"><span class="dna-key">{k}</span>{v_html}</div>'

        # Structural dimension card
        dim_rows = (
            row("Hidden dim",  _h(hs))
            + row("Heads",     _n(f"{nh}H"))
            + (row("KV heads", _w(f"{nkv} (GQA)")) if nkv < nh else
               row("KV heads", _n(f"{nkv} (MHA)")))
            + row("Head dim",  _n(head_d))
            + row("Layers",    _n(arch.total_layers))
            + (row("Vocab",    _n(f"{vs//1000}k")) if vs else "")
            + (row("Max len",  _h(f"{mlen//1024}k tok")) if mlen else "")
            + (row("Experts",  _w(f"{n_exp}")) if n_exp else "")
            + (row("SWA win",  _w(f"{sw}")) if sw else "")
        )

        # Memory / compute card
        mem = arch.memory_fp16_gb
        mem_color = _w if mem > 30 else (_h if mem < 5 else _n)
        compute_rows = (
            row("FP16 mem",   mem_color(f"{mem:.1f} GB"))
            + row("Params",  _n(f"{arch.total_params/1e9:.1f}B"))
        )

        # Attention mechanism card
        attn_type = ("GQA" if "GQA" in feats else
                     "MQA" if "MQA" in feats else "MHA")
        pos_type  = ("RoPE"   if "RoPE"   in feats else
                     "ALiBi"  if "ALiBi"  in feats else
                     "RelPos" if "RelPos" in feats else
                     "AbsPos" if "AbsPos" in feats else "Learned")
        norm_type = "RMSNorm" if "RMSNorm" in feats else "LayerNorm"
        act_type  = ("SwiGLU" if "SwiGLU" in feats else
                     "GeGLU"  if "GeGLU"  in feats else
                     "GELU"   if "GELU"   in feats else "ReLU")
        mech_rows = (
            row("Attention",  _h(attn_type))
            + row("Position", _n(pos_type))
            + row("Norm",     _n(norm_type))
            + row("Activation", _o(act_type))
            + row("Masking",  _w("Causal") if "Causal" in feats
                              else _o("Bidirectional"))
            + (row("Extras",  _w("SWA+MoE")) if ("SWA" in feats and "MoE" in feats) else
               row("Extras",  _w("SWA"))  if "SWA" in feats else
               row("Extras",  _w("MoE"))  if "MoE" in feats else
               row("Extras",  _n("—")))
        )

        # Architecture comparison table
        def chk(cond) -> str:  return '<span class="cell-yes">✔</span>' if cond else '<span class="cell-no">–</span>'
        def act_cell(cond) -> str: return f'<td class="cell-active">{chk(cond)}</td>'
        def nrm_cell(cond) -> str: return f'<td>{chk(cond)}</td>'

        is_dec = at == "decoder-only"
        is_enc = at == "encoder-only"
        is_ed  = at == "encoder-decoder"

        compare_table = f"""
        <table class="dna-compare-table">
            <thead><tr>
                <th>Feature</th>
                <th class="cell-dec">Dec-Only</th>
                <th class="cell-enc">Enc-Only</th>
                <th class="cell-encdec">Enc–Dec</th>
            </tr></thead>
            <tbody>
            <tr>
                <td class="row-key">Causal Mask</td>
                {act_cell(is_dec)}{nrm_cell(False)}{nrm_cell(True)}
            </tr><tr>
                <td class="row-key">Bidirectional</td>
                {nrm_cell(False)}{act_cell(is_enc)}{nrm_cell(True)}
            </tr><tr>
                <td class="row-key">Cross-Attention</td>
                {nrm_cell(False)}{nrm_cell(False)}{act_cell(is_ed)}
            </tr><tr>
                <td class="row-key">Autoregressive</td>
                {act_cell(is_dec)}{nrm_cell(False)}{act_cell(is_ed)}
            </tr><tr>
                <td class="row-key">Text Generation</td>
                {act_cell(is_dec)}{nrm_cell(False)}{act_cell(is_ed)}
            </tr><tr>
                <td class="row-key">Seq2Seq Tasks</td>
                {nrm_cell(False)}{nrm_cell(False)}{act_cell(is_ed)}
            </tr><tr>
                <td class="row-key">Classification</td>
                {nrm_cell(is_dec)}{act_cell(is_enc)}{nrm_cell(True)}
            </tr>
            </tbody>
        </table>"""

        return f"""
        <div class="column col-dna">
            <div class="column-header">
                <div class="column-title">Architecture DNA</div>
                <div class="column-subtitle">Model Fingerprint</div>
            </div>

            <!-- Dimensions -->
            <div class="dna-card">
                <div class="dna-card-title">📐 Dimensions</div>
                {dim_rows}
            </div>

            <!-- Memory / Compute -->
            <div class="dna-card">
                <div class="dna-card-title">⚡ Compute Budget</div>
                {compute_rows}
            </div>

            <!-- Mechanism fingerprint -->
            <div class="dna-card">
                <div class="dna-card-title">🔬 Mechanism</div>
                {mech_rows}
            </div>

            <!-- Cross-arch comparison -->
            <div class="dna-card">
                <div class="dna-card-title">⚖ Architecture Comparison</div>
                {compare_table}
            </div>
        </div>"""

    def _render_column_components(self, arch: Architecture) -> str:
        arch.parameters.get("num_heads", 32)
        arch.parameters.get("hidden_size", 4096)
        hy3_summary = self._render_hy3_components_summary(arch)
        mlp_group_label = "MoE FFN" if self._is_hy3(arch) else "MLP"
        mlp_first_box = "Dense Prefix Gate" if self._is_hy3(arch) else "Linear"
        mlp_second_box = "Expert Router" if self._is_hy3(arch) else "Linear"
        hy3_section_tag = '<div class="hy3-section-tag moe">MoE Blocks</div>' if self._is_hy3(arch) else ""
        mlp_note = (
            '<div class="hy3-inline-note">Dense Prefix uses the first gate only. Routed blocks activate top-k experts plus the shared expert.</div>'
            if self._is_hy3(arch)
            else ""
        )
        return f"""
        <div class="column col-3">
            <div class="column-header">
                <div class="column-title">Micro-Architecture</div>
                <div class="column-subtitle">Component Details</div>
            </div>

            {hy3_summary}

            <!-- ── Attention detail ── -->
            <div class="group-box group-attention"
                 id="attention-detail" data-node-id="attention-detail"
                 style="border-style:dashed;">
                <div class="group-label">Attn</div>

                <div class="box box-tensor"><span class="box-label">Hidden_states</span></div>
                <div class="connector-vertical connector-arrow"></div>

                <div class="flex-row">
                    <div class="box box-weight interactive tooltip" data-tooltip="Query Projection">
                        <span class="box-label">Query</span>
                    </div>
                    <div class="box box-weight interactive tooltip" data-tooltip="Key Projection">
                        <span class="box-label">Key</span>
                    </div>
                    <div class="box box-weight interactive tooltip" data-tooltip="Value Projection">
                        <span class="box-label">Value</span>
                    </div>
                </div>
                <div class="connector-vertical connector-arrow"></div>

                <div class="box box-tensor" style="background:var(--color-tensor);">
                    <span class="box-label">Apply_rotary_pos_emb</span>
                </div>
                <div class="connector-vertical connector-arrow"></div>

                <div class="flex-row">
                    <div class="box box-weight"><span class="box-label">Query</span></div>
                    <div class="box box-weight"><span class="box-label">Key</span></div>
                </div>
                <div class="connector-vertical connector-arrow"></div>

                <div class="box box-tensor" style="background:var(--color-tensor);">
                    <span class="box-label">Compute_module</span>
                </div>
                <div class="connector-vertical connector-arrow"></div>

                <div class="flex-row">
                    <div class="box box-weight"><span class="box-label">Query</span></div>
                    <div class="box box-weight"><span class="box-label">Key</span></div>
                </div>
                <div class="connector-vertical connector-arrow"></div>

                <div class="box box-weight"
                     style="background:var(--color-layer);color:white;transform:skewX(-10deg);">
                    <span class="box-label">Dot_attn</span>
                </div>
                <div class="connector-vertical connector-arrow"></div>

                <div class="box box-tensor" style="background:var(--color-tensor);">
                    <span class="box-label">Attention_weight</span>
                </div>
                <div class="connector-vertical connector-arrow"></div>

                <div class="box box-operator" style="width:30px;height:30px;">+</div>
                <div class="connector-vertical connector-arrow"></div>

                <div class="box box-norm" style="background:var(--color-norm);">
                    <span class="box-label">Softmax</span>
                </div>
                <div class="connector-vertical connector-arrow"></div>

                <div class="box box-tensor" style="background:var(--color-tensor);">
                    <span class="box-label">Matmul</span>
                </div>
                <div class="connector-vertical connector-arrow"></div>

                <div class="flex-row">
                    <div class="box box-input"><span class="box-label">O_Linear</span></div>
                    <div class="box box-input"><span class="box-label">Output</span></div>
                </div>
            </div>

            <div class="spacer-lg"></div>

            <!-- ── MLP detail ── -->
            <div class="group-box group-mlp"
                 id="mlp-detail" data-node-id="mlp-detail"
                 style="border-style:dashed;">
                <div class="group-label">{mlp_group_label}</div>
                {hy3_section_tag}

                <div class="box box-tensor" style="width:80px;"><span class="box-label">HS</span></div>
                <div class="connector-vertical connector-arrow"></div>

                <div class="flex-row">
                    <div class="flex-column">
                        <div class="box box-weight interactive"
                             style="background:var(--color-layer);color:white;">
                            <span class="box-label">{mlp_first_box}</span>
                        </div>
                        <div class="connector-vertical connector-arrow"></div>
                        <div class="box box-norm" style="background:var(--color-norm);">
                            <span class="box-label">Act</span>
                        </div>
                    </div>
                    <div class="flex-column">
                        <div class="box box-activation" style="background:var(--color-activation);">
                            <span class="box-label">{mlp_second_box}</span>
                        </div>
                    </div>
                </div>
                <div class="connector-vertical connector-arrow"></div>

                <div class="box box-operator"
                     style="background:var(--color-tensor);width:30px;height:30px;">×</div>
                <div class="connector-vertical connector-arrow"></div>

                <div class="flex-row">
                    <div class="box box-weight"
                         style="background:var(--color-weight);color:#3c4043;">
                        <span class="box-label">Linear</span>
                    </div>
                    <div class="box box-tensor" style="width:60px;">
                        <span class="box-label">HS</span>
                    </div>
                </div>
                {mlp_note}
            </div>
        </div>"""

    # ── Export library tags ───────────────────────────────────────────────────
