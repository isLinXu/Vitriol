"""Header, feature badges and per-family overview sections."""

from typing import Dict

from ..core import Architecture


class _HtmlSectionsMixin:
    def _render_header(self, arch: Architecture, stats: Dict) -> str:
        theme_buttons = " ".join(
            f'<button class="theme-btn{" active" if tid == self.theme else ""}" '
            f'data-theme-id="{tid}">{td["display_name"]}</button>'
            for tid, td in self.THEMES.items()
        )
        model_type = self._html(arch.model_type)
        return f"""
    <header>
        <!-- Left: Logo + model badge -->
        <div class="header-left">
            <div class="logo">
                <div class="logo-icon">A</div>
                <span>Vitriol</span>
            </div>
            <div class="model-badge">{model_type}</div>
        </div>

        <!-- Center: Architecture stats -->
        <div class="header-center">
            <div class="stats-bar">
                <div class="stat-item">
                    <span class="stat-label">Params</span>
                    <span class="stat-value">{stats['total_params']/1e9:.2f}B</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Layers</span>
                    <span class="stat-value">{stats['total_layers']}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Head Dim</span>
                    <span class="stat-value">{stats['head_dim']}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Mem FP16</span>
                    <span class="stat-value">{stats['memory_fp16_gb']:.1f} GB</span>
                </div>
            </div>
        </div>

        <!-- Right: Theme switcher + Export buttons -->
        <div class="header-right">
            <div class="theme-switcher">
                <span class="theme-switcher-label">Theme</span>
                {theme_buttons}
            </div>

            <div class="export-group">
                <span class="export-label">Export</span>
                <button class="export-btn" id="btn-export-png" title="Export as PNG (2× retina)">
                    🖼 PNG
                </button>
                <button class="export-btn" id="btn-export-svg" title="Export as SVG (vector)">
                    ✏️ SVG
                </button>
            </div>
        </div>
    </header>

    <!-- Toast notification -->
    <div id="export-toast"></div>"""

    # ── Main content (3 columns) ──────────────────────────────────────────────

    # ── Feature-badge helper ──────────────────────────────────────────────────
    _BADGE_MAP = {
        # Attention
        "MHA":  ("MHA",  "badge-attn"),
        "GQA":  ("GQA",  "badge-attn"),
        "MQA":  ("MQA",  "badge-attn"),
        "SWA":  ("SWA",  "badge-special"),
        "CSA/HCA": ("CSA/HCA", "badge-special"),
        "Hash Attention": ("HashAttn", "badge-special"),
        "Compressed Attention": ("CompAttn", "badge-special"),
        "Sequence Mixer": ("SeqMix", "badge-special"),
        "State Space": ("SSM", "badge-special"),
        # Norm
        "RMSNorm":   ("RMSNorm",  "badge-norm"),
        "LayerNorm": ("LayerNorm","badge-norm"),
        "DeepNorm":  ("DeepNorm", "badge-norm"),
        "QK Norm":   ("QK Norm",  "badge-norm"),
        "QK-Norm":   ("QK Norm",  "badge-norm"),
        "RouteNorm": ("RouteNorm", "badge-norm"),
        # Activation
        "SwiGLU": ("SwiGLU","badge-act"),
        "GELU":   ("GELU",  "badge-act"),
        "GeGLU":  ("GeGLU", "badge-act"),
        "ReLU":   ("ReLU",  "badge-act"),
        # Positional
        "RoPE":       ("RoPE",    "badge-pos"),
        "YARN RoPE":  ("YARN",    "badge-pos"),
        "Compressed RoPE": ("CompRoPE", "badge-pos"),
        "ALiBi":      ("ALiBi",   "badge-pos"),
        "AbsPos":     ("AbsPos",  "badge-pos"),
        "LearnedPos": ("LrnPos",  "badge-pos"),
        "RelPos":     ("RelPos",  "badge-pos"),
        # Special
        "MoE":         ("MoE",    "badge-special"),
        "FP8":         ("FP8",    "badge-special"),
        "DeepSeek-V4": ("DS-V4",  "badge-special"),
        "Mamba":       ("Mamba",  "badge-special"),
        "RWKV":        ("RWKV",   "badge-special"),
        "RetNet":      ("RetNet", "badge-special"),
        "Hyena":       ("Hyena",  "badge-special"),
        "No KV Cache": ("NoKV",   "badge-special"),
        "Sigmoid Router": ("Sigmoid", "badge-special"),
        "Router Bias": ("RouterBias", "badge-special"),
        "CrossAttn":   ("XAttn",  "badge-special"),
        "Causal":      ("Causal", "badge-causal"),
        "Bidirectional":("BiDir", "badge-bidir"),
        "Hybrid":      ("Hybrid", "badge-special"),
        "Hybrid Attention": ("Hybrid", "badge-special"),
    }

    def _feature_badges(self, features, subset=None):
        """Render a row of feature badges for a given feature list."""
        badges = []
        for f in features:
            if subset and f not in subset:
                continue
            if f in self._BADGE_MAP:
                label, cls = self._BADGE_MAP[f]
                badges.append(f'<span class="feat-badge {cls}">{label}</span>')
        if not badges:
            return ""
        return f'<div class="feat-badges">{" ".join(badges)}</div>'

    @staticmethod
    def _is_hy3(arch: Architecture) -> bool:
        return str(arch.model_type or "").lower() == "hy_v3"

    @staticmethod
    def _is_deepseek_v4(arch: Architecture) -> bool:
        return str(arch.model_type or "").lower() == "deepseek_v4"

    def _render_deepseek_v4_overview(self, arch: Architecture) -> str:
        if not self._is_deepseek_v4(arch):
            return ""
        hash_layers = int(arch.parameters.get("num_hash_layers", 0) or 0)
        compressed_layers = int(arch.parameters.get("compressed_attention_layers", 0) or 0)
        total_layers = int(arch.total_layers or 0)
        other_layers = max(total_layers - hash_layers - compressed_layers, 0)
        kv_heads = int(arch.parameters.get("num_kv_heads", 0) or 0)
        num_heads = int(arch.parameters.get("num_heads", 0) or 0)
        max_pos = int(arch.parameters.get("max_position", 0) or 0)
        quant_method = str(arch.parameters.get("quant_method") or "runtime").upper()
        index_topk = int(arch.parameters.get("index_topk", 0) or 0)
        compress_rope_theta = arch.parameters.get("compress_rope_theta", 0)
        context_label = f"{max_pos // 1024}K Context" if max_pos else "Long Context"
        return f"""
    <section class="hy3-overview">
        <div class="hy3-overview-grid">
            <div class="hy3-card">
                <div class="hy3-card-title">DeepSeek V4 Compression Map</div>
                <div class="hy3-card-copy">
                    <strong>Hash Attention</strong> covers the first {hash_layers} layers, then
                    <strong>CSA/HCA</strong> compressed attention covers {compressed_layers} decoder blocks.
                    KV compression presets treat these as non-full attention unless explicitly overridden.
                </div>
            </div>
            <div class="hy3-card">
                <div class="hy3-card-title">Attention Layout</div>
                <div class="hy3-chip-row">
                    <span class="hy3-chip hy3-chip-router">Hash · {hash_layers} layers</span>
                    <span class="hy3-chip hy3-chip-moe">CSA/HCA · {compressed_layers} layers</span>
                    <span class="hy3-chip hy3-chip-dense">Other · {other_layers} layers</span>
                    <span class="hy3-chip hy3-chip-attn">MQA · {kv_heads} / {num_heads} kv-heads</span>
                </div>
            </div>
            <div class="hy3-card">
                <div class="hy3-card-title">Compression + Runtime</div>
                <div class="hy3-chip-row">
                    <span class="hy3-chip hy3-chip-context">{context_label}</span>
                    <span class="hy3-chip hy3-chip-context">FP8 · {quant_method}</span>
                    <span class="hy3-chip hy3-chip-router">Index top-k · {index_topk}</span>
                    <span class="hy3-chip hy3-chip-attn">Compressed RoPE · {compress_rope_theta:g}</span>
                </div>
            </div>
        </div>
    </section>"""

    def _render_hy3_overview(self, arch: Architecture) -> str:
        if not self._is_hy3(arch):
            return ""
        dense_prefix = int(arch.parameters.get("dense_prefix_layers", 0) or 0)
        num_experts = int(arch.parameters.get("num_experts", 0) or 0)
        top_k = int(arch.parameters.get("top_k_experts", 0) or 0)
        shared = int(arch.parameters.get("shared_experts", arch.parameters.get("num_shared_experts", 0)) or 0)
        mtp_layers = int(arch.parameters.get("mtp_layers", 0) or 0)
        kv_heads = int(arch.parameters.get("num_kv_heads", 0) or 0)
        num_heads = int(arch.parameters.get("num_heads", 0) or 0)
        max_pos = int(arch.parameters.get("max_position", 0) or 0)

        def _fmt_billion(value: float) -> str:
            """Format a value in billions for compact display (e.g. 123B / 21B)."""
            if value <= 0:
                return "N/A"
            if value >= 100:
                s = f"{value:.0f}"
            elif value >= 10:
                s = f"{value:.1f}"
            else:
                s = f"{value:.2f}"
            s = s.rstrip("0").rstrip(".")
            return f"{s}B"

        # Total params: use Architecture.total_params as the source of truth (avoid hardcoded "looks-real" numbers).
        total_params_b = float(getattr(arch, "total_params", 0) or 0) / 1e9
        total_params_label = _fmt_billion(total_params_b) if total_params_b > 0 else "N/A"

        # Active params: prefer analyzer-provided activated_params (unit: raw parameter count),
        # fall back to the legacy active_params_b (unit: billions).
        active_params_label = ""
        activated_params = arch.parameters.get("activated_params", 0)
        try:
            activated_params_int = int(activated_params or 0)
        except Exception:
            activated_params_int = 0
        if activated_params_int > 0:
            active_params_label = _fmt_billion(activated_params_int / 1e9)
        else:
            apb = arch.parameters.get("active_params_b", None)
            try:
                apb_float = float(apb) if apb is not None else 0.0
            except Exception:
                apb_float = 0.0
            if apb_float > 0:
                active_params_label = _fmt_billion(apb_float)

        params_chip = ""
        if total_params_label != "N/A" and active_params_label:
            params_chip = f'<span class="hy3-chip hy3-chip-context">{total_params_label} / {active_params_label} active</span>'
        elif total_params_label != "N/A":
            params_chip = f'<span class="hy3-chip hy3-chip-context">{total_params_label} total</span>'

        router_scale = float(arch.parameters.get("router_scaling_factor", 0.0) or 0.0)
        router_chips = []
        if arch.parameters.get("router_sigmoid"):
            router_chips.append('<span class="hy3-chip hy3-chip-router">Sigmoid Router</span>')
        if arch.parameters.get("router_bias"):
            router_chips.append('<span class="hy3-chip hy3-chip-router">Router Bias</span>')
        if arch.parameters.get("route_norm"):
            router_chips.append('<span class="hy3-chip hy3-chip-router">RouteNorm</span>')
        if router_scale:
            router_chips.append(f'<span class="hy3-chip hy3-chip-router">Scale · {router_scale:g}</span>')
        router_chip_html = "".join(router_chips)
        context_label = f"{max_pos // 1024}K Context" if max_pos else "Long Context"
        return f"""
    <section class="hy3-overview">
        <div class="hy3-overview-grid">
            <div class="hy3-card">
                <div class="hy3-card-title">Hy3 Summary</div>
                <div class="hy3-card-copy">
                    <strong>Dense Prefix</strong> starts with {dense_prefix} layer, then the stack switches to
                    <strong>MoE top-{top_k} / {num_experts}</strong> routing for the remaining decoder blocks.
                    A separate <strong>MTP Head</strong> adds {mtp_layers} next-token prediction layer.
                </div>
            </div>
            <div class="hy3-card">
                <div class="hy3-card-title">Routing</div>
                <div class="hy3-chip-row">
                    <span class="hy3-chip hy3-chip-dense">Dense Prefix · {dense_prefix} layer</span>
                    <span class="hy3-chip hy3-chip-moe">MoE top-{top_k} / {num_experts}</span>
                    <span class="hy3-chip hy3-chip-moe">Shared Experts · {shared}</span>
                    {router_chip_html}
                </div>
            </div>
            <div class="hy3-card">
                <div class="hy3-card-title">Attention + Context</div>
                <div class="hy3-chip-row">
                    <span class="hy3-chip hy3-chip-attn">GQA · {kv_heads} / {num_heads} kv-heads</span>
                    <span class="hy3-chip hy3-chip-mtp">MTP · {mtp_layers} layer</span>
                    <span class="hy3-chip hy3-chip-context">{context_label}</span>
                    {params_chip}
                </div>
            </div>
        </div>
    </section>"""

    def _render_hy3_layer_groups(self, arch: Architecture) -> str:
        if not self._is_hy3(arch):
            return f"""
                    <div class="box box-layer interactive" id="layer-first">
                        <span class="box-label">Layer 1</span>
                    </div>
                    <div class="box box-layer" style="opacity:0.5;cursor:default;">
                        <span class="box-label">⋮</span>
                    </div>
                    <div class="box box-layer interactive" id="layer-last">
                        <span class="box-label">Layer {arch.total_layers}</span>
                    </div>"""

        dense_prefix = int(arch.parameters.get("dense_prefix_layers", 0) or 0)
        mtp_layers = int(arch.parameters.get("mtp_layers", 0) or 0)
        moe_layers = max(int(arch.total_layers or 0) - dense_prefix, 0)
        return f"""
                    <div class="hy3-layer-groups">
                        <div class="hy3-layer-box dense interactive" id="layer-first">
                            <strong>Dense Prefix · {dense_prefix} layer</strong>
                            <span>Block 0 keeps dense SwiGLU before MoE routing begins.</span>
                        </div>
                        <div class="hy3-layer-box moe interactive" id="layer-last">
                            <strong>MoE Blocks · {moe_layers} layers</strong>
                            <span>Blocks 1-{arch.total_layers} use top-{arch.parameters.get('top_k_experts', 0)} of {arch.parameters.get('num_experts', 0)} active experts.</span>
                        </div>
                        <div class="hy3-layer-box mtp">
                            <strong>MTP Head · {mtp_layers} layer</strong>
                            <span>Next-N prediction branch stays separate from the main decoder stack.</span>
                        </div>
                    </div>"""

    def _render_hy3_decoder_hint(self, arch: Architecture) -> str:
        if not self._is_hy3(arch):
            return ""
        dense_prefix = int(arch.parameters.get("dense_prefix_layers", 0) or 0)
        return (
            f'<div class="hy3-subgroup-note">'
            f'Dense prefix ends at block {max(dense_prefix - 1, 0)}. '
            f'All later blocks switch to routed MoE FFN while attention remains GQA.'
            f'</div>'
        )

    def _render_hy3_mlp_label(self, arch: Architecture) -> tuple[str, str]:
        if not self._is_hy3(arch):
            feats = arch.special_features
            act_type = ("SwiGLU" if "SwiGLU" in feats else "GELU" if "GELU" in feats else "ReLU")
            return "MLP", act_type
        num_experts = int(arch.parameters.get("num_experts", 0) or 0)
        top_k = int(arch.parameters.get("top_k_experts", 0) or 0)
        return "MoE FFN", f"Top-{top_k} / {num_experts}"

    def _render_hy3_components_summary(self, arch: Architecture) -> str:
        if not self._is_hy3(arch):
            return ""
        dense_prefix = int(arch.parameters.get("dense_prefix_layers", 0) or 0)
        moe_layers = max(int(arch.total_layers or 0) - dense_prefix, 0)
        mtp_layers = int(arch.parameters.get("mtp_layers", 0) or 0)
        return f"""
            <div class="group-box" style="border-style:dashed; border-color:rgba(123,31,162,.28);">
                <div class="group-label">Hy3 Layer Map</div>
                <div class="hy3-chip-row" style="justify-content:center;">
                    <span class="hy3-chip hy3-chip-dense">Dense Prefix · {dense_prefix} layer</span>
                    <span class="hy3-chip hy3-chip-moe">MoE Blocks · {moe_layers} layers</span>
                    <span class="hy3-chip hy3-chip-mtp">MTP Head · {mtp_layers} layer</span>
                </div>
                <div class="hy3-inline-note">Dense prefix handles the first block, routed experts dominate the remaining decoder stack, and MTP stays outside the core loop.</div>
            </div>

            <div class="spacer-lg"></div>"""
