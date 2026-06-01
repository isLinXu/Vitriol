# """
# HTML Renderer for Neural Network Architecture Visualization
# ============================================================
# Generates publication-quality interactive visualizations with:
# - Multi-level hierarchical layout
# - SVG-based dynamic connections
# - Responsive design with theme support
# - Export capabilities (SVG, PNG, JSON)
# """
"""
html_renderer.py - Optimized HTMLRenderer with multi-theme support and PNG/SVG export.
"""

import json
import os
import tempfile

from html import escape
from pathlib import Path
from typing import Any, Dict
from urllib.parse import unquote

from ..core import Architecture


class HTMLRenderer:
    """
    Renders publication-quality interactive HTML visualization of neural network architectures.

    Features:
    - Multi-column hierarchical layout (Backbone → Layer → Components)
    - IEEE/ACM paper-compliant styling
    - Interactive cross-references with SVG connectors
    - 5 built-in themes: light / dark / nord / solarized / ocean
    - Client-side export: PNG (html2canvas) & SVG (dom-to-image-more)
    - Python-side export: render_png() & render_svg() via Playwright (optional dep)
    - Real-time hover highlight: fog-of-war + animated flow paths + tooltip
    - Responsive design with semantic color coding
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Semantic component color palettes (2 variants: default for light themes,
    # vivid for dark themes – slightly more saturated / luminous).
    # ─────────────────────────────────────────────────────────────────────────
    COLOR_SCHEMES: Dict[str, Dict[str, str]] = {
        "default": {
            "tensor":     "#5c97f5",
            "input":      "#4285f4",
            "layer":      "#e67c73",
            "weight":     "#fbbc04",
            "norm":       "#5f6368",
            "activation": "#34a853",
            "attention":  "#e67c73",
            "mlp":        "#e67c73",
            "residual":   "#9aa0a6",
        },
        "vivid": {
            "tensor":     "#6ba3f7",
            "input":      "#5a95f5",
            "layer":      "#f28b82",
            "weight":     "#fdd663",
            "norm":       "#9aa0a6",
            "activation": "#46c25f",
            "attention":  "#f28b82",
            "mlp":        "#f28b82",
            "residual":   "#8a9099",
        },
    }

    # ─────────────────────────────────────────────────────────────────────────
    # Theme registry – each entry defines CSS background/text/border tokens
    # and which COLOR_SCHEME variant to use for semantic component colors.
    # ─────────────────────────────────────────────────────────────────────────
    THEMES: Dict[str, Dict[str, str]] = {
        "light": {
            "bg_primary":   "#ffffff",
            "bg_secondary": "#fafafa",
            "bg_gradient":  "linear-gradient(to bottom, #ffffff 0%, #fafafa 100%)",
            "text_primary":   "#202124",
            "text_secondary": "#5f6368",
            "text_tertiary":  "#9aa0a6",
            "border_light":  "#e0e0e0",
            "border_medium": "#dadce0",
            "shadow_color":  "rgba(0,0,0,0.1)",
            "group_bg":      "rgba(255,255,255,0.6)",
            "label_bg":      "#ffffff",
            "header_bg":     "#ffffff",
            "residual_copy_bg": "white",
            "color_scheme": "default",
            "display_name": "☀️ Light",
        },
        "dark": {
            "bg_primary":   "#1e1e2e",
            "bg_secondary": "#181825",
            "bg_gradient":  "linear-gradient(to bottom, #1e1e2e 0%, #181825 100%)",
            "text_primary":   "#cdd6f4",
            "text_secondary": "#a6adc8",
            "text_tertiary":  "#6c7086",
            "border_light":  "#313244",
            "border_medium": "#45475a",
            "shadow_color":  "rgba(0,0,0,0.4)",
            "group_bg":      "rgba(30,30,46,0.8)",
            "label_bg":      "#1e1e2e",
            "header_bg":     "#181825",
            "residual_copy_bg": "#1e1e2e",
            "color_scheme": "vivid",
            "display_name": "🌙 Dark",
        },
        "nord": {
            "bg_primary":   "#2e3440",
            "bg_secondary": "#3b4252",
            "bg_gradient":  "linear-gradient(to bottom, #2e3440 0%, #3b4252 100%)",
            "text_primary":   "#eceff4",
            "text_secondary": "#d8dee9",
            "text_tertiary":  "#81a1c1",
            "border_light":  "#4c566a",
            "border_medium": "#5e6779",
            "shadow_color":  "rgba(0,0,0,0.3)",
            "group_bg":      "rgba(46,52,64,0.85)",
            "label_bg":      "#2e3440",
            "header_bg":     "#3b4252",
            "residual_copy_bg": "#2e3440",
            "color_scheme": "vivid",
            "display_name": "❄️ Nord",
        },
        "solarized": {
            "bg_primary":   "#fdf6e3",
            "bg_secondary": "#eee8d5",
            "bg_gradient":  "linear-gradient(to bottom, #fdf6e3 0%, #eee8d5 100%)",
            "text_primary":   "#073642",
            "text_secondary": "#586e75",
            "text_tertiary":  "#93a1a1",
            "border_light":  "#ddd6c1",
            "border_medium": "#c4baaa",
            "shadow_color":  "rgba(0,0,0,0.06)",
            "group_bg":      "rgba(253,246,227,0.85)",
            "label_bg":      "#fdf6e3",
            "header_bg":     "#fdf6e3",
            "residual_copy_bg": "#fdf6e3",
            "color_scheme": "default",
            "display_name": "🌅 Solarized",
        },
        "ocean": {
            "bg_primary":   "#0d1117",
            "bg_secondary": "#161b22",
            "bg_gradient":  "linear-gradient(to bottom, #0d1117 0%, #161b22 100%)",
            "text_primary":   "#c9d1d9",
            "text_secondary": "#8b949e",
            "text_tertiary":  "#6e7681",
            "border_light":  "#21262d",
            "border_medium": "#30363d",
            "shadow_color":  "rgba(0,0,0,0.5)",
            "group_bg":      "rgba(13,17,23,0.8)",
            "label_bg":      "#0d1117",
            "header_bg":     "#161b22",
            "residual_copy_bg": "#0d1117",
            "color_scheme": "vivid",
            "display_name": "🌊 Ocean",
        },
    }

    LAYOUT_CONFIG = {
        "col_widths": [280, 380, 340, 340, 320],   # col1..5; extra cols for enc/cross/dna
        "col_gap": 56,
        "header_height": 60,
        "padding": 40,
    }

    # ─────────────────────────────────────────────────────────────────────────
    # Construction
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, theme: str = "light") -> None:
        """
        Initialize renderer.

        Args:
            theme: One of 'light', 'dark', 'nord', 'solarized', 'ocean'.
        """
        if theme not in self.THEMES:
            available = list(self.THEMES.keys())
            raise ValueError(f"Unknown theme '{theme}'. Available: {available}")
        self.theme = theme
        self._t = self.THEMES[theme]
        self._colors = self.COLOR_SCHEMES[self._t["color_scheme"]]

    # ─────────────────────────────────────────────────────────────────────────
    # Public render API
    # ─────────────────────────────────────────────────────────────────────────

    def render(self, architecture: Architecture, output_path: str) -> None:
        """Write interactive HTML visualization to *output_path*."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.render_to_string(architecture), encoding="utf-8")

    def render_to_string(self, architecture: Architecture) -> str:
        """Return the full HTML string (useful for embedding / testing)."""
        data = self._prepare_data(architecture)
        return self._generate_html(architecture, data)

    @staticmethod
    def _html(value: Any) -> str:
        """Escape untrusted config-derived values before embedding in HTML."""
        return escape(str(value), quote=True)

    @staticmethod
    def _script_json(data: Dict[str, Any]) -> str:
        """Serialize JSON for inline script blocks without allowing </script> breakout."""
        return (
            json.dumps(data, indent=2, default=str, ensure_ascii=False)
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
            .replace("&", "\\u0026")
        )

    def render_png(
        self,
        architecture: Architecture,
        output_path: str,
        viewport_width: int = 1800,
        viewport_height: int = 960,
        device_scale_factor: float = 2.0,
    ) -> None:
        """
        Export visualization as a high-DPI PNG via Playwright headless Chromium.

        Requirements::

            pip install playwright
            playwright install chromium

        Args:
            architecture: Parsed architecture object.
            output_path: Destination .png file path.
            viewport_width: Browser viewport width (px).
            viewport_height: Browser viewport height (px).
            device_scale_factor: Pixel density multiplier (2 = @2x / retina).
        """
        playwright = self._require_playwright()
        html_str = self.render_to_string(architecture)

        with tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, mode="w", encoding="utf-8"
        ) as fh:
            fh.write(html_str)
            tmp = fh.name

        try:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)

            with playwright.sync_playwright() as pw:
                browser = pw.chromium.launch()
                page = browser.new_page(
                    viewport={"width": viewport_width, "height": viewport_height},
                    device_scale_factor=device_scale_factor,
                )
                page.goto(f"file://{os.path.abspath(tmp)}")
                # Wait for fonts and the JS SVG-drawing pass.
                page.wait_for_timeout(900)
                # Expand overflow so full content is captured.
                page.evaluate(
                    """() => {
                        document.body.style.overflow = 'visible';
                        document.body.style.height = 'auto';
                        const mc = document.querySelector('.main-container');
                        if (mc) { mc.style.overflow = 'visible'; mc.style.height = 'auto'; }
                    }"""
                )
                page.wait_for_timeout(200)
                page.screenshot(path=str(out), full_page=True)
                browser.close()
        finally:
            os.unlink(tmp)

    def render_svg(
        self,
        architecture: Architecture,
        output_path: str,
        viewport_width: int = 1800,
        viewport_height: int = 960,
    ) -> None:
        """
        Export visualization as an SVG (with foreignObject) via Playwright +
        dom-to-image-more.

        Requirements::

            pip install playwright
            playwright install chromium

        .. note::
            The resulting SVG uses ``<foreignObject>`` to embed HTML content.
            This is well-supported in modern browsers and Inkscape, but some
            minimal SVG viewers may not render it.

        Args:
            architecture: Parsed architecture object.
            output_path: Destination .svg file path.
            viewport_width: Browser viewport width (px).
            viewport_height: Browser viewport height (px).
        """
        playwright = self._require_playwright()
        html_str = self.render_to_string(architecture)

        with tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, mode="w", encoding="utf-8"
        ) as fh:
            fh.write(html_str)
            tmp = fh.name

        try:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)

            with playwright.sync_playwright() as pw:
                browser = pw.chromium.launch()
                page = browser.new_page(
                    viewport={"width": viewport_width, "height": viewport_height}
                )
                page.goto(f"file://{os.path.abspath(tmp)}")
                page.wait_for_timeout(1000)

                # Inject dom-to-image-more from CDN.
                page.add_script_tag(
                    url="https://cdn.jsdelivr.net/npm/dom-to-image-more@3.4.0/dist/dom-to-image-more.min.js"
                )
                page.wait_for_timeout(600)

                raw = page.evaluate(
                    """async () => {
                        document.body.style.overflow = 'visible';
                        document.body.style.height = 'auto';
                        const mc = document.querySelector('.main-container');
                        if (mc) { mc.style.overflow = 'visible'; mc.style.height = 'auto'; }
                        await new Promise(r => setTimeout(r, 250));
                        return await domtoimage.toSvg(document.body);
                    }"""
                )

                # Decode the data-URI.
                PREFIX_PLAIN  = "data:image/svg+xml;charset=utf-8,"
                PREFIX_BASE64 = "data:image/svg+xml;base64,"
                if raw.startswith(PREFIX_PLAIN):
                    svg_content = unquote(raw[len(PREFIX_PLAIN):])
                elif raw.startswith(PREFIX_BASE64):
                    import base64
                    svg_content = base64.b64decode(raw[len(PREFIX_BASE64):]).decode("utf-8")
                else:
                    svg_content = raw

                out.write_text(svg_content, encoding="utf-8")
                browser.close()
        finally:
            os.unlink(tmp)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _require_playwright():
        """Import playwright or raise a helpful ImportError."""
        try:
            import playwright  # noqa: F401
            return playwright
        except ImportError:
            raise ImportError(
                "playwright is required for PNG/SVG export.\n"
                "Install with:\n"
                "    pip install playwright\n"
                "    playwright install chromium"
            ) from None

    def _prepare_data(self, arch: Architecture) -> Dict[str, Any]:
        """Extract and structure architecture metadata for the JS layer."""
        topology = self._build_topology_data(arch)
        return {
            "model_type":  arch.model_type,
            "arch_type":   arch.arch_type,
            "features":    list(arch.features),
            "parameters": {
                k: v
                for k, v in arch.parameters.items()
                if k in ("hidden_size","num_heads","num_kv_heads","num_layers",
                         "vocab_size","num_experts","top_k_experts","sliding_window")
            },
            "statistics": {
                "total_params":    arch.total_params,
                "memory_fp16_gb":  arch.memory_fp16_gb,
                "total_layers":    arch.total_layers,
                "encoder_layers":  getattr(arch, "encoder_layers", 0),
                "decoder_layers":  getattr(arch, "decoder_layers", 0),
                "head_dim": (
                    arch.parameters.get("hidden_size", 0)
                    // arch.parameters.get("num_heads", 1)
                ),
            },
            "topology":         topology,
            "theme":            self.theme,
            "available_themes": [
                {"id": k, "label": v["display_name"]}
                for k, v in self.THEMES.items()
            ],
        }

    def _build_topology_data(self, arch: Architecture) -> Dict[str, Any]:
        """Generate dynamic connection graph and SVG paths based on architecture type."""
        t = arch.arch_type
        conns = []
        # Base graph with backbone nodes
        graph = {
            'node-text-input':      {'up': [], 'down': ['node-tokenizer'], 'label': 'Text Input', 'desc': 'Raw text / token IDs'},
            'node-tokenizer':       {'up': ['node-text-input'], 'down': ['node-embedding'], 'label': 'Tokenizer', 'desc': 'BPE subword encoding'},
            'node-embedding':       {'up': ['node-tokenizer'], 'down': ['node-hidden-entry'], 'label': 'Embedding', 'desc': 'Token → dense vector'},
            'node-hidden-entry':    {'up': ['node-embedding'], 'down': ['layers-container'], 'label': 'Hidden States', 'desc': '[B, L, D] initial repr.'},
            'layers-container':     {'up': ['node-hidden-entry'], 'down': ['node-rms-final'], 'label': 'Layers', 'desc': f'Stacked Blocks (x{arch.total_layers})'},
            'node-rms-final':       {'up': ['layers-container'], 'down': ['node-linear-logits'], 'label': 'Final Norm', 'desc': 'Pre-output normalisation'},
            'node-linear-logits':   {'up': ['node-rms-final'], 'down': ['node-loss'], 'label': 'Linear (Logits)', 'desc': 'Project to vocab size'},
            'node-loss':            {'up': ['node-linear-logits'], 'down': ['node-output'], 'label': 'Loss', 'desc': 'Cross-entropy objective'},
            'node-output':          {'up': ['node-loss'], 'down': [], 'label': 'Output', 'desc': 'Final token distribution'},
        }

        if t == "encoder-only":
            # ── Encoder-Only Topology ──
            conns = [
                {'from': 'layers-container', 'to': 'encoder-layer-detail', 'curve': 0.6, 'type': 'expansion'},
                {'from': 'enc-attn', 'to': 'enc-attn-detail', 'curve': 0.5, 'type': 'expansion'}
            ]
            graph['layers-container']['down'].append('encoder-layer-detail')

            graph.update({
                'encoder-layer-detail': {'up': ['layers-container'], 'down': ['enc-input'], 'label': 'Encoder Block', 'desc': 'Bidirectional Transformer Layer'},
                'enc-input':            {'up': ['encoder-layer-detail'], 'down': ['enc-norm-attn'], 'label': 'Input', 'desc': 'Layer Input'},
                'enc-norm-attn':        {'up': ['enc-input'], 'down': ['enc-attn'], 'label': 'Norm', 'desc': 'Pre-Attention Norm'},
                'enc-attn':             {'up': ['enc-norm-attn'], 'down': ['enc-post-attn', 'enc-attn-detail'], 'label': 'Self-Attention', 'desc': 'Bidirectional Mixing'},
                'enc-post-attn':        {'up': ['enc-attn'], 'down': ['enc-add-attn'], 'label': 'Attn Out', 'desc': 'Post-Attn Projection'},
                'enc-add-attn':         {'up': ['enc-post-attn'], 'down': ['enc-hidden-mid'], 'label': 'Add', 'desc': 'Residual Connection'},
                'enc-hidden-mid':       {'up': ['enc-add-attn'], 'down': ['enc-norm-ffn'], 'label': 'Mid State', 'desc': 'Post-Add State'},
                'enc-norm-ffn':         {'up': ['enc-hidden-mid'], 'down': ['enc-ffn'], 'label': 'Norm', 'desc': 'Pre-FFN Norm'},
                'enc-ffn':              {'up': ['enc-norm-ffn'], 'down': ['enc-post-ffn'], 'label': 'FFN', 'desc': 'Feed-Forward Network'},
                'enc-post-ffn':         {'up': ['enc-ffn'], 'down': ['enc-add-ffn'], 'label': 'FFN Out', 'desc': 'Post-FFN Projection'},
                'enc-add-ffn':          {'up': ['enc-post-ffn'], 'down': ['enc-output'], 'label': 'Add', 'desc': 'Residual Connection'},
                'enc-output':           {'up': ['enc-add-ffn'], 'down': [], 'label': 'Output', 'desc': 'Layer Output'},
                'enc-attn-detail':      {'up': ['enc-attn'], 'down': [], 'label': 'Attn Detail', 'desc': 'Detailed Attention View'}
            })

        elif t == "encoder-decoder":
            # ── Encoder-Decoder Topology ──
            conns = [
                {'from': 'layers-container', 'to': 'encoder-layer-detail', 'curve': 0.6, 'type': 'expansion'},
                {'from': 'enc-output', 'to': 'cross-attn-detail', 'curve': 0.4, 'type': 'flow'}, # Enc out -> Cross Attn
                {'from': 'decoder-layer-detail', 'to': 'cross-attn-detail', 'curve': 0.4, 'type': 'flow'} # Dec layer -> Cross Attn
            ]
            graph['layers-container']['down'].append('encoder-layer-detail')

            # Encoder nodes (reuse logic roughly)
            graph.update({
                'encoder-layer-detail': {'up': ['layers-container'], 'down': ['enc-input'], 'label': 'Encoder Stack', 'desc': 'Full Encoder'},
                'enc-input': {'up': ['encoder-layer-detail'], 'down': ['enc-attn'], 'label': 'Enc Input', 'desc': ''},
                'enc-attn': {'up': ['enc-input'], 'down': ['enc-output'], 'label': 'Enc Self-Attn', 'desc': ''},
                'enc-output': {'up': ['enc-attn'], 'down': ['cross-attn-detail'], 'label': 'Enc Output', 'desc': 'Feeds into Decoder'},
            })
            # Decoder nodes
            graph.update({
                'decoder-layer-detail': {'up': [], 'down': ['decoder-input'], 'label': 'Decoder Stack', 'desc': 'Full Decoder'},
                'decoder-input': {'up': ['decoder-layer-detail'], 'down': ['attention-module'], 'label': 'Dec Input', 'desc': ''},
                'attention-module': {'up': ['decoder-input'], 'down': ['cross-attn-detail'], 'label': 'Dec Self-Attn', 'desc': 'Causal'},
                'cross-attn-detail': {'up': ['attention-module', 'enc-output'], 'down': ['mlp-module'], 'label': 'Cross-Attn', 'desc': 'Encoder-Decoder Attention'},
                'mlp-module': {'up': ['cross-attn-detail'], 'down': ['decoder-output'], 'label': 'MLP', 'desc': ''},
                'decoder-output': {'up': ['mlp-module'], 'down': [], 'label': 'Dec Output', 'desc': ''},
            })

        else:
            # ── Decoder-Only Topology (Default) ──
            conns = [
                {'from': 'layers-container', 'to': 'decoder-layer-detail', 'curve': 0.6, 'type': 'expansion'},
                {'from': 'attention-module', 'to': 'attention-detail', 'curve': 0.5, 'type': 'expansion'},
                {'from': 'mlp-module', 'to': 'mlp-detail', 'curve': 0.5, 'type': 'expansion'}
            ]
            graph['layers-container']['down'].append('decoder-layer-detail')

            graph.update({
                'decoder-layer-detail': {'up': ['layers-container'], 'down': ['decoder-input'], 'label': 'Transformer Block', 'desc': 'Zoomed-in layer view'},
                'decoder-input':        {'up': ['decoder-layer-detail'], 'down': ['node-rms-attn'], 'label': 'Hidden States (in)', 'desc': 'Layer input [B, L, D]'},
                'node-rms-attn':        {'up': ['decoder-input'], 'down': ['attention-module'], 'label': 'Norm (Pre-Attn)', 'desc': 'Pre-attention norm'},
                'attention-module':     {'up': ['node-rms-attn'], 'down': ['node-hidden-post-attn','attention-detail'], 'label': 'Self-Attention', 'desc': 'Token mixing'},
                'node-hidden-post-attn':{'up': ['attention-module'], 'down': ['node-add-attn'], 'label': 'Attn Output', 'desc': 'Post-attention tensor'},
                'node-add-attn':        {'up': ['node-hidden-post-attn'], 'down': ['node-hidden-mid'], 'label': '⊕ Residual', 'desc': 'Skip connection'},
                'node-hidden-mid':      {'up': ['node-add-attn'], 'down': ['node-rms-mlp'], 'label': 'Hidden States (mid)', 'desc': 'Between Attn and MLP'},
                'node-rms-mlp':         {'up': ['node-hidden-mid'], 'down': ['mlp-module'], 'label': 'Norm (Pre-MLP)', 'desc': 'Pre-MLP norm'},
                'mlp-module':           {'up': ['node-rms-mlp'], 'down': ['node-hidden-post-mlp','mlp-detail'], 'label': 'MLP', 'desc': 'Channel mixing'},
                'node-hidden-post-mlp': {'up': ['mlp-module'], 'down': ['node-add-mlp'], 'label': 'MLP Output', 'desc': 'Post-MLP tensor'},
                'node-add-mlp':         {'up': ['node-hidden-post-mlp'], 'down': ['decoder-output'], 'label': '⊕ Residual', 'desc': 'Skip connection'},
                'decoder-output':       {'up': ['node-add-mlp'], 'down': [], 'label': 'Hidden States (out)', 'desc': 'Layer output'},
                'attention-detail':     {'up': ['attention-module'], 'down': [], 'label': 'Attn Detail', 'desc': 'Detailed view'},
                'mlp-detail':           {'up': ['mlp-module'], 'down': [], 'label': 'MLP Detail', 'desc': 'Detailed view'}
            })

        return {"connections": conns, "graph": graph}

    # ── HTML generation ───────────────────────────────────────────────────────

    def _generate_html(self, arch: Architecture, data: Dict) -> str:
        model_title = self._html(arch.model_type)
        theme = self._html(self.theme)
        return f"""<!DOCTYPE html>
<html lang="en" data-theme="{theme}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Architecture Visualization: {model_title}</title>
    {self._get_fonts()}
    {self._get_styles()}
</head>
<body>
    {self._render_header(arch, data["statistics"])}
    {self._render_hy3_overview(arch)}
    {self._render_deepseek_v4_overview(arch)}
    {self._render_main_content(arch)}
    {self._render_export_libs()}
    {self._render_scripts(data)}
</body>
</html>"""

    # ── Fonts ─────────────────────────────────────────────────────────────────

    def _get_fonts(self) -> str:
        return """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">"""

    # ── Styles ────────────────────────────────────────────────────────────────

    def _get_styles(self) -> str:
        layout = self.LAYOUT_CONFIG
        t = self._t                     # active theme dict
        colors = self._colors           # active color scheme dict

        # ── Build per-theme CSS variable blocks (enables JS theme switching) ──
        per_theme_blocks = []
        for theme_id, td in self.THEMES.items():
            cs = self.COLOR_SCHEMES[td["color_scheme"]]
            per_theme_blocks.append(f"""
        html[data-theme="{theme_id}"] {{
            --bg-primary:   {td['bg_primary']};
            --bg-secondary: {td['bg_secondary']};
            --bg-gradient:  {td['bg_gradient']};
            --text-primary:   {td['text_primary']};
            --text-secondary: {td['text_secondary']};
            --text-tertiary:  {td['text_tertiary']};
            --border-light:  {td['border_light']};
            --border-medium: {td['border_medium']};
            --shadow-color:  {td['shadow_color']};
            --group-bg:      {td['group_bg']};
            --label-bg:      {td['label_bg']};
            --header-bg:     {td['header_bg']};
            --residual-copy-bg: {td['residual_copy_bg']};
            --color-tensor:     {cs['tensor']};
            --color-input:      {cs['input']};
            --color-layer:      {cs['layer']};
            --color-weight:     {cs['weight']};
            --color-norm:       {cs['norm']};
            --color-activation: {cs['activation']};
            --color-attention:  {cs['attention']};
            --color-residual:   {cs['residual']};
        }}""")
        all_theme_css = "\n".join(per_theme_blocks)

        return f"""
    <style>
        /* ═══════════════════════════════════════════════════
           1. PER-THEME CSS VARIABLES
           (one [data-theme] block per registered theme)
        ═══════════════════════════════════════════════════ */
        {all_theme_css}

        /* ═══════════════════════════════════════════════════
           2. INITIAL / FALLBACK VALUES  (matches Python-chosen theme)
           Applied on <html> so they work before JS runs.
        ═══════════════════════════════════════════════════ */
        html {{
            --bg-primary:   {t['bg_primary']};
            --bg-secondary: {t['bg_secondary']};
            --bg-gradient:  {t['bg_gradient']};
            --text-primary:   {t['text_primary']};
            --text-secondary: {t['text_secondary']};
            --text-tertiary:  {t['text_tertiary']};
            --border-light:  {t['border_light']};
            --border-medium: {t['border_medium']};
            --shadow-color:  {t['shadow_color']};
            --group-bg:      {t['group_bg']};
            --label-bg:      {t['label_bg']};
            --header-bg:     {t['header_bg']};
            --residual-copy-bg: {t['residual_copy_bg']};
            --color-tensor:     {colors['tensor']};
            --color-input:      {colors['input']};
            --color-layer:      {colors['layer']};
            --color-weight:     {colors['weight']};
            --color-norm:       {colors['norm']};
            --color-activation: {colors['activation']};
            --color-attention:  {colors['attention']};
            --color-residual:   {colors['residual']};
        }}

        /* ═══════════════════════════════════════════════════
           3. LAYOUT CONSTANTS  (theme-independent)
        ═══════════════════════════════════════════════════ */
        :root {{
            --col-1-width: {layout['col_widths'][0]}px;
            --col-2-width: {layout['col_widths'][1]}px;
            --col-3-width: {layout['col_widths'][2]}px;
            --col-gap: {layout['col_gap']}px;
            --header-height: {layout['header_height']}px;
            --container-padding: {layout['padding']}px;
            --font-primary: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-mono: 'JetBrains Mono', 'Courier New', monospace;
            --shadow-sm: 0 1px 2px var(--shadow-color);
            --shadow-md: 0 2px 8px var(--shadow-color);
            --shadow-lg: 0 4px 16px var(--shadow-color);
            --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
            --transition-base: 250ms cubic-bezier(0.4, 0, 0.2, 1);
        }}

        /* ═══════════════════════════════════════════════════
           4. BASE RESET & BODY
        ═══════════════════════════════════════════════════ */
        *, *::before, *::after {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: var(--font-primary);
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            height: 100vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            transition:
                background-color var(--transition-base),
                color var(--transition-base);
        }}

        /* ═══════════════════════════════════════════════════
           5. HEADER
        ═══════════════════════════════════════════════════ */
        header {{
            flex-shrink: 0;
            height: var(--header-height);
            border-bottom: 1px solid var(--border-light);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 20px;
            background: var(--header-bg);
            box-shadow: var(--shadow-sm);
            z-index: 100;
            gap: 16px;
            transition: background-color var(--transition-base),
                        border-color var(--transition-base);
        }}

        .header-left  {{ display: flex; align-items: center; gap: 12px; flex-shrink: 0; }}
        .header-center {{ display: flex; align-items: center; gap: 24px; flex: 1; justify-content: center; }}
        .header-right {{ display: flex; align-items: center; gap: 10px; flex-shrink: 0; }}

        .logo {{
            display: flex; align-items: center; gap: 10px;
            font-weight: 600; font-size: 17px; color: var(--text-primary);
        }}
        .logo-icon {{
            width: 30px; height: 30px;
            background: linear-gradient(135deg, var(--color-tensor), var(--color-attention));
            border-radius: 6px;
            display: flex; align-items: center; justify-content: center;
            color: white; font-weight: 700; font-size: 15px;
        }}
        .model-badge {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-family: var(--font-mono);
            font-weight: 500;
        }}

        /* ── Stats bar ── */
        .stats-bar {{ display: flex; gap: 20px; font-size: 12px; color: var(--text-secondary); }}
        .stat-item {{ display: flex; align-items: center; gap: 5px; }}
        .stat-label {{ font-weight: 500; color: var(--text-tertiary); }}
        .stat-value {{ font-family: var(--font-mono); font-weight: 600; color: var(--text-primary); }}

        /* ── Theme switcher ── */
        .theme-switcher {{
            display: flex; align-items: center; gap: 6px;
        }}
        .theme-switcher-label {{
            font-size: 11px; font-weight: 600; color: var(--text-tertiary);
            text-transform: uppercase; letter-spacing: 0.8px;
            white-space: nowrap;
        }}
        .theme-btn {{
            padding: 4px 8px;
            border-radius: 5px;
            border: 1px solid var(--border-medium);
            background: transparent;
            color: var(--text-secondary);
            font-size: 11px;
            font-family: var(--font-primary);
            cursor: pointer;
            transition: all var(--transition-fast);
            white-space: nowrap;
        }}
        .theme-btn:hover {{
            background: var(--border-light);
            color: var(--text-primary);
        }}
        .theme-btn.active {{
            background: var(--color-tensor);
            color: white;
            border-color: transparent;
        }}

        /* ── Export buttons ── */
        .export-group {{ display: flex; align-items: center; gap: 6px; }}
        .export-label {{
            font-size: 11px; font-weight: 600; color: var(--text-tertiary);
            text-transform: uppercase; letter-spacing: 0.8px;
            white-space: nowrap;
        }}
        .export-btn {{
            display: inline-flex; align-items: center; gap: 5px;
            padding: 5px 12px;
            border-radius: 6px;
            border: 1px solid var(--border-medium);
            background: transparent;
            color: var(--text-secondary);
            font-size: 12px;
            font-family: var(--font-primary);
            font-weight: 500;
            cursor: pointer;
            transition: all var(--transition-fast);
            white-space: nowrap;
        }}
        .export-btn:hover {{
            background: var(--color-tensor);
            color: white;
            border-color: transparent;
            box-shadow: var(--shadow-sm);
        }}
        .export-btn:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}
        .export-btn.loading::after {{
            content: '';
            display: inline-block;
            width: 10px; height: 10px;
            border: 2px solid currentColor;
            border-top-color: transparent;
            border-radius: 50%;
            animation: spin 0.6s linear infinite;
            margin-left: 4px;
        }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
        .export-btn.success {{
            background: var(--color-activation);
            color: white;
            border-color: transparent;
        }}

        /* Toast notification */
        #export-toast {{
            position: fixed; bottom: 24px; right: 24px;
            padding: 10px 18px;
            background: var(--text-primary);
            color: var(--bg-primary);
            border-radius: 8px;
            font-size: 13px;
            font-weight: 500;
            box-shadow: var(--shadow-lg);
            opacity: 0;
            transform: translateY(12px);
            transition: opacity var(--transition-base), transform var(--transition-base);
            pointer-events: none;
            z-index: 9999;
        }}
        #export-toast.show {{
            opacity: 1; transform: translateY(0);
        }}

        /* ═══════════════════════════════════════════════════
           6. MAIN LAYOUT
        ═══════════════════════════════════════════════════ */
        .main-container {{
            flex: 1;
            display: flex;
            overflow: auto;
            padding: var(--container-padding);
            gap: var(--col-gap);
            position: relative;
            background: var(--bg-gradient);
        }}

        /* ── Columns ── */
        .column {{
            display: flex; flex-direction: column; align-items: center;
            position: relative; flex-shrink: 0;
        }}
        .col-1 {{ width: var(--col-1-width); }}
        .col-2 {{ width: var(--col-2-width); }}
        .col-3 {{ width: var(--col-3-width); }}

        .column-header {{ width: 100%; margin-bottom: 32px; text-align: center; }}
        .column-title {{
            font-size: 11px; font-weight: 700; color: var(--text-tertiary);
            text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 8px;
        }}
        .column-subtitle {{ font-size: 13px; color: var(--text-secondary); font-weight: 400; }}

        /* ═══════════════════════════════════════════════════
           7. COMPONENT BOXES
        ═══════════════════════════════════════════════════ */
        .box {{
            padding: 12px 20px;
            border-radius: 8px;
            font-weight: 500; font-size: 14px; text-align: center;
            min-width: 140px;
            position: relative;
            box-shadow: var(--shadow-md);
            margin-bottom: 16px;
            transition: transform var(--transition-base), box-shadow var(--transition-base);
            z-index: 10;
            border: 1px solid rgba(0, 0, 0, 0.08);
        }}
        .box-label    {{ display: block; font-weight: 600; line-height: 1.4; }}
        .box-sublabel {{ display: block; font-size: 11px; opacity: 0.85; margin-top: 4px; font-weight: 400; }}

        /* Semantic color variants */
        .box-tensor     {{ background: linear-gradient(135deg, var(--color-tensor) 0%, #4a7fd9 100%); color: white; }}
        .box-input      {{ background: linear-gradient(135deg, var(--color-input)  0%, #1967d2 100%); color: white; }}
        .box-layer      {{ background: linear-gradient(135deg, var(--color-layer)  0%, #d96459 100%); color: white; }}
        .box-weight     {{ background: linear-gradient(135deg, var(--color-weight) 0%, #f9ab00 100%); color: #3c4043; }}
        .box-norm       {{ background: linear-gradient(135deg, var(--color-norm)   0%, #4a4f54 100%); color: white; }}
        .box-activation {{ background: linear-gradient(135deg, var(--color-activation) 0%, #2d8f3f 100%); color: white; }}
        .box-attention  {{
            background: linear-gradient(135deg, #fceceb 0%, #fff3e0 100%);
            color: #d93025;
            border: 1px solid #f28b82;
        }}
        .box-mlp        {{
            background: linear-gradient(135deg, #fceceb 0%, #fff3e0 100%);
            color: #d93025;
            border: 1px solid #f28b82;
        }}
        /* Dark themes: invert attention/mlp box colours */
        html[data-theme="dark"]      .box-attention,
        html[data-theme="dark"]      .box-mlp,
        html[data-theme="nord"]      .box-attention,
        html[data-theme="nord"]      .box-mlp,
        html[data-theme="ocean"]     .box-attention,
        html[data-theme="ocean"]     .box-mlp {{
            background: linear-gradient(135deg, rgba(242,139,130,0.15) 0%, rgba(253,214,99,0.1) 100%);
            color: var(--color-layer);
            border: 1px solid rgba(242,139,130,0.4);
        }}

        /* Operator / math circles */
        .box-operator {{
            background: var(--color-tensor);
            color: white; border: none;
            width: 36px; height: 36px; min-width: unset;
            padding: 0; display: flex; align-items: center; justify-content: center;
            border-radius: 50%;
            font-weight: 700; font-size: 20px;
            box-shadow: var(--shadow-md);
        }}
        .box-operation {{
            background: rgba(92, 151, 245, 0.1);
            border: 1.5px solid var(--color-tensor);
            color: var(--color-tensor);
            font-size: 12px; font-weight: 600;
            padding: 8px 14px;
        }}

        /* ═══════════════════════════════════════════════════
           8. GROUP CONTAINERS
        ═══════════════════════════════════════════════════ */
        .group-box {{
            border: 2px dashed var(--border-medium);
            border-radius: 12px;
            padding: 24px 20px;
            position: relative;
            background: var(--group-bg);
            backdrop-filter: blur(10px);
            margin-bottom: 24px;
            width: 100%;
            display: flex; flex-direction: column; align-items: center;
            box-shadow: var(--shadow-sm);
        }}
        .group-label {{
            position: absolute; top: -12px; left: 16px;
            background: var(--label-bg);
            padding: 4px 12px;
            font-size: 12px; font-weight: 600;
            color: var(--text-secondary);
            border-radius: 4px; border: 1px solid var(--border-light);
            letter-spacing: 0.3px;
            transition: background-color var(--transition-base);
        }}
        .group-layer     {{ border-color: var(--color-layer); }}
        .group-attention {{ border-color: var(--color-layer); }}
        .group-mlp       {{ border-color: var(--color-layer); }}

        /* ═══════════════════════════════════════════════════
           9. CONNECTORS
        ═══════════════════════════════════════════════════ */
        .connector-vertical {{
            width: 2px; height: 24px;
            background: var(--text-tertiary);
            margin: -8px auto 8px;
            position: relative; z-index: 1;
        }}
        .connector-arrow::after {{
            content: '';
            position: absolute; bottom: 0; left: 50%;
            transform: translateX(-50%);
            width: 0; height: 0;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 6px solid var(--text-tertiary);
        }}

        /* SVG overlay */
        #svg-connections {{
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            pointer-events: none; z-index: 1;
        }}
        .svg-connector {{
            fill: none;
            stroke: var(--text-tertiary);
            stroke-width: 2;
            stroke-dasharray: 6 4;
            opacity: 0.6;
            transition: stroke var(--transition-base),
                        stroke-width var(--transition-base),
                        opacity var(--transition-base);
        }}
        .svg-connector.highlighted {{
            stroke: var(--color-tensor);
            stroke-width: 2.5;
            opacity: 1;
        }}

        /* ═══════════════════════════════════════════════════
           10. LAYOUT UTILITIES
        ═══════════════════════════════════════════════════ */
        .flex-row    {{ display: flex; gap: 12px; justify-content: center; width: 100%; flex-wrap: wrap; }}
        .flex-column {{ display: flex; flex-direction: column; align-items: center; flex: 1; width: 100%; }}
        .spacer      {{ height: 16px; }}
        .spacer-lg   {{ height: 32px; }}

        /* ═══════════════════════════════════════════════════
           11. INTERACTIVE STATES
        ═══════════════════════════════════════════════════ */
        .interactive {{ cursor: pointer; }}
        .interactive:hover {{
            transform: translateY(-2px) scale(1.02);
            box-shadow: var(--shadow-lg);
            z-index: 20;
        }}
        .interactive:active {{ transform: translateY(0) scale(0.98); }}

        .highlight {{ animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }}
        @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.7; }} }}

        /* Tooltips */
        .tooltip {{ position: relative; }}
        .tooltip::before {{
            content: attr(data-tooltip);
            position: absolute; bottom: 100%; left: 50%;
            transform: translateX(-50%) translateY(-8px);
            background: rgba(0, 0, 0, 0.88);
            color: white; padding: 6px 12px; border-radius: 6px;
            font-size: 12px; white-space: nowrap;
            opacity: 0; pointer-events: none;
            transition: opacity var(--transition-fast); z-index: 1000;
        }}
        .tooltip:hover::before {{ opacity: 1; }}

        /* ═══════════════════════════════════════════════════
           12. RESPONSIVE
        ═══════════════════════════════════════════════════ */
        @media (max-width: 1400px) {{
            :root {{ --col-gap: 40px; --container-padding: 24px; }}
        }}
        @media (max-width: 1200px) {{
            .main-container {{ flex-direction: column; align-items: center; }}
            .column {{ width: 100% !important; max-width: 600px; margin-bottom: 40px; }}
            .header-center {{ display: none; }}
        }}

        /* ═══════════════════════════════════════════════════
           13. PRINT
        ═══════════════════════════════════════════════════ */
        @media print {{
            body {{ overflow: visible; height: auto; }}
            .main-container {{ overflow: visible; }}
            .interactive:hover {{ transform: none; box-shadow: var(--shadow-md); }}
            .export-group, .theme-switcher {{ display: none !important; }}
        }}

        /* ═══════════════════════════════════════════════════
           14. HOVER HIGHLIGHT SYSTEM
           ─ Fog-of-war dims non-related elements;
             node-self / node-upstream / node-downstream
             classes override the dimming.
           ─ Animated SVG paths show direction of data flow.
           ─ Floating tooltip shows connection metadata.
        ═══════════════════════════════════════════════════ */

        /* Fog-of-war: ALL boxes & groups dim unless highlighted */
        .main-container.is-highlighting
            .box:not(.node-self):not(.node-upstream):not(.node-downstream) {{
            opacity: 0.11;
            filter: grayscale(65%) brightness(0.75);
            pointer-events: none;
            transition: opacity 200ms ease, filter 200ms ease;
        }}
        .main-container.is-highlighting
            .group-box:not(.node-self):not(.node-upstream):not(.node-downstream) {{
            opacity: 0.14;
            filter: grayscale(50%) brightness(0.80);
            transition: opacity 200ms ease, filter 200ms ease;
        }}
        .main-container.is-highlighting .connector-vertical {{
            opacity: 0.05;
            transition: opacity 200ms ease;
        }}

        /* ── Highlighted node states ── */
        .node-self {{
            opacity: 1 !important; filter: none !important;
            outline: 3px solid #ff6b6b !important;
            outline-offset: 3px;
            box-shadow: 0 0 0 4px rgba(255,107,107,0.18),
                        0 0 24px rgba(255,107,107,0.45),
                        var(--shadow-lg) !important;
            transform: translateY(-3px) scale(1.06) !important;
            z-index: 50 !important;
        }}
        .node-upstream {{
            opacity: 1 !important; filter: none !important;
            outline: 2px solid var(--color-layer) !important;
            outline-offset: 2px;
            box-shadow: 0 0 0 3px rgba(230,124,115,0.18),
                        0 0 18px rgba(230,124,115,0.38),
                        var(--shadow-md) !important;
            transform: translateY(-2px) scale(1.03) !important;
            z-index: 30 !important;
        }}
        .node-downstream {{
            opacity: 1 !important; filter: none !important;
            outline: 2px solid var(--color-tensor) !important;
            outline-offset: 2px;
            box-shadow: 0 0 0 3px rgba(92,151,245,0.18),
                        0 0 18px rgba(92,151,245,0.38),
                        var(--shadow-md) !important;
            transform: translateY(-2px) scale(1.03) !important;
            z-index: 30 !important;
        }}

        /* ── SVG cross-column path highlighting ── */
        .svg-connector-up {{
            stroke: var(--color-layer) !important;
            stroke-width: 3 !important;
            stroke-dasharray: 9 4 !important;
            opacity: 1 !important;
            animation: flow-march-up 0.48s linear infinite;
        }}
        .svg-connector-down {{
            stroke: var(--color-tensor) !important;
            stroke-width: 3 !important;
            stroke-dasharray: 9 4 !important;
            opacity: 1 !important;
            animation: flow-march-down 0.48s linear infinite;
        }}

        /* ── Temporary within-column animated paths ── */
        .svg-temp-up {{
            fill: none;
            stroke: var(--color-layer);
            stroke-width: 2.5;
            stroke-dasharray: 7 3;
            opacity: 0.90;
            animation: flow-march-up 0.42s linear infinite;
        }}
        .svg-temp-down {{
            fill: none;
            stroke: var(--color-tensor);
            stroke-width: 2.5;
            stroke-dasharray: 7 3;
            opacity: 0.90;
            animation: flow-march-down 0.42s linear infinite;
        }}

        /* marching-ant direction: up → negative offset drift, down → positive */
        @keyframes flow-march-up   {{ to {{ stroke-dashoffset: -20; }} }}
        @keyframes flow-march-down {{ to {{ stroke-dashoffset:  20; }} }}

        /* ── Hover tooltip ── */
        .hover-tooltip {{
            position: fixed;
            background: var(--text-primary);
            color: var(--bg-primary);
            border-radius: 9px;
            padding: 9px 14px 10px;
            font-size: 12px;
            font-family: var(--font-primary);
            pointer-events: none;
            z-index: 9100;
            box-shadow: 0 6px 24px rgba(0,0,0,0.3);
            min-width: 155px;
            max-width: 220px;
            opacity: 0;
            transform: translateY(6px);
            transition: opacity 130ms ease, transform 130ms ease;
        }}
        .hover-tooltip.visible {{
            opacity: 1;
            transform: translateY(0);
        }}
        .ht-title {{
            font-weight: 700; font-size: 13px;
            margin-bottom: 5px;
            padding-bottom: 5px;
            border-bottom: 1px solid rgba(128,128,128,0.25);
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }}
        .ht-stats {{
            display: flex; gap: 10px;
            font-size: 11px; font-weight: 500;
            margin-bottom: 3px;
        }}
        .ht-upstream   {{ color: #f28b82; }}
        .ht-downstream {{ color: #6ba3f7; }}
        .ht-desc {{
            font-size: 10px; opacity: 0.68;
            font-style: italic; line-height: 1.4;
            margin-top: 4px;
        }}
        .ht-total {{
            font-size: 10px; opacity: 0.55;
            margin-top: 2px;
        }}

        /* ═══════════════════════════════════════════════════
           15. FEATURE BADGES  (arch-specific capability tags)
        ═══════════════════════════════════════════════════ */
        .feat-badges {{
            display: flex; flex-wrap: wrap; gap: 4px;
            justify-content: center;
            margin-top: 5px;
        }}
        .feat-badge {{
            font-size: 9.5px; font-weight: 700;
            padding: 2px 6px; border-radius: 3px;
            letter-spacing: 0.4px; line-height: 1.4;
            text-transform: uppercase;
            opacity: 0.92;
        }}
        /* Badge colour palette */
        .badge-attn      {{ background:#e8f0fe; color:#1967d2; }}
        .badge-norm      {{ background:#fce8e6; color:#c5221f; }}
        .badge-act       {{ background:#e6f4ea; color:#137333; }}
        .badge-pos       {{ background:#fff3cd; color:#856404; }}
        .badge-special   {{ background:#f4eaff; color:#7b1fa2; }}
        .badge-causal    {{ background:#fff8e1; color:#e65100; border:1px solid #ffcc80; }}
        .badge-bidir     {{ background:#e8f5e9; color:#1b5e20; border:1px solid #a5d6a7; }}
        /* dark-theme overrides */
        html[data-theme="dark"]    .badge-attn,
        html[data-theme="nord"]    .badge-attn,
        html[data-theme="ocean"]   .badge-attn  {{ background:rgba(25,103,210,.22); color:#6ba3f7; }}
        html[data-theme="dark"]    .badge-norm,
        html[data-theme="nord"]    .badge-norm,
        html[data-theme="ocean"]   .badge-norm  {{ background:rgba(197,34,31,.22);  color:#f28b82; }}
        html[data-theme="dark"]    .badge-act,
        html[data-theme="nord"]    .badge-act,
        html[data-theme="ocean"]   .badge-act   {{ background:rgba(19,115,51,.22);  color:#46c25f; }}
        html[data-theme="dark"]    .badge-pos,
        html[data-theme="nord"]    .badge-pos,
        html[data-theme="ocean"]   .badge-pos   {{ background:rgba(133,100,4,.22);  color:#fdd663; }}
        html[data-theme="dark"]    .badge-special,
        html[data-theme="nord"]    .badge-special,
        html[data-theme="ocean"]   .badge-special {{ background:rgba(123,31,162,.22);color:#d1b3f5; }}
        html[data-theme="dark"]    .badge-causal,
        html[data-theme="nord"]    .badge-causal,
        html[data-theme="ocean"]   .badge-causal  {{ background:rgba(230,81,0,.18); color:#ffb74d; }}
        html[data-theme="dark"]    .badge-bidir,
        html[data-theme="nord"]    .badge-bidir,
        html[data-theme="ocean"]   .badge-bidir   {{ background:rgba(27,94,32,.22); color:#69d97e; }}

        /* ═══════════════════════════════════════════════════
           16. ARCH-TYPE IDENTITY STRIP  (top of col-2)
        ═══════════════════════════════════════════════════ */
        .arch-type-strip {{
            width: 100%; text-align: center;
            padding: 6px 12px; border-radius: 6px;
            font-size: 11px; font-weight: 700;
            letter-spacing: 0.6px; text-transform: uppercase;
            margin-bottom: 20px;
        }}
        .arch-strip-decoder  {{ background:rgba(230,124,115,.14); color:var(--color-layer);
                                 border:1.5px solid rgba(230,124,115,.35); }}
        .arch-strip-encoder  {{ background:rgba(92,151,245,.14);  color:var(--color-tensor);
                                 border:1.5px solid rgba(92,151,245,.35); }}
        .arch-strip-encdec   {{ background:rgba(52,168,83,.14);  color:var(--color-activation);
                                 border:1.5px solid rgba(52,168,83,.35); }}

        /* ═══════════════════════════════════════════════════
           17. CROSS-ATTENTION COLUMN  (encoder-decoder only)
        ═══════════════════════════════════════════════════ */
        .box-cross-attn {{
            background: linear-gradient(135deg,rgba(52,168,83,.85) 0%,rgba(19,115,51,.9) 100%);
            color: white;
        }}
        .group-cross {{ border-color: var(--color-activation); }}

        /* ═══════════════════════════════════════════════════
           18. ARCHITECTURE DNA COLUMN  (rightmost comparison)
        ═══════════════════════════════════════════════════ */
        .col-dna {{ width: 320px; }}
        .dna-card {{
            background: var(--group-bg);
            border: 1.5px solid var(--border-medium);
            border-radius: 10px;
            padding: 16px 18px;
            margin-bottom: 16px;
            width: 100%;
            box-shadow: var(--shadow-sm);
        }}
        .dna-card-title {{
            font-size: 10px; font-weight: 700;
            text-transform: uppercase; letter-spacing: 0.8px;
            color: var(--text-tertiary); margin-bottom: 10px;
            padding-bottom: 6px;
            border-bottom: 1px solid var(--border-light);
        }}
        .dna-row {{
            display: flex; justify-content: space-between; align-items: center;
            font-size: 11.5px; padding: 5px 0;
            border-bottom: 1px solid var(--border-light);
        }}
        .dna-row:last-child {{ border-bottom: none; }}
        .dna-key {{
            color: var(--text-secondary); font-weight: 500; flex: 1;
        }}
        .dna-val {{
            font-family: var(--font-mono); font-weight: 600;
            color: var(--text-primary); font-size: 11px;
            text-align: right; flex: 1;
        }}
        .dna-val.highlight {{ color: var(--color-tensor); }}
        .dna-val.warn       {{ color: var(--color-layer);  }}
        .dna-val.ok         {{ color: var(--color-activation); }}

        /* Comparison table: 3 arch types side by side */
        .dna-compare-table {{
            width: 100%; border-collapse: collapse; font-size: 10.5px;
        }}
        .dna-compare-table th {{
            font-size: 9.5px; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.5px; padding: 4px 6px;
            border-bottom: 1.5px solid var(--border-medium);
            color: var(--text-tertiary); text-align: center;
        }}
        .dna-compare-table td {{
            padding: 5px 6px; text-align: center; vertical-align: middle;
            border-bottom: 1px solid var(--border-light);
            font-size: 10px;
        }}
        .dna-compare-table tr:last-child td {{ border-bottom: none; }}
        .dna-compare-table .row-key {{
            text-align: left; font-weight: 600;
            color: var(--text-secondary); padding-right: 8px;
        }}
        .cell-yes    {{ color: var(--color-activation); font-weight: 700; }}
        .cell-no     {{ color: var(--text-tertiary); }}
        .cell-active {{ background: rgba(92,151,245,.10); font-weight: 700;
                        color: var(--color-tensor); border-radius: 3px; }}
        .cell-enc    {{ color: var(--color-tensor); font-weight: 600; }}
        .cell-dec    {{ color: var(--color-layer);  font-weight: 600; }}
        .cell-encdec {{ color: var(--color-activation); font-weight: 600; }}

        /* ═══════════════════════════════════════════════════
           19. BIDIRECTIONAL ATTENTION indicator  (encoder)
        ═══════════════════════════════════════════════════ */
        .bidir-indicator {{
            display: flex; align-items: center; justify-content: center;
            gap: 4px; font-size: 10px; font-weight: 700;
            color: var(--color-tensor);
            padding: 4px 10px; border-radius: 4px;
            background: rgba(92,151,245,.12);
            border: 1px dashed rgba(92,151,245,.4);
            margin: 4px 0 8px; letter-spacing: 0.3px;
        }}

        /* SWA (Sliding Window Attention) indicator */
        .swa-indicator {{
            display: flex; align-items: center; justify-content: center;
            gap: 4px; font-size: 10px; font-weight: 700;
            color: var(--color-layer);
            padding: 4px 10px; border-radius: 4px;
            background: rgba(230,124,115,.12);
            border: 1px dashed rgba(230,124,115,.4);
            margin: 4px 0 8px; letter-spacing: 0.3px;
        }}

        /* MoE expert routing visualisation */
        .moe-router {{
            display: flex; align-items: center; justify-content: center;
            gap: 6px; flex-wrap: wrap;
            padding: 8px; margin: 4px 0 8px;
            background: rgba(123,31,162,.08);
            border: 1.5px dashed rgba(123,31,162,.3);
            border-radius: 8px;
        }}
        .moe-expert {{
            width: 26px; height: 26px; border-radius: 4px;
            display: flex; align-items: center; justify-content: center;
            font-size: 9px; font-weight: 700; color: white;
        }}
        .moe-expert.active  {{ background: #7b1fa2; box-shadow: 0 0 6px rgba(123,31,162,.5); }}
        .moe-expert.passive {{ background: rgba(123,31,162,.25); color: var(--text-tertiary); }}

        /* Hy3 overview + grouping */
        .hy3-overview {{
            padding: 14px 40px 0;
            background: var(--header-bg);
            border-bottom: 1px solid var(--border-light);
        }}
        .hy3-overview-grid {{
            display: grid;
            grid-template-columns: 1.25fr 1fr 1fr;
            gap: 12px;
            margin-bottom: 14px;
        }}
        .hy3-card {{
            background: var(--group-bg);
            border: 1.5px solid var(--border-medium);
            border-radius: 12px;
            padding: 14px 16px;
            box-shadow: var(--shadow-sm);
        }}
        .hy3-card-title {{
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: var(--text-tertiary);
            margin-bottom: 10px;
        }}
        .hy3-card-copy {{
            font-size: 13px;
            color: var(--text-secondary);
            line-height: 1.55;
        }}
        .hy3-card-copy strong {{
            color: var(--text-primary);
        }}
        .hy3-chip-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}
        .hy3-chip {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 10px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.2px;
            border: 1px solid transparent;
        }}
        .hy3-chip-dense {{
            background: rgba(52,168,83,.12);
            color: #1b5e20;
            border-color: rgba(52,168,83,.28);
        }}
        .hy3-chip-moe {{
            background: rgba(123,31,162,.10);
            color: #6a1b9a;
            border-color: rgba(123,31,162,.25);
        }}
        .hy3-chip-router {{
            background: rgba(239,108,0,.10);
            color: #bf5f00;
            border-color: rgba(239,108,0,.24);
        }}
        .hy3-chip-attn {{
            background: rgba(25,103,210,.10);
            color: #174ea6;
            border-color: rgba(25,103,210,.22);
        }}
        .hy3-chip-mtp {{
            background: rgba(103,58,183,.10);
            color: #5e35b1;
            border-color: rgba(103,58,183,.24);
        }}
        .hy3-chip-context {{
            background: rgba(230,124,115,.10);
            color: #b3261e;
            border-color: rgba(230,124,115,.24);
        }}
        html[data-theme="dark"] .hy3-chip-dense,
        html[data-theme="nord"] .hy3-chip-dense,
        html[data-theme="ocean"] .hy3-chip-dense {{
            background: rgba(70,194,95,.16);
            color: #69d97e;
        }}
        html[data-theme="dark"] .hy3-chip-moe,
        html[data-theme="nord"] .hy3-chip-moe,
        html[data-theme="ocean"] .hy3-chip-moe {{
            background: rgba(123,31,162,.18);
            color: #d1b3f5;
        }}
        html[data-theme="dark"] .hy3-chip-router,
        html[data-theme="nord"] .hy3-chip-router,
        html[data-theme="ocean"] .hy3-chip-router {{
            background: rgba(239,108,0,.20);
            color: #ffb74d;
        }}
        html[data-theme="dark"] .hy3-chip-attn,
        html[data-theme="nord"] .hy3-chip-attn,
        html[data-theme="ocean"] .hy3-chip-attn {{
            background: rgba(25,103,210,.18);
            color: #8ab4f8;
        }}
        html[data-theme="dark"] .hy3-chip-mtp,
        html[data-theme="nord"] .hy3-chip-mtp,
        html[data-theme="ocean"] .hy3-chip-mtp {{
            background: rgba(103,58,183,.20);
            color: #c7b6ff;
        }}
        html[data-theme="dark"] .hy3-chip-context,
        html[data-theme="nord"] .hy3-chip-context,
        html[data-theme="ocean"] .hy3-chip-context {{
            background: rgba(230,124,115,.18);
            color: #f28b82;
        }}
        .hy3-layer-groups {{
            width: 100%;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .hy3-layer-box {{
            width: 100%;
            border-radius: 10px;
            padding: 12px 14px;
            color: white;
            box-shadow: var(--shadow-sm);
        }}
        .hy3-layer-box strong {{
            display: block;
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 4px;
        }}
        .hy3-layer-box span {{
            display: block;
            font-size: 11px;
            opacity: 0.92;
        }}
        .hy3-layer-box.dense {{
            background: linear-gradient(135deg, rgba(52,168,83,.92) 0%, rgba(27,94,32,.95) 100%);
        }}
        .hy3-layer-box.moe {{
            background: linear-gradient(135deg, rgba(123,31,162,.92) 0%, rgba(74,20,140,.95) 100%);
        }}
        .hy3-layer-box.mtp {{
            background: linear-gradient(135deg, rgba(103,58,183,.92) 0%, rgba(49,27,146,.95) 100%);
        }}
        .hy3-subgroup-note {{
            margin-top: 10px;
            padding: 8px 10px;
            border-radius: 8px;
            background: rgba(123,31,162,.08);
            border: 1px dashed rgba(123,31,162,.25);
            color: var(--text-secondary);
            font-size: 10.5px;
            line-height: 1.45;
            text-align: center;
        }}
        .hy3-section-tag {{
            display: inline-flex;
            align-items: center;
            padding: 4px 8px;
            border-radius: 999px;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}
        .hy3-section-tag.dense {{
            background: rgba(52,168,83,.12);
            color: #1b5e20;
        }}
        .hy3-section-tag.moe {{
            background: rgba(123,31,162,.10);
            color: #6a1b9a;
        }}
        .hy3-section-tag.mtp {{
            background: rgba(103,58,183,.10);
            color: #5e35b1;
        }}
        .hy3-inline-note {{
            margin-top: 8px;
            font-size: 10px;
            color: var(--text-secondary);
            text-align: center;
            line-height: 1.45;
        }}
        .hy3-router-caption {{
            width: 100%;
            text-align: center;
            font-size: 10px;
            color: var(--text-secondary);
            margin-top: 4px;
        }}

        @media (max-width: 1200px) {{
            .hy3-overview {{ padding: 14px 24px 0; }}
            .hy3-overview-grid {{ grid-template-columns: 1fr; }}
        }}

        /* col widths for extra columns */
        .col-4 {{ width: 340px; }}
        .col-5 {{ width: 320px; }}
    </style>"""

    # ── Header ────────────────────────────────────────────────────────────────

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

        def row(k, v_html):
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
        def chk(cond):  return '<span class="cell-yes">✔</span>' if cond else '<span class="cell-no">–</span>'
        def act_cell(cond): return f'<td class="cell-active">{chk(cond)}</td>'
        def nrm_cell(cond): return f'<td>{chk(cond)}</td>'

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

    def _render_export_libs(self) -> str:
        return """
    <!-- ── Export Libraries (loaded async, export buttons wait for them) ── -->
    <script
        src="https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js"
        defer></script>
    <script
        src="https://cdn.jsdelivr.net/npm/dom-to-image-more@3.4.0/dist/dom-to-image-more.min.js"
        defer></script>"""

    # ── JavaScript ────────────────────────────────────────────────────────────

    def _render_scripts(self, data: Dict) -> str:
        json_data = self._script_json(data)

        return f"""
    <script>
    // ═══════════════════════════════════════════════════════════════
    //  Architecture metadata (injected by Python renderer)
    // ═══════════════════════════════════════════════════════════════
    const ARCH_DATA = {json_data};

    // ═══════════════════════════════════════════════════════════════
    //  ThemeManager  – client-side theme switching
    // ═══════════════════════════════════════════════════════════════
    class ThemeManager {{
        constructor() {{
            this.root = document.documentElement;
            this.current = this.root.getAttribute('data-theme') || 'light';
            this._bindButtons();
        }}

        switch(themeId) {{
            this.root.setAttribute('data-theme', themeId);
            this.current = themeId;
            // Update active state on buttons
            document.querySelectorAll('.theme-btn').forEach(btn => {{
                btn.classList.toggle('active', btn.dataset.themeId === themeId);
            }});
            // Redraw SVG connections because CSS variable --text-tertiary changed.
            window._connMgr?.redraw();
            // Persist choice across page reloads.
            try {{ localStorage.setItem('vitriol-theme', themeId); }} catch(_) {{}}
        }}

        _bindButtons() {{
            document.querySelectorAll('.theme-btn').forEach(btn => {{
                btn.addEventListener('click', () => this.switch(btn.dataset.themeId));
            }});
            // Restore persisted theme.
            try {{
                const saved = localStorage.getItem('vitriol-theme');
                if (saved && saved !== this.current) this.switch(saved);
            }} catch(_) {{}}
        }}
    }}

    // ═══════════════════════════════════════════════════════════════
    //  ConnectionManager  – cross-column SVG bezier curves
    // ═══════════════════════════════════════════════════════════════
    class ConnectionManager {{
        constructor() {{
            this.svg = document.getElementById('svg-connections');
            this.connections = ARCH_DATA.topology.connections || [];
        }}

        _drawOne(conn) {{
            const fromId = conn.from;
            const toId   = conn.to;
            const type   = conn.type || 'flow';
            const curvature = conn.curve || 0.5;

            const fromEl = document.getElementById(fromId);
            const toEl   = document.getElementById(toId);
            if (!fromEl || !toEl) {{ console.warn(`Connection missing: ${{fromId}} → ${{toId}}`); return; }}

            const cr  = this.svg.parentElement.getBoundingClientRect();
            const fr  = fromEl.getBoundingClientRect();
            const tr  = toEl.getBoundingClientRect();

            // Start from right center
            const x1 = fr.right  - cr.left;
            const y1 = fr.top    + fr.height / 2 - cr.top;
            // End at left center (usually)
            const x2 = tr.left   - cr.left;
            let y2 = tr.top    + tr.height / 2 - cr.top;

            // For expansion arrows, point slightly higher to "open up" the block
            if (type === 'expansion') {{
                y2 = tr.top + 40 - cr.top;
            }}

            const dx = x2 - x1;
            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');

            // Set class based on type
            if (type === 'expansion') {{
                path.setAttribute('class', 'svg-connector svg-connector-expansion');
                // Dashed style handled in CSS or here
                path.style.strokeDasharray = "6 4";
                path.style.opacity = "0.4";
            }} else {{
                path.setAttribute('class', 'svg-connector');
            }}

            path.setAttribute('d',
                `M ${{x1}} ${{y1}} C ${{x1 + dx * curvature}} ${{y1}}, ${{x2 - dx * curvature}} ${{y2}}, ${{x2}} ${{y2}}`
            );
            path.setAttribute('marker-end', 'url(#arrowhead)');
            path.dataset.from = fromId;
            path.dataset.to   = toId;
            this.svg.appendChild(path);
        }}

        drawAll() {{
            this.svg.querySelectorAll('.svg-connector').forEach(el => el.remove());
            this.connections.forEach(c => this._drawOne(c));
        }}

        redraw() {{
            requestAnimationFrame(() => this.drawAll());
        }}
    }}

    // ═══════════════════════════════════════════════════════════════
    //  HoverHighlightManager
    //  ─ Fog-of-war dimming of non-connected nodes
    //  ─ Animated SVG paths (cross-column + within-column)
    //  ─ Floating tooltip with component metadata
    // ═══════════════════════════════════════════════════════════════
    class HoverHighlightManager {{
        constructor() {{
            this.svg       = document.getElementById('svg-connections');
            this.container = document.querySelector('.main-container');
            this.tooltip   = this._createTooltip();
            this.graph     = ARCH_DATA.topology.graph || {{}};
            this._active   = false;
            this._setupHover();
            this.container?.addEventListener('mouseleave', () => this.deactivate());
        }}

        // ── Tooltip DOM element ──────────────────────────────────────
        _createTooltip() {{
            const el = document.createElement('div');
            el.id = 'hover-tooltip';
            el.className = 'hover-tooltip';
            document.body.appendChild(el);
            return el;
        }}

        // ── Set up mouseenter / mousemove / mouseleave ───────────────
        _setupHover() {{
            document.querySelectorAll('[data-node-id]').forEach(el => {{
                el.addEventListener('mouseenter', () => {{
                    const nid = el.dataset.nodeId;
                    if (this.graph[nid]) this.activate(nid);
                }});
                el.addEventListener('mousemove', e => {{
                    if (this._active) this._positionTooltip(e);
                }});
                el.addEventListener('mouseleave', () => this.deactivate());
            }});
        }}

        // ── Resolve element by graph id ──────────────────────────────
        _find(id) {{
            return document.querySelector(`[data-node-id="${{id}}"]`)
                || document.getElementById(id);
        }}

        // ── Activate hover highlight for a node ──────────────────────
        activate(nodeId) {{
            this._active = true;
            const node = this.graph[nodeId];
            if (!node) return;

            this.container.classList.add('is-highlighting');

            // Self
            this._find(nodeId)?.classList.add('node-self');

            // Upstream neighbours  (orange glow)
            node.up.forEach(id => this._find(id)?.classList.add('node-upstream'));

            // Downstream neighbours (blue glow)
            node.down.forEach(id => this._find(id)?.classList.add('node-downstream'));

            // Highlight existing cross-column SVG paths
            this._styleSVGPaths(nodeId, node.up, node.down);

            // Draw new within-column animated temp paths
            this._drawTempPaths(nodeId, node.up, node.down);

            // Update floating tooltip
            this._updateTooltip(node);
        }}

        // ── Deactivate & restore normal state ────────────────────────
        deactivate() {{
            if (!this._active) return;
            this._active = false;

            this.container?.classList.remove('is-highlighting');
            document.querySelectorAll('.node-self,.node-upstream,.node-downstream')
                    .forEach(el => el.classList.remove('node-self','node-upstream','node-downstream'));

            this.svg.querySelectorAll('.svg-temp-up,.svg-temp-down').forEach(e => e.remove());
            this.svg.querySelectorAll('.svg-connector')
                    .forEach(p => p.classList.remove('svg-connector-up','svg-connector-down'));

            this.tooltip.classList.remove('visible');
        }}

        // ── Style existing cross-column bezier paths ─────────────────
        _styleSVGPaths(nodeId, upIds, downIds) {{
            this.svg.querySelectorAll('.svg-connector').forEach(path => {{
                const f = path.dataset.from;
                const t = path.dataset.to;
                // Path goes FROM an upstream node (or self) to self or a downstream
                if (upIds.includes(f) || (f === nodeId && !downIds.includes(t)))
                    path.classList.add('svg-connector-up');
                if (f === nodeId || (downIds.includes(t) && f === nodeId))
                    path.classList.add('svg-connector-down');
                // Cross-column: from self → downstream panel
                if (f === nodeId && downIds.includes(t))
                    path.classList.add('svg-connector-down');
                // Cross-column: from upstream → self
                if (t === nodeId && upIds.includes(f))
                    path.classList.add('svg-connector-up');
            }});
        }}

        // ── Draw animated within-column temp paths ───────────────────
        _drawTempPaths(nodeId, upIds, downIds) {{
            const cr  = this.container.getBoundingClientRect();
            const scT = this.container.scrollTop;
            const scL = this.container.scrollLeft;

            const draw = (fromId, toId, dir) => {{
                // Skip if a static cross-column path already covers this link
                if (this.svg.querySelector(`path[data-from="${{fromId}}"][data-to="${{toId}}"]`)) return;

                const fEl = this._find(fromId);
                const tEl = this._find(toId);
                if (!fEl || !tEl) return;

                const fr = fEl.getBoundingClientRect();
                const tr = tEl.getBoundingClientRect();

                // SVG coordinates (account for container scroll)
                const x1 = fr.left + fr.width  / 2 - cr.left + scL;
                const y1 = fr.bottom               - cr.top  + scT;
                const x2 = tr.left + tr.width  / 2 - cr.left + scL;
                const y2 = tr.top                  - cr.top  + scT;
                const cy = (y1 + y2) / 2;

                const p = document.createElementNS('http://www.w3.org/2000/svg','path');
                p.setAttribute('class', `svg-temp-${{dir}}`);
                p.setAttribute('d', `M ${{x1}} ${{y1}} C ${{x1}} ${{cy}} ${{x2}} ${{cy}} ${{x2}} ${{y2}}`);
                p.setAttribute('marker-end', `url(#arrowhead-${{dir}})`);
                this.svg.appendChild(p);
            }};

            upIds.forEach(id   => draw(id,     nodeId, 'up'));
            downIds.forEach(id => draw(nodeId, id,     'down'));
        }}

        // ── Update tooltip content ───────────────────────────────────
        _updateTooltip(node) {{
            const u = node.up.length;
            const d = node.down.length;
            this.tooltip.innerHTML = `
                <div class="ht-title">${{node.label}}</div>
                <div class="ht-stats">
                    ${{u > 0 ? `<span class="ht-upstream">↑ ${{u}} upstream</span>` : ''}}
                    ${{d > 0 ? `<span class="ht-downstream">↓ ${{d}} downstream</span>` : ''}}
                </div>
                ${{node.desc ? `<div class="ht-desc">${{node.desc}}</div>` : ''}}
                <div class="ht-total">${{u+d}} direct connection${{u+d!==1?'s':''}}</div>`;
            this.tooltip.classList.add('visible');
        }}

        // ── Keep tooltip near cursor ─────────────────────────────────
        _positionTooltip(e) {{
            const OFF = 18;
            const tw = this.tooltip.offsetWidth;
            const th = this.tooltip.offsetHeight;
            let x = e.clientX + OFF;
            let y = e.clientY + OFF;
            if (x + tw > window.innerWidth  - 6) x = e.clientX - tw - OFF;
            if (y + th > window.innerHeight - 6) y = e.clientY - th - OFF;
            this.tooltip.style.left = `${{x}}px`;
            this.tooltip.style.top  = `${{y}}px`;
        }}
    }}

    // ═══════════════════════════════════════════════════════════════
    //  InteractionManager  – click logging only (hover handled above)
    // ═══════════════════════════════════════════════════════════════
    class InteractionManager {{
        constructor() {{ this._setup(); }}
        _setup() {{
            document.querySelectorAll('.interactive').forEach(el => {{
                el.addEventListener('click', e =>
                    console.log('Clicked:', e.currentTarget.textContent.trim()));
            }});
        }}
    }}

    // ═══════════════════════════════════════════════════════════════
    //  ExportManager  – PNG via html2canvas, SVG via dom-to-image-more
    // ═══════════════════════════════════════════════════════════════
    class ExportManager {{
        constructor() {{
            this.modelType = ARCH_DATA.model_type || 'architecture';
            this._bindButtons();
        }}

        // ── Utility: show toast ─────────────────────────────────────
        _toast(msg, duration = 3000) {{
            const el = document.getElementById('export-toast');
            el.textContent = msg;
            el.classList.add('show');
            setTimeout(() => el.classList.remove('show'), duration);
        }}

        // ── Utility: temporarily unlock scroll for full capture ─────
        _unlockScroll() {{
            const body = document.body;
            const mc   = document.querySelector('.main-container');
            const prev = {{
                bodyOverflow: body.style.overflow,
                bodyHeight:   body.style.height,
                mcOverflow:   mc ? mc.style.overflow : '',
                mcHeight:     mc ? mc.style.height   : '',
            }};
            body.style.overflow = 'visible';
            body.style.height   = 'auto';
            if (mc) {{ mc.style.overflow = 'visible'; mc.style.height = 'auto'; }}
            return () => {{
                body.style.overflow = prev.bodyOverflow;
                body.style.height   = prev.bodyHeight;
                if (mc) {{ mc.style.overflow = prev.mcOverflow; mc.style.height = prev.mcHeight; }}
            }};
        }}

        // ── Utility: button loading / success states ────────────────
        _setLoading(btn, loading) {{
            btn.disabled = loading;
            btn.classList.toggle('loading', loading);
        }}
        _flashSuccess(btn, label) {{
            btn.classList.remove('loading');
            btn.classList.add('success');
            const orig = btn.innerHTML;
            btn.innerHTML = '✅ Done';
            setTimeout(() => {{ btn.classList.remove('success'); btn.innerHTML = orig; btn.disabled = false; }}, 1800);
        }}

        // ── PNG export ─────────────────────────────────────────────
        async exportPNG() {{
            if (typeof html2canvas === 'undefined') {{
                this._toast('⚠️ html2canvas not loaded yet – try again in a moment.');
                return;
            }}
            const btn = document.getElementById('btn-export-png');
            this._setLoading(btn, true);
            const restore = this._unlockScroll();
            // Redraw connections at new size.
            window._connMgr?.drawAll();
            await new Promise(r => setTimeout(r, 150));

            try {{
                const canvas = await html2canvas(document.body, {{
                    scale: 2,
                    useCORS: true,
                    logging: false,
                    allowTaint: false,
                    backgroundColor: getComputedStyle(document.documentElement)
                                        .getPropertyValue('--bg-primary').trim() || '#ffffff',
                }});
                const link = document.createElement('a');
                link.download = `vitriol-${{this.modelType}}.png`;
                link.href = canvas.toDataURL('image/png');
                link.click();
                this._toast('🖼 PNG exported successfully!');
                this._flashSuccess(btn, '🖼 PNG');
            }} catch (err) {{
                console.error('PNG export failed:', err);
                this._toast('❌ PNG export failed – see console for details.');
                btn.disabled = false;
                btn.classList.remove('loading');
            }} finally {{
                restore();
            }}
        }}

        // ── SVG export ─────────────────────────────────────────────
        async exportSVG() {{
            if (typeof domtoimage === 'undefined') {{
                this._toast('⚠️ dom-to-image-more not loaded yet – try again in a moment.');
                return;
            }}
            const btn = document.getElementById('btn-export-svg');
            this._setLoading(btn, true);
            const restore = this._unlockScroll();
            window._connMgr?.drawAll();
            await new Promise(r => setTimeout(r, 150));

            try {{
                const svgDataUrl = await domtoimage.toSvg(document.body, {{
                    bgcolor: getComputedStyle(document.documentElement)
                                 .getPropertyValue('--bg-primary').trim() || '#ffffff',
                }});
                const link = document.createElement('a');
                link.download = `vitriol-${{this.modelType}}.svg`;
                link.href = svgDataUrl;
                link.click();
                this._toast('✏️ SVG exported successfully!');
                this._flashSuccess(btn, '✏️ SVG');
            }} catch (err) {{
                console.error('SVG export failed:', err);
                this._toast('❌ SVG export failed – see console for details.');
                btn.disabled = false;
                btn.classList.remove('loading');
            }} finally {{
                restore();
            }}
        }}

        _bindButtons() {{
            document.getElementById('btn-export-png')
                ?.addEventListener('click', () => this.exportPNG());
            document.getElementById('btn-export-svg')
                ?.addEventListener('click', () => this.exportSVG());
        }}
    }}

    // ═══════════════════════════════════════════════════════════════
    //  StatsManager  – console diagnostics
    // ═══════════════════════════════════════════════════════════════
    class StatsManager {{
        constructor(data) {{
            const s = data.statistics;
            console.group('📊 Vitriol – Architecture Statistics');
            console.log('Model Type:',         data.model_type);
            console.log('Parameters:',         (s.total_params / 1e9).toFixed(2) + 'B');
            console.log('Memory (FP16):',      s.memory_fp16_gb.toFixed(2) + ' GB');
            console.log('Decoder Layers:',     s.total_layers);
            console.log('Head Dimension:',     s.head_dim);
            console.log('Available Themes:',   data.available_themes.map(t=>t.id).join(', '));
            console.groupEnd();
        }}
    }}

    // ═══════════════════════════════════════════════════════════════
    //  Responsive handler
    // ═══════════════════════════════════════════════════════════════
    class ResponsiveManager {{
        constructor(connMgr) {{
            let t;
            window.addEventListener('resize', () => {{
                clearTimeout(t);
                t = setTimeout(() => connMgr.redraw(), 150);
            }});
        }}
    }}

    // ═══════════════════════════════════════════════════════════════
    //  Bootstrap
    // ═══════════════════════════════════════════════════════════════
    document.addEventListener('DOMContentLoaded', () => {{
        console.log('🚀 Vitriol Visualizer – initialising…');

        const connMgr  = new ConnectionManager();
        window._connMgr = connMgr;          // exposed for ThemeManager

        new ThemeManager();
        new HoverHighlightManager();
        new InteractionManager();
        new ResponsiveManager(connMgr);
        new ExportManager();
        new StatsManager(ARCH_DATA);

        // Draw connections after first layout pass.
        setTimeout(() => connMgr.drawAll(), 120);
    }});
    </script>"""


# ─── quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    OUT = "/mnt/user-data/outputs"
    os.makedirs(OUT, exist_ok=True)

    arch = Architecture(
        model_type="Qwen2-7B",
        arch_type="decoder-only",
        total_layers=32,
        total_params=7_000_000_000,
        memory_fp16_gb=13.0,
        parameters={
            "hidden_size": 4096,
            "num_heads": 32,
            "num_kv_heads": 8,
            "intermediate_size": 11008,
            "vocab_size": 152064,
            "max_position": 32768,
        },
        features=["GQA", "RoPE", "RMSNorm", "SwiGLU", "Causal"],
    )

    # Render all themes for a representative decoder-only architecture.
    for theme in HTMLRenderer.THEMES:
        r = HTMLRenderer(theme=theme)
        r.render(arch, f"{OUT}/vitriol_{theme}.html")
        print(f"[✓] theme={theme:12s}  {arch.model_type}")

    print("\nAll done. Open any vitriol_*.html in a browser.")
