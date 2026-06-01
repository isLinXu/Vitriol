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
from ._html_columns import _HtmlColumnsMixin
from ._html_scripts import _HtmlScriptsMixin
from ._html_sections import _HtmlSectionsMixin
from ._html_styles import _HtmlStylesMixin


class HTMLRenderer(
    _HtmlStylesMixin,
    _HtmlSectionsMixin,
    _HtmlColumnsMixin,
    _HtmlScriptsMixin,
):
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


# ─── quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
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
