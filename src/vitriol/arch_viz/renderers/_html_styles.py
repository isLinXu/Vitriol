"""Theme CSS + web-font helpers for HTMLRenderer."""

class _HtmlStylesMixin:
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
