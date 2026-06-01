"""Client-side export libs and the interactive JavaScript bundle."""

from typing import Dict


class _HtmlScriptsMixin:
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
