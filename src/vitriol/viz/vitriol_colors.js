/**
 * Vitriol Shared Color Constants
 * Unified color system for all 3D visualizers.
 *
 * Two color namespaces:
 *   - ARCH: Architecture type colors (attention, MoE, FFN, etc.)
 *   - WEIGHT: Weight value colors (positive, negative, near-zero, etc.)
 *   - UI: Shared UI chrome colors (background, panel, accent, etc.)
 */
const VITRIOL_COLORS = {
    // ── Architecture type colors ──
    ARCH: {
        input:      0x3b82f6,
        embedding:  0x3b82f6,
        attention:  0x06b6d4,
        moe:        0xf59e0b,
        expert:     0x8b5cf6,
        expert_active: 0xa855f7,
        ffn:        0x10b981,
        output:     0xef4444,
        shared:     0xec4899,
        router:     0xf97316,
        norm:       0x64748b,
        connection: 0x3b82f6,
        flow:       0x60a5fa,
        linear_attn: 0x14b8a6,
        full_attn:  0x0284c7,
    },

    // ── Weight value colors ──
    WEIGHT: {
        positive:   0x4ade80,  // Green
        negative:   0xf87171,  // Red
        near_zero:  0x60a5fa,  // Blue
        ultra:      0xfbbf24,  // Gold (ultra-strided / high-magnitude)
    },

    // ── UI chrome colors ──
    UI: {
        background:  0x050508,
        fog:         0x050508,
        panel_bg:    'rgba(15, 18, 30, 0.9)',
        panel_border:'rgba(255,255,255,0.08)',
        accent:      0x3b82f6,
        accent_text: '#60a5fa',
        text_primary:'#ffffff',
        text_secondary:'#64748b',
    }
};
