from pathlib import Path


def test_model_visualizer_blocks_implicit_demo_fallback() -> None:
    html = Path("src/vitriol/viz/model_visualizer.html").read_text(encoding="utf-8")

    # Demo mode must require an explicit demo=1 to avoid implicit "fake-truth" fallback.
    assert "demo=1" in html
    assert "isDemoEnabled" in html
    assert "showBlockedIndicator" in html

    # There should be no implicit fallback copy like "use demo data on load failure".
    assert "using demo data" not in html.lower()


def test_model_visualizer_marks_js_estimation_and_prefers_inline_total_params() -> None:
    html = Path("src/vitriol/viz/model_visualizer.html").read_text(encoding="utf-8")

    # When only JS estimation is available, the page must carry an explicit marker (estimated/source).
    assert "估算" in html or "ESTIMATED" in html

    # When CLI injected __INLINE_MODEL_CONFIG__, prefer total_params (real/backend source of truth).
    assert "__INLINE_MODEL_CONFIG__" in html
    assert "total_params" in html

    # 2D should also reuse structured fields from inline config (avoid hard-coded defaults).
    assert "inlineModelConfig" in html
    assert "hidden_size" in html
    assert "num_layers" in html

    # Additionally: 2D should support backend-injected "layer details" (from ArchitectureAnalyzer).
    assert "INLINE_ARCH_MARKER" in html
    assert "__INLINE_ARCH_DATA__" in html

    # For models like Qwen3.5 MoE: fall back to num_experts_per_tok (top-k) from config to avoid showing 0.
    assert "num_experts_per_tok" in html
