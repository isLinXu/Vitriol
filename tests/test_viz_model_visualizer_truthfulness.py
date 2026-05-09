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


def test_model_visualizer_exposes_stats_narrative_and_details_note() -> None:
    html = Path("src/vitriol/viz/model_visualizer.html").read_text(encoding="utf-8")
    assert "statsNarrative" in html
    assert "This 2D view is a structural explanation layer." in html
    assert "Current totals are browser-side estimates" in html
    assert "Current totals come from backend-derived metadata" in html


def test_model_visualizer_layer_list_shows_trace_hint() -> None:
    html = Path("src/vitriol/viz/model_visualizer.html").read_text(encoding="utf-8")
    assert "layer-hint" in html
    assert "${layer.trace_id || layer.id}" in html


def test_model_visualizer_has_toast_feedback_for_load_and_blocked_states() -> None:
    html = Path("src/vitriol/viz/model_visualizer.html").read_text(encoding="utf-8")
    assert "toastStack" in html
    assert "function showToast" in html
    assert "2D architecture loaded" in html
    assert "Model load blocked" in html


def test_model_visualizer_uses_shared_copy_feedback_instead_of_button_text_flip() -> None:
    html = Path("src/vitriol/viz/model_visualizer.html").read_text(encoding="utf-8")
    assert "copyTextWithFeedback" in html
    assert "HTTP link copied" in html
    assert "Fallback HTTP link copied" in html
    assert "Clipboard unavailable" in html
    assert "copyBtn.textContent = 'Copied'" not in html


def test_model_visualizer_detail_panel_exposes_provenance_and_clear_selection() -> None:
    html = Path("src/vitriol/viz/model_visualizer.html").read_text(encoding="utf-8")
    assert "detailsProvenance" in html
    assert "getDetailProvenance" in html
    assert "Clear Selection" in html
    assert "resetDetailPanel" in html
    assert "Trace ID" in html
    assert "Detail: structural preview" in html
    assert "Selection cleared" in html


def test_model_visualizer_sync_highlight_reuses_layer_details_context() -> None:
    html = Path("src/vitriol/viz/model_visualizer.html").read_text(encoding="utf-8")
    assert "_findLayerByNodeId" in html
    assert "_highlightExactNode" in html
    assert "showDetails(matched.layer)" in html
    assert "this.clearSelection(false, 'playback')" in html


def test_model_visualizer_submodule_details_preserve_parent_context() -> None:
    html = Path("src/vitriol/viz/model_visualizer.html").read_text(encoding="utf-8")
    assert "parent_trace_id" in html
    assert "parent_name" in html
    assert "Parent Layer" in html
    assert "Parent: ${parentTraceId}" in html
    assert "this._selectedLayerKey = traceId" in html


def test_model_visualizer_deduplicates_repeated_playback_focus_updates() -> None:
    html = Path("src/vitriol/viz/model_visualizer.html").read_text(encoding="utf-8")
    assert "lastPlaybackNodeId = null" in html
    assert "const nextNodeId = s.nodeId ? String(s.nodeId) : null" in html
    assert "if (nextNodeId === lastPlaybackNodeId)" in html
    assert "visualizer.selectByNodeId(nextNodeId)" in html


def test_model_visualizer_submodule_highlight_activates_parent_layer_row() -> None:
    html = Path("src/vitriol/viz/model_visualizer.html").read_text(encoding="utf-8")
    assert "const k = layer.parent_trace_id || layer.trace_id || layer.id" in html


def test_model_visualizer_distinguishes_manual_and_playback_selection_sources() -> None:
    html = Path("src/vitriol/viz/model_visualizer.html").read_text(encoding="utf-8")
    assert "this._selectionSource = 'none'" in html
    assert "this.clearSelection(false, 'manual')" in html
    assert "this.clearSelection(false, 'playback')" in html
    assert "visualizer._selectionSource === 'playback'" in html
    assert "visualizer.clearSelection(true, 'none')" in html
