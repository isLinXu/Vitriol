from pathlib import Path


def test_3d_visualizer_has_token_playback_toolbar_skeleton_and_truthfulness_copy() -> None:
    """
    "Inference playback" in the 3D visualizer is an optional enhancement.

    Here we assert a minimal, testable skeleton (DOM anchors) to ensure:
    - future JS can reliably attach behavior (play/pause, step, speed, token slider)
    - the page includes explicit truthfulness copy so users don't mistake structure playback for
      real token-by-token inference
    """
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")

    # 1) Toolbar container and key control anchors (for future JS mounting).
    assert 'id="playbackToolbar"' in html
    assert 'id="playPauseBtn"' in html
    assert 'id="stepBtn"' in html
    assert 'id="speedSelect"' in html
    assert 'id="tokenSlider"' in html

    # 2) Truthfulness copy: must explicitly indicate this is structure-driven playback.
    assert "Structure-driven" in html

    # 3) Minimal testable skeleton for PlaybackEngine (structure-driven, RAF-driven): class + key methods/fields.
    assert "class PlaybackEngine" in html
    for method_sig in ["play()", "pause()", "step()", "setSpeed(x)", "onChange(cb)"]:
        assert method_sig in html
    for state_field in ["paused", "speed", "tokenIndex", "layerIndex"]:
        assert state_field in html


def test_3d_visualizer_has_node_index_builder() -> None:
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    assert "function buildNodeIndex" in html
    assert "nodeIndex" in html


def test_3d_visualizer_has_token_playback_markers() -> None:
    """
    Task 4: 3D visualizer should expose minimal "playback marker" anchors:
    - createTokenParticle(tokenIndex): create/reuse the cursor particle
    - highlightNode(nodeId): highlight the current node and reset the previous one
    """
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    assert "function createTokenParticle" in html
    assert "function highlightNode" in html
    assert "lerpVectors(tokenParticleAnim.from, tokenParticleAnim.to, t)" in html
    assert "from.clone().lerp" not in html


def test_3d_visualizer_binds_playback_controls() -> None:
    """
    Task 5: should provide a UI → PlaybackEngine binding function (structure-only assertion).
    """
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    assert "function bindPlaybackControls" in html


def test_3d_visualizer_caches_nav_and_token_lookup_hotspots() -> None:
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    assert "navItemsByNodeId" in html
    assert "navTreeItems" in html
    assert "activeNavTreeItem" in html
    assert "tokenChipByIndex" in html
    assert "eventByTokenGlobalIndex" in html
    assert "moduleSelectionByMesh" in html
    assert "moduleLayoutByName" in html
    assert "__vitriolTabs" in html
    assert "getElementsByClassName('vitriol-tab')" in html
    assert "getElementsByClassName('vitriol-drawer-panel')" in html
    assert "document.createDocumentFragment()" in html
    assert "replaceChildren(fragment)" in html


def test_2d_visualizer_has_playback_status_panel() -> None:
    """
    Task 6: 2D visualizer should provide DOM anchors for a read-only "sync status panel".
    """
    html = Path("src/vitriol/viz/model_visualizer.html").read_text(encoding="utf-8")
    assert 'id="playbackStatusPanel"' in html
    assert 'id="currentTokenText"' in html
    assert 'id="currentLayerText"' in html
