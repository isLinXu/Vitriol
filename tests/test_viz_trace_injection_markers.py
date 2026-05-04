from pathlib import Path
import re


def test_3d_has_inline_trace_marker() -> None:
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    assert "INLINE_TRACE_MARKER" in html
    assert "__VITRIOL_TRACE__" in html


def test_token_panel_does_not_cover_playback_toolbar() -> None:
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    # Use a dynamic CSS variable for token panel bottom to avoid covering playback toolbar buttons/toggles.
    assert "--vitriol-playback-h" in html
    assert "calc(var(--vitriol-playback-h)" in html or "var(--vitriol-playback-h)" in html


def test_3d_has_auto_focus_toggle_and_attn_heat_css() -> None:
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    assert 'id="autoFocusToggle"' in html
    assert "token-chip-attn" in html or "attn-heat" in html
    # Attention Lens: edge overlay + histogram heat-bar canvas.
    assert 'id="attnCanvas"' in html
    assert 'id="attnHistogramCanvas"' in html
    # Task 4: trace replay DOM markers
    assert 'id="tokenListPanel"' in html
    assert 'id="tokenList"' in html
    assert 'id="followCameraToggle"' in html
    # UI Professionalization Task 1: design tokens + skeleton containers
    assert "--ui-panel" in html
    assert "--ui-stroke" in html
    assert 'id="lensDrawer"' in html
    assert 'id="playbackBar"' in html


def test_lens_drawer_and_playback_shell_classnames_are_stable() -> None:
    """
    Task 4 (professional polish):
    Add stable class names to key containers (as anchors for UI refactors) to avoid regressions
    during subsequent polish/refactors.
    """
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")

    # lensDrawer must include vitriol-drawer-shell (stable container class).
    assert re.search(
        r'<aside[^>]*id="lensDrawer"[^>]*class="[^"]*vitriol-drawer-shell',
        html,
        flags=re.IGNORECASE,
    ), "lensDrawer must include class vitriol-drawer-shell"

    # playbackBar must include vitriol-playback-shell (stable container class).
    assert re.search(
        r'<div[^>]*id="playbackBar"[^>]*class="[^"]*vitriol-playback-shell',
        html,
        flags=re.IGNORECASE,
    ), "playbackBar must include class vitriol-playback-shell"


def test_token_list_panel_is_migrated_into_lens_drawer_and_has_tabs() -> None:
    """
    UI Professionalization Task 3:
    1) tokenListPanel must be moved into aside#lensDrawer (avoid floating layer overlap/layout conflicts)
    2) lensDrawer must include a tabs structure (tokens/lens/meta) to enable further panel splitting
    """
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")

    drawer_start = html.index('id="lensDrawer"')
    drawer_end = html.find("</aside>", drawer_start)
    assert drawer_end != -1, "missing </aside> closing tag for #lensDrawer"

    token_panel_idx = html.find('id="tokenListPanel"')
    assert token_panel_idx != -1, 'missing id="tokenListPanel"'
    assert drawer_start < token_panel_idx < drawer_end, "tokenListPanel must be inside #lensDrawer"

    # Tabs: either explicit ids (tokensTab/lensTab/metaTab) or data-tab structure
    has_id_tabs = all(t in html for t in ['id="tokensTab"', 'id="lensTab"', 'id="metaTab"'])
    has_data_tabs = all(t in html for t in ['data-tab="tokens"', 'data-tab="lens"', 'data-tab="meta"'])
    assert has_id_tabs or has_data_tabs, "lensDrawer must include tabs (tokens/lens/meta)"


def _find_matching_div_end(html: str, div_open_idx: int) -> int:
    """
    Approximate DOM range detection:
    starting from a given <div ...>, count nested <div / </div> tags to find the matching </div>.
    """
    # Find the first "<div" starting at/after div_open_idx to anchor the depth counter.
    start = html.find("<div", div_open_idx)
    if start < 0:
        raise AssertionError("cannot find <div> from div_open_idx")

    tag_re = re.compile(r"</?div\b", re.IGNORECASE)
    depth = 0

    for m in tag_re.finditer(html, start):
        token = m.group(0).lower()
        if token == "<div":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                # include the full closing tag
                close_gt = html.find(">", m.end())
                return (close_gt + 1) if close_gt >= 0 else m.end()

    raise AssertionError("cannot find matching </div> for playbackBar")


def test_playback_controls_are_inside_playback_bar() -> None:
    """
    UI Professionalization Task 2:
    Playback controls must be placed inside the bottom #playbackBar container to avoid interaction
    drift when styles/layouts are refactored later.
    We use a string-range (approximate DOM) assertion here.
    """
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")

    pb_start = html.index('id="playbackBar"')
    pb_end = _find_matching_div_end(html, pb_start)

    for dom_id in [
        "playPauseBtn",
        "stepBtn",
        "speedSelect",
        "tokenSlider",
        "followCameraToggle",
        "autoFocusToggle",
    ]:
        needle = f'id="{dom_id}"'
        idx = html.find(needle)
        assert idx != -1, f"missing {needle}"
        assert pb_start < idx < pb_end, f"{needle} must be inside #playbackBar"


def test_playback_bar_is_shown_by_default() -> None:
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    # Regression guard: playback bar should not be permanently display:none (otherwise users can't find play).
    assert 'id="playbackBar"' in html
    assert 'id="playPauseBtn"' in html
    assert "playbackBar.style.display" in html or "playbackBarEl.style.display" in html
