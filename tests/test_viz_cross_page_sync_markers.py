from pathlib import Path
import re


def test_3d_has_cross_page_sync_markers() -> None:
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    assert "BroadcastChannel" in html
    assert "VITRIOL_PLAYBACK_STATE" in html
    assert 'id="syncToggle"' in html or "sync=1" in html

    # Cross-page bi-directional sync (Task 1 / RED): 3D needs to handle "command" messages too.
    # We intentionally assert for explicit markers in the HTML/JS so missing features fail fast.
    assert re.search(r"type\s*:\s*['\"]command['\"]", html), "3D HTML should handle type:'command' messages."
    assert "focus_node" in html, "3D HTML should include the focus_node command/parameter marker."
    assert "engine.pause" in html or "pause();" in html, "3D command handler should pause before focusing a node."
    assert (
        "followCameraToNode(" in html or "highlightNode(" in html
    ), "3D command handler should include followCameraToNode()/highlightNode() logic."


def test_2d_has_cross_page_sync_markers() -> None:
    html = Path("src/vitriol/viz/model_visualizer.html").read_text(encoding="utf-8")
    assert "BroadcastChannel" in html
    assert "VITRIOL_PLAYBACK_STATE" in html
    assert "selectByNodeId" in html
    assert "storage" in html  # fallback listener

    # Cross-page bi-directional sync (Task 1 / RED): 2D needs to publish "command" messages.
    # Accept either BroadcastChannel postMessage or localStorage command key based publishing.
    assert re.search(r"type\s*:\s*['\"]command['\"]", html), "2D HTML should publish type:'command' messages."
    assert "focus_node" in html, "2D HTML should include the focus_node command/parameter marker."
    assert (
        # BroadcastChannel publish
        ("postMessage" in html and "BroadcastChannel" in html)
        # localStorage publish (command key)
        or re.search(r"localStorage\.(setItem|removeItem)\(\s*['\"][^'\"]*command[^'\"]*['\"]", html, re.I)
    ), "2D HTML should publish via BroadcastChannel postMessage or a localStorage command key."
