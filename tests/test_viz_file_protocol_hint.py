from pathlib import Path


def test_2d_visualizer_has_file_protocol_sync_hint() -> None:
    html = Path("src/vitriol/viz/model_visualizer.html").read_text(encoding="utf-8")
    # Must explicitly detect file://; otherwise users won't understand why sync fails.
    assert "location.protocol" in html and "file:" in html
    # Must include sync/same-origin hint anchors and the one-click open link (for UI acceptance).
    assert "同源" in html or "http://localhost" in html
    assert "openSameOrigin" in html or "same-origin" in html


def test_2d_file_protocol_hint_has_handshake_probe_markers() -> None:
    """
    Automatic port probing: 2D (file://) must use an iframe + postMessage handshake with
    3D (http://localhost:PORT) to determine the port.
    """
    html = Path("src/vitriol/viz/model_visualizer.html").read_text(encoding="utf-8")
    assert "postMessage" in html
    assert "vitriol_handshake" in html
    assert "iframe" in html



def test_2d_file_protocol_hint_uses_toast_feedback_for_open_and_copy() -> None:
    html = Path("src/vitriol/viz/model_visualizer.html").read_text(encoding="utf-8")
    assert "Opening HTTP demo" in html
    assert "Opening fallback HTTP demo" in html
    assert "copyTextWithFeedback" in html


def test_2d_file_protocol_hint_caches_detected_same_origin_port() -> None:
    html = Path("src/vitriol/viz/model_visualizer.html").read_text(encoding="utf-8")
    assert "__VITRIOL_SAME_ORIGIN_HTTP__" in html
    assert "sessionStorage.getItem(sameOriginCacheKey)" in html
    assert "localStorage.getItem(sameOriginCacheKey)" in html
    assert "persistOrigin(origin)" in html
    assert "configureOriginActions(cachedOrigin, 'cached')" in html
    assert "Use cached" in html
