from pathlib import Path


def test_3d_has_handshake_response_markers() -> None:
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    assert "handshake" in html
    assert "vitriol_handshake" in html
    assert "postMessage" in html

