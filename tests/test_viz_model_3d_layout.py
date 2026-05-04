from pathlib import Path


def test_3d_layout_uses_dynamic_header_height_safe_area() -> None:
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")

    # Use CSS variables for safe-area + header height to avoid overlap/regressions.
    assert "--vitriol-header-h" in html
    assert "--vitriol-safe" in html

    # Key panels should be positioned using a top offset derived from the header height.
    assert "calc(var(--vitriol-header-h)" in html

    # Provide stable DOM anchor ids for testability.
    assert 'id="topHeader"' in html
    assert 'id="leftNav"' in html


def test_3d_stats_dependency_uses_browser_script_build() -> None:
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    assert "stats.js@0.17.0/build/stats.min.js" in html
    assert "cdnjs.cloudflare.com/ajax/libs/stats.js/r17/Stats.min.js" not in html
