from __future__ import annotations

import json
from pathlib import Path
import sys

# Allow running tests without installing the package (this repo uses a src-layout).
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))


def test_model_3d_visualizer_does_not_fallback_to_default_config() -> None:
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    assert "return getDefaultConfigForPath(path)" not in html
    assert "function getDefaultConfigForPath" not in html


def test_model_3d_visualizer_no_implicit_default_model_path() -> None:
    """Ensure no hardcoded default model path fallback (e.g., Qwen3.5-397B)."""
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    # The old implicit fallback that was removed:
    assert "modelPath = 'output/Qwen3.5-397B-A17B-Vitriol-ultra-dummy'" not in html, \
        "Implicit fallback to default model path still present - P0 violation"


def test_model_3d_visualizer_info_missing_path_shows_error() -> None:
    """Ensure missing model path triggers error handling, not fallback."""
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    # Should show BLOCKED state instead of using fallback
    assert "showBlockedIndicator" in html
    assert "window.__VITRIOL_VIZ_MODE__ = 'blocked'" in html


def test_model_3d_visualizer_has_explicit_demo_switch() -> None:
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    assert "demo=1" in html
    assert "isDemoEnabled" in html


def test_model_3d_visualizer_no_hardcoded_397b_params() -> None:
    """Ensure hardcoded ~397B parameter display is removed."""
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    # The old hardcoded pattern:
    assert "${cfg.isMoE ? '~397B'" not in html, \
        "Hardcoded ~397B parameter display still present - P0 violation"


def test_model_3d_visualizer_parameter_display_shows_source() -> None:
    """Ensure parameter display indicates source or availability status."""
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    # New pattern that shows actual params or indicates they're not available:
    assert (
        "parameters (estimated in browser)" in html
        or "Parameters not available" in html
        or "parameters (' + (cfg.paramsSource || 'backend') + ')" in html
    ), \
        "Parameter display should indicate source or availability"


def test_model_3d_visualizer_exposes_weight_stats_provenance_markers() -> None:
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    assert "Params ${_esc(weightStats.params_source)}" in html
    assert "Sample ${Number(weightStats.sampling.sample_size).toLocaleString()}" in html
    assert "Search filters visible modules only" in html
    assert "Header-only metadata" in html
    assert "Displayed slice:" in html


def test_model_3d_visualizer_compare_mode_avoids_placeholder_competitor_data() -> None:
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    assert "No placeholder competitor data is shown" in html
    assert "Only one trustworthy model snapshot is available" in html
    assert "Compare import unavailable" in html
    assert "Qwen3.5-397B-A17B" not in html
    assert "DeepSeek-R1" not in html


def test_model_3d_visualizer_search_has_status_and_clear_affordance() -> None:
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    assert "clearSearchBtn" in html
    assert "searchStatus" in html
    assert "No filter applied." in html
    assert "modules visible" in html


def test_model_3d_visualizer_shell_source_copy_is_explicit() -> None:
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    assert "meta-config.json (restored source)" in html
    assert "meta-config.json (direct source)" in html
    assert "Shell model only (Ultra export)" in html


def test_model_3d_visualizer_detail_export_is_implemented() -> None:
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    assert "function _renderPanelToCanvas(panel)" in html
    assert "function _downloadCanvas(canvas, filename)" in html
    assert "Open a layer or module detail panel before exporting." in html
    assert "vitriol-detail-" in html


def test_model_3d_visualizer_has_toast_feedback_for_copy_and_export() -> None:
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    assert 'id="toastStack"' in html
    assert "function showToast(title, message, level = 'success')" in html
    assert "function copyTextWithFeedback(text, successTitle, successMessage)" in html
    assert "Module config copied" in html
    assert "Model config copied" in html
    assert "JSON report exported" in html
    assert "Compare import unavailable" in html


def test_weight_3d_visualizer_exposes_provenance_markers() -> None:
    html = Path("src/vitriol/viz/weight_3d_visualizer.html").read_text(encoding="utf-8")
    assert "Display Slice" in html
    assert "Sampling" in html
    assert "(${paramsSource})" in html
    assert "synthetic preview" in html
    assert "metadata-only stats" in html


def test_weight_inspector_viz_data_has_p0_metadata(tmp_path: Path) -> None:
    from vitriol.viz.weight_inspector import generate_viz_data

    model_dir = tmp_path / "m"
    model_dir.mkdir()
    cfg = {
        "model_type": "llama",
        "vocab_size": 100,
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "intermediate_size": 64,
        "tie_word_embeddings": False,
    }
    (model_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (model_dir / "meta-config.json").write_text(json.dumps(cfg), encoding="utf-8")

    data = generate_viz_data(str(model_dir), max_layers=2)
    assert "model_total_params" in data
    assert "display_params_estimate" in data
    assert "params_source" in data
    assert "sampling" in data


def test_weight_inspector_sampling_is_deterministic_with_seed() -> None:
    import pytest

    torch = pytest.importorskip("torch")
    from vitriol.viz.weight_inspector import _compute_tensor_stats

    t = torch.arange(0, 1000, dtype=torch.float32).reshape(100, 10)
    a = _compute_tensor_stats(t, seed=42, sample_size=128)
    b = _compute_tensor_stats(t, seed=42, sample_size=128)

    assert a["mean"] == b["mean"]
    assert a["std"] == b["std"]
    assert a["sparsity"] == b["sparsity"]


def test_weight_inspector_sampling_avoids_full_tensor_float32_cast(monkeypatch) -> None:
    import pytest

    torch = pytest.importorskip("torch")
    from vitriol.viz.weight_inspector import _compute_tensor_stats

    tensor = torch.arange(0, 2048, dtype=torch.float16)
    orig_to = torch.Tensor.to
    cast_sizes = []

    def _tracking_to(self, *args, **kwargs):
        dtype = kwargs.get("dtype")
        if dtype is None and args and isinstance(args[0], torch.dtype):
            dtype = args[0]
        if dtype == torch.float32:
            cast_sizes.append(int(self.numel()))
        return orig_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", _tracking_to, raising=False)

    _compute_tensor_stats(tensor, seed=42, sample_size=128)

    assert cast_sizes
    assert max(cast_sizes) <= 128
