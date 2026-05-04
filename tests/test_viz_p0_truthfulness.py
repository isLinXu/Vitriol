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


def test_model_3d_visualizer_has_explicit_demo_switch() -> None:
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    assert "demo=1" in html
    assert "isDemoEnabled" in html


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
