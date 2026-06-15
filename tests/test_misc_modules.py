"""Tests for remaining modules: vocab_viz, webui, models_legacy, resilience."""

import pytest

from vitriol.resilience.checkpoint import CheckpointManager

# vocab_viz requires plotly (optional [viz] extra); skip gracefully when absent
try:
    from vitriol.vocab_viz.core import VocabVisualizer
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False


# ─────────────────────────────────────────────────────────────────────────────
# vocab_viz tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
class TestVocabVisualizer:
    def test_init(self):
        viz = VocabVisualizer()
        assert viz is not None
        assert len(viz.models) > 0

    def test_init_custom_models(self):
        custom = [{"model": "test", "vocab": 1000, "family": "Test"}]
        viz = VocabVisualizer(models=custom)
        assert viz.models == custom

    def test_generate_treemap(self, tmp_path):
        viz = VocabVisualizer()
        output = tmp_path / "treemap.html"
        result = viz.generate_treemap(str(output))
        assert result == str(output)
        assert output.exists()

    def test_generate_bar_chart(self, tmp_path):
        viz = VocabVisualizer()
        output = tmp_path / "bar.html"
        result = viz.generate_bar_chart(str(output))
        assert result == str(output)
        assert output.exists()


# ─────────────────────────────────────────────────────────────────────────────
# resilience checkpoint tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckpointManager:
    def test_init(self, tmp_path):
        cm = CheckpointManager(str(tmp_path))
        assert cm.checkpoint_dir == tmp_path

    def test_create_and_load(self, tmp_path):
        cm = CheckpointManager(str(tmp_path))
        state = {"epoch": 5, "loss": 0.1}
        ckpt = cm.create_checkpoint("test_op", state)
        assert ckpt.operation == "test_op"
        assert ckpt.state == state

        loaded = cm.load_checkpoint(ckpt.id)
        assert loaded is not None
        assert loaded.state == state

    def test_load_missing(self, tmp_path):
        cm = CheckpointManager(str(tmp_path))
        loaded = cm.load_checkpoint("nonexistent")
        assert loaded is None

    def test_list_checkpoints(self, tmp_path):
        cm = CheckpointManager(str(tmp_path))
        cm.create_checkpoint("op1", {"a": 1})
        cm.create_checkpoint("op2", {"b": 2})

        ckpts = cm.list_checkpoints()
        assert len(ckpts) == 2

    def test_delete_checkpoint(self, tmp_path):
        cm = CheckpointManager(str(tmp_path))
        ckpt = cm.create_checkpoint("to_delete", {"a": 1})
        assert len(cm.list_checkpoints()) == 1

        cm.delete_checkpoint(ckpt.id)
        assert len(cm.list_checkpoints()) == 0

    def test_delete_nonexistent(self, tmp_path):
        cm = CheckpointManager(str(tmp_path))
        cm.delete_checkpoint("nonexistent")  # Should not raise

