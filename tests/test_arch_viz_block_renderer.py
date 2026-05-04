"""Tests for vitriol.arch_viz.renderers.block module."""
import tempfile
from unittest.mock import MagicMock, patch


from vitriol.arch_viz.renderers.block import BlockRenderer


class TestBlockRenderer:
    def test_init_default_style(self):
        br = BlockRenderer()
        assert br.style == "default"
        assert "embedding" in br.colors
        assert br.edge_color == "#E0E0E0"

    def test_init_academic_style(self):
        br = BlockRenderer(style="academic")
        assert br.style == "academic"
        assert br.colors["embedding"] == "#F0F0F0"
        assert br.edge_color == "#000000"

    def test_feedforward_color_dense(self):
        br = BlockRenderer()
        arch = MagicMock()
        arch.model_type = "llama"
        arch.special_features = []
        color = br._feedforward_color(arch)
        assert color == br.colors["feedforward_dense"]

    def test_feedforward_color_moe(self):
        br = BlockRenderer()
        arch = MagicMock()
        arch.model_type = "hy_v3"
        arch.special_features = ["MoE"]
        color = br._feedforward_color(arch)
        assert color == br.colors["feedforward_moe"]

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_render_basic(self, mock_close, mock_savefig, mock_subplots):
        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        br = BlockRenderer()
        arch = MagicMock()
        arch.model_type = "llama"
        arch.total_params = 7e9
        arch.total_layers = 32
        arch.special_features = []
        arch.parameters = {"num_layers": 32}
        arch.layers = [
            MagicMock(type="embedding", shape="(32000, 4096)", description=""),
            MagicMock(type="attention", shape="", description="Multi-Head Attention"),
            MagicMock(type="feedforward", shape="", description="11008"),
        ]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            br.render(arch, tmp.name)
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_render_with_mtp(self, mock_close, mock_savefig, mock_subplots):
        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        br = BlockRenderer()
        arch = MagicMock()
        arch.model_type = "hy_v3"
        arch.total_params = 7e9
        arch.total_layers = 32
        arch.special_features = ["GQA"]
        arch.parameters = {"num_layers": 32, "mtp_layers": 2}
        arch.layers = [
            MagicMock(type="embedding", shape="(32000, 4096)", description=""),
            MagicMock(type="attention", shape="", description="GQA Attention"),
            MagicMock(type="feedforward", shape="", description="11008"),
        ]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            br.render(arch, tmp.name)
            mock_savefig.assert_called_once()

    def test_draw_feature_badges_hy_v3(self):
        br = BlockRenderer()
        arch = MagicMock()
        arch.model_type = "hy_v3"
        arch.parameters = {
            "dense_prefix_layers": 2,
            "num_experts": 8,
            "top_k_experts": 2,
            "mtp_layers": 1,
        }
        arch.special_features = ["GQA"]
        ax = MagicMock()
        br._draw_feature_badges(ax, arch, x_center=0.5, y=10.0)
        assert ax.text.call_count >= 4  # Dense Prefix, MoE, GQA, MTP badges

    def test_draw_feature_badges_empty(self):
        br = BlockRenderer()
        arch = MagicMock()
        arch.model_type = "llama"
        arch.parameters = {}
        arch.special_features = []
        ax = MagicMock()
        br._draw_feature_badges(ax, arch, x_center=0.5, y=10.0)
        ax.text.assert_not_called()

    def test_draw_hy3_legend(self):
        br = BlockRenderer()
        arch = MagicMock()
        arch.model_type = "hy_v3"
        arch.parameters = {
            "dense_prefix_layers": 3,
            "num_layers": 32,
            "mtp_layers": 1,
        }
        ax = MagicMock()
        br._draw_hy3_legend(ax, arch, x=1.0, y=10.0)
        assert ax.text.call_count >= 3

    def test_draw_hy3_legend_non_hy_v3(self):
        br = BlockRenderer()
        arch = MagicMock()
        arch.model_type = "llama"
        ax = MagicMock()
        br._draw_hy3_legend(ax, arch, x=1.0, y=10.0)
        ax.text.assert_not_called()

    def test_draw_box(self):
        br = BlockRenderer()
        ax = MagicMock()
        br._draw_box(ax, 0.5, 10.0, 1.2, 0.6, "Test\nSub", "#FFF", "#333", "#E0E0E0")
        assert ax.add_patch.call_count == 2  # shadow + box
        assert ax.text.call_count == 2  # main_text + sub_text

    def test_draw_residual(self):
        br = BlockRenderer()
        ax = MagicMock()
        br._draw_residual(ax, 10.0, 8.0, 1.2)
        assert ax.plot.call_count == 3  # out + vertical + in
        assert ax.add_patch.call_count == 1  # circle
        assert ax.text.call_count == 2  # Residual label + "+"
