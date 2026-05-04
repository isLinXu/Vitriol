"""Tests for vitriol.visualization.visualizer module."""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from vitriol.visualization.visualizer import WeightVisualizer


class TestWeightVisualizer:
    def test_init_default(self):
        wv = WeightVisualizer()
        assert wv.figsize == (12, 8)
        assert wv.seed == 42
        assert wv.sample_size == 1_000_000

    def test_init_custom(self):
        wv = WeightVisualizer(figsize=(8, 6), seed=123, sample_size=5000)
        assert wv.figsize == (8, 6)
        assert wv.seed == 123
        assert wv.sample_size == 5000

    def test_init_bad_style_fallback(self):
        wv = WeightVisualizer(style="nonexistent_style")
        # Should fallback to default
        assert wv is not None

    def test_flatten_weights_empty(self):
        wv = WeightVisualizer()
        result = wv._flatten_weights({})
        assert result.size == 0

    def test_flatten_weights_small(self):
        wv = WeightVisualizer()
        weights = {
            "layer1": torch.randn(10, 10),
        }
        result = wv._flatten_weights(weights)
        assert result.size == 100

    def test_flatten_weights_large_sampling(self):
        wv = WeightVisualizer(sample_size=100)
        weights = {
            "layer1": torch.randn(1000, 1000),
        }
        result = wv._flatten_weights(weights)
        assert result.size == 100  # sampled

    def test_flatten_weights_deterministic(self):
        wv1 = WeightVisualizer(seed=42)
        wv2 = WeightVisualizer(seed=42)
        weights = {"layer1": torch.randn(500, 500)}
        r1 = wv1._flatten_weights(weights)
        r2 = wv2._flatten_weights(weights)
        np.testing.assert_array_equal(r1, r2)

    @patch("matplotlib.pyplot.subplots")
    @patch("seaborn.histplot")
    def test_visualize_weight_distribution(self, mock_histplot, mock_subplots):
        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        wv = WeightVisualizer()
        weights = {"layer1": torch.randn(50, 50)}
        fig = wv.visualize_weight_distribution(weights, title="Test")
        assert fig is not None
        mock_histplot.assert_called_once()
        assert "Test" in mock_ax.set_title.call_args[0][0]

    def test_visualize_weight_distribution_empty(self):
        wv = WeightVisualizer()
        fig = wv.visualize_weight_distribution({})
        assert fig is None

    @patch("matplotlib.pyplot.subplots")
    @patch("seaborn.heatmap")
    def test_visualize_weight_heatmap(self, mock_heatmap, mock_subplots):
        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        wv = WeightVisualizer()
        weights = {"layer1": torch.randn(20, 30)}
        fig = wv.visualize_weight_heatmap(weights)
        assert fig is not None
        mock_heatmap.assert_called_once()

    def test_visualize_weight_heatmap_no_2d(self):
        wv = WeightVisualizer()
        weights = {"bias": torch.randn(10)}
        fig = wv.visualize_weight_heatmap(weights)
        assert fig is None

    @patch("matplotlib.pyplot.subplots")
    def test_visualize_sparsity_pattern(self, mock_subplots):
        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        wv = WeightVisualizer()
        weights = {"layer1": torch.randn(20, 30)}
        fig = wv.visualize_sparsity_pattern(weights)
        assert fig is not None
        mock_ax.imshow.assert_called_once()

    def test_visualize_sparsity_pattern_no_2d(self):
        wv = WeightVisualizer()
        weights = {"bias": torch.randn(10)}
        fig = wv.visualize_sparsity_pattern(weights)
        assert fig is None

    @patch("matplotlib.pyplot.subplots")
    def test_visualize_value_frequency(self, mock_subplots):
        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        wv = WeightVisualizer()
        weights = {"layer1": torch.tensor([1.0, 1.0, 2.0, 3.0, 3.0, 3.0])}
        fig = wv.visualize_value_frequency(weights, top_k=3)
        assert fig is not None
        mock_ax.bar.assert_called_once()

    def test_visualize_value_frequency_empty(self):
        wv = WeightVisualizer()
        fig = wv.visualize_value_frequency({})
        assert fig is None

    @patch("matplotlib.pyplot.subplots")
    @patch("seaborn.barplot")
    @patch("matplotlib.pyplot.tight_layout")
    def test_visualize_statistical_comparison(self, mock_tight, mock_barplot, mock_subplots):
        class MockAxesArray:
            def __init__(self):
                self._grid = [[MagicMock() for _ in range(2)] for _ in range(2)]
            def __getitem__(self, key):
                return self._grid[key[0]][key[1]]

        mock_fig = MagicMock()
        mock_subplots.return_value = (mock_fig, MockAxesArray())

        wv = WeightVisualizer()
        weights_dict = {
            "strategy_a": {"w": torch.randn(10, 10)},
            "strategy_b": {"w": torch.randn(10, 10)},
        }
        fig = wv.visualize_statistical_comparison(weights_dict)
        assert fig is not None
        assert mock_barplot.call_count == 4

    @patch("matplotlib.pyplot.subplots")
    def test_visualize_compression_potential(self, mock_subplots):
        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        wv = WeightVisualizer()
        weights = {"layer1": torch.randn(50, 50)}
        fig = wv.visualize_compression_potential(weights)
        assert fig is not None
        mock_ax.text.assert_called_once()

    def test_visualize_compression_potential_empty(self):
        wv = WeightVisualizer()
        fig = wv.visualize_compression_potential({})
        assert fig is None

    @patch("sklearn.decomposition.PCA")
    def test_visualize_3d_structure(self, mock_pca_cls):
        mock_pca = MagicMock()
        mock_pca.fit_transform.return_value = np.random.randn(20, 3)
        mock_pca_cls.return_value = mock_pca

        wv = WeightVisualizer()
        weights = {"layer1": torch.randn(20, 30)}
        fig = wv.visualize_3d_structure(weights)
        assert fig is not None
        mock_pca_cls.assert_called_once_with(n_components=3)

    def test_visualize_3d_structure_no_valid_layer(self):
        wv = WeightVisualizer()
        weights = {"bias": torch.randn(5)}
        fig = wv.visualize_3d_structure(weights)
        assert fig is None

    def test_generate_comprehensive_report(self):
        wv = WeightVisualizer()
        weights = {"layer1": torch.randn(30, 30)}
        with tempfile.TemporaryDirectory() as tmpdir:
            wv.generate_comprehensive_report(weights, tmpdir)
            # At least some files should be created
            files = list(Path(tmpdir).glob("*.png"))
            assert len(files) >= 3
