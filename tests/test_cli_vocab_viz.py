"""Tests for vitriol.cli.commands.vocab_viz module."""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from vitriol.cli.main import cli

# Vocabulary visualization mock tests require plotly (part of [viz] optional dependency)
# to be importable for patch targets. Skip the mocked tests if unavailable.
_viz_available: bool = True
try:
    import plotly  # noqa: F401
except ImportError:
    _viz_available = False


class TestVocabVizCommandHelp:
    def test_vocab_viz_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["vocab-viz", "--help"])
        assert result.exit_code == 0
        assert "vocab" in result.output.lower()


class TestVocabVizLoadLocalTokenizer:
    def test_load_vocab_from_local_tokenizer_files(self):
        from vitriol.cli.commands.vocab_viz import _load_vocab_from_local_tokenizer_files
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer = {
                "model": {
                    "vocab": {"hello": 0, "world": 1}
                }
            }
            tok_path = Path(tmpdir) / "tokenizer.json"
            tok_path.write_text(json.dumps(tokenizer))

            vocab, special = _load_vocab_from_local_tokenizer_files(tmpdir)
            assert vocab == {"hello": 0, "world": 1}
            assert isinstance(special, set)

    def test_load_vocab_missing_tokenizer_json(self):
        from vitriol.cli.commands.vocab_viz import _load_vocab_from_local_tokenizer_files
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                _load_vocab_from_local_tokenizer_files(tmpdir)

    def test_load_vocab_missing_vocab_dict(self):
        from vitriol.cli.commands.vocab_viz import _load_vocab_from_local_tokenizer_files
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer = {"model": {}}
            tok_path = Path(tmpdir) / "tokenizer.json"
            tok_path.write_text(json.dumps(tokenizer))

            with pytest.raises(ValueError):
                _load_vocab_from_local_tokenizer_files(tmpdir)

    def test_load_vocab_with_special_tokens(self):
        from vitriol.cli.commands.vocab_viz import _load_vocab_from_local_tokenizer_files
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer = {
                "model": {
                    "vocab": {"<s>": 0, "hello": 1}
                }
            }
            tok_path = Path(tmpdir) / "tokenizer.json"
            tok_path.write_text(json.dumps(tokenizer))

            config = {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "additional_special_tokens": ["[SPECIAL]"],
            }
            config_path = Path(tmpdir) / "tokenizer_config.json"
            config_path.write_text(json.dumps(config))

            vocab, special = _load_vocab_from_local_tokenizer_files(tmpdir)
            assert "<s>" in special
            assert "</s>" in special
            assert "[SPECIAL]" in special


@pytest.mark.skipif(not _viz_available, reason="plotly not installed ([viz] extra)")
class TestVocabVizCommandMocked:
    @patch("vitriol.vocab_viz.core.VocabVisualizer")
    def test_vocab_viz_treemap(self, mock_viz_cls):
        runner = CliRunner()
        mock_viz = MagicMock()
        mock_viz.generate_treemap.return_value = "output.html"
        mock_viz_cls.return_value = mock_viz

        result = runner.invoke(cli, ["vocab-viz"])
        assert result.exit_code == 0
        mock_viz.generate_treemap.assert_called_once()

    @patch("vitriol.vocab_viz.core.VocabVisualizer")
    def test_vocab_viz_bar(self, mock_viz_cls):
        runner = CliRunner()
        mock_viz = MagicMock()
        mock_viz.generate_bar_chart.return_value = "output.html"
        mock_viz_cls.return_value = mock_viz

        result = runner.invoke(cli, ["vocab-viz", "--type", "bar"])
        assert result.exit_code == 0
        mock_viz.generate_bar_chart.assert_called_once()

    @patch("vitriol.vocab_viz.core.VocabVisualizer")
    def test_vocab_viz_single(self, mock_viz_cls):
        runner = CliRunner()
        mock_viz = MagicMock()
        mock_viz.generate_single_distribution.return_value = "output.html"
        mock_viz_cls.return_value = mock_viz

        result = runner.invoke(cli, ["vocab-viz", "--type", "single", "--model-id", "test/model"])
        assert result.exit_code == 0
        mock_viz.generate_single_distribution.assert_called_once()

    @patch("vitriol.vocab_viz.core.VocabVisualizer")
    def test_vocab_viz_with_model_id(self, mock_viz_cls):
        runner = CliRunner()
        mock_viz = MagicMock()
        mock_viz.generate_treemap.return_value = "output.html"
        mock_viz_cls.return_value = mock_viz

        result = runner.invoke(cli, ["vocab-viz", "--model-id", "test/model"])
        assert result.exit_code == 0
        mock_viz.add_model_from_id.assert_called_once_with("test/model", family="Custom")

    @patch("vitriol.vocab_viz.core.VocabVisualizer")
    def test_vocab_viz_single_no_model_id(self, mock_viz_cls):
        runner = CliRunner()
        mock_viz = MagicMock()
        mock_viz_cls.return_value = mock_viz

        result = runner.invoke(cli, ["vocab-viz", "--type", "single"])
        assert result.exit_code == 0
        # Should log error but not crash
        mock_viz.generate_single_distribution.assert_not_called()

    @patch("vitriol.vocab_viz.core.VocabVisualizer")
    def test_vocab_viz_custom_output(self, mock_viz_cls):
        runner = CliRunner()
        mock_viz = MagicMock()
        mock_viz.generate_treemap.return_value = "custom.html"
        mock_viz_cls.return_value = mock_viz

        result = runner.invoke(cli, ["vocab-viz", "--output", "custom.html"])
        assert result.exit_code == 0
        mock_viz.generate_treemap.assert_called_once_with("custom.html")


@pytest.mark.skipif(not _viz_available, reason="plotly not installed ([viz] extra)")
class TestVocabVizMissingDependency:
    @patch("vitriol.vocab_viz.core.VocabVisualizer", side_effect=ModuleNotFoundError("No module named 'plotly'"))
    def test_vocab_viz_missing_dependency(self, mock_viz_cls):
        runner = CliRunner()
        result = runner.invoke(cli, ["vocab-viz"])
        assert result.exit_code != 0
        assert isinstance(result.exception, Exception)


class TestVocabViz3DMode:
    @patch("vitriol.cli.commands.vocab_viz.serve_3d_vocab")
    @patch("vitriol.cli.commands.vocab_viz._load_vocab_from_local_tokenizer_files")
    def test_vocab_viz_3d_local(self, mock_load_local, mock_serve):
        from vitriol.cli.commands.vocab_viz import vocab_viz
        runner = CliRunner()
        mock_load_local.return_value = (
            {"hello": 0, "world": 1},
            {"hello"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer = {
                "model": {
                    "vocab": {"hello": 0, "world": 1}
                }
            }
            tok_path = Path(tmpdir) / "tokenizer.json"
            tok_path.write_text(json.dumps(tokenizer))

            result = runner.invoke(cli, ["vocab-viz", "--3d", "--model-id", tmpdir])
            assert result.exit_code == 0
            mock_load_local.assert_called_once()
            mock_serve.assert_called_once()

    def test_vocab_viz_3d_no_model_id(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["vocab-viz", "--3d"])
        assert result.exit_code == 0
        # Error is logged, not printed to output
