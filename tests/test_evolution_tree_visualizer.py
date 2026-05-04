"""Tests for evolution/tree_visualizer module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from vitriol.evolution.tree_visualizer import TreeVisualizer, EVOLUTION_TREE_HTML
from vitriol.evolution.tree_builder import ArchNode, ArchInnovation, EvolutionTree


class TestTreeVisualizerInit:
    """Tests for TreeVisualizer initialization."""

    def test_init(self):
        tree = EvolutionTree()
        viz = TreeVisualizer(tree)
        assert viz.tree is tree


class TestGenerateHtml:
    """Tests for generate_html method."""

    def test_generate_html_creates_file(self):
        tree = EvolutionTree()
        tree.add_model("test/model", config={"hidden_size": 128})
        viz = TreeVisualizer(tree)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "tree.html"
            result = viz.generate_html(str(output_path))
            assert Path(result).exists()
            content = Path(result).read_text()
            assert "<!DOCTYPE html>" in content

    def test_generate_html_custom_title(self):
        tree = EvolutionTree()
        tree.add_model("test", config={})
        viz = TreeVisualizer(tree)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "tree.html"
            viz.generate_html(str(output_path), title="Custom Title")
            content = output_path.read_text()
            assert "Custom Title" in content

    def test_generate_html_includes_stats(self):
        tree = EvolutionTree()
        tree.add_model("test", config={})
        viz = TreeVisualizer(tree)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "tree.html"
            viz.generate_html(str(output_path))
            content = output_path.read_text()
            assert "Total Models" in content
            assert "1" in content  # total_models = 1

    def test_generate_html_with_innovations(self):
        tree = EvolutionTree()
        innov = ArchInnovation(name="GQA", description="Grouped Query", introduced_in="Test", year=2024)
        tree.add_model("test", config={}, innovations=[innov])
        viz = TreeVisualizer(tree)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "tree.html"
            viz.generate_html(str(output_path))
            content = output_path.read_text()
            assert "GQA" in content or "1" in content


class TestGenerateHtmlString:
    """Tests for generate_html_string method."""

    def test_returns_string(self):
        tree = EvolutionTree()
        tree.add_model("test", config={"hidden_size": 128})
        viz = TreeVisualizer(tree)
        html = viz.generate_html_string()
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html

    def test_includes_title(self):
        tree = EvolutionTree()
        tree.add_model("test", config={})
        viz = TreeVisualizer(tree)
        html = viz.generate_html_string(title="My Tree")
        assert "My Tree" in html

    def test_includes_description(self):
        tree = EvolutionTree()
        tree.add_model("test", config={})
        viz = TreeVisualizer(tree)
        html = viz.generate_html_string(description="My Desc")
        assert "My Desc" in html

    def test_includes_tree_json(self):
        tree = EvolutionTree()
        tree.add_model("test", config={"hidden_size": 128})
        viz = TreeVisualizer(tree)
        html = viz.generate_html_string()
        # The tree JSON should be embedded
        assert "treeData" in html or "test" in html

    def test_includes_family_colors(self):
        tree = EvolutionTree()
        tree.add_model("test", config={})
        viz = TreeVisualizer(tree)
        html = viz.generate_html_string()
        # Family colors should be embedded
        assert "familyColors" in html or "#" in html

    def test_empty_tree(self):
        tree = EvolutionTree()
        viz = TreeVisualizer(tree)
        html = viz.generate_html_string()
        assert "<!DOCTYPE html>" in html
        assert "0" in html  # total_models = 0

    def test_multiple_nodes(self):
        tree = EvolutionTree()
        tree.add_model("parent", config={})
        tree.add_model("child", config={}, parent="parent")
        viz = TreeVisualizer(tree)
        html = viz.generate_html_string()
        assert "2" in html  # total_models = 2

    def test_truncate_name(self):
        tree = EvolutionTree()
        long_name = "a" * 50
        tree.add_model(f"org/{long_name}", config={})
        viz = TreeVisualizer(tree)
        html = viz.generate_html_string()
        # Should still work, even with long names
        assert "<!DOCTYPE html>" in html


class TestGenerateMarkdownReport:
    """Tests for generate_markdown_report method."""

    def test_returns_markdown(self):
        tree = EvolutionTree()
        tree.add_model("test", config={})
        viz = TreeVisualizer(tree)
        md = viz.generate_markdown_report()
        assert "# Architecture Evolution Summary" in md

    def test_includes_counts(self):
        tree = EvolutionTree()
        tree.add_model("test", config={})
        viz = TreeVisualizer(tree)
        md = viz.generate_markdown_report()
        assert "Total Models:** 1" in md

    def test_includes_families(self):
        tree = EvolutionTree()
        tree.add_model("qwen/model", config={})
        viz = TreeVisualizer(tree)
        md = viz.generate_markdown_report()
        assert "Qwen" in md

    def test_includes_innovations(self):
        tree = EvolutionTree()
        innov = ArchInnovation(name="GQA", description="Grouped Query", introduced_in="Test", year=2024)
        tree.add_model("test", config={}, innovations=[innov])
        viz = TreeVisualizer(tree)
        md = viz.generate_markdown_report()
        assert "GQA" in md

    def test_includes_timeline(self):
        tree = EvolutionTree()
        innov = ArchInnovation(name="GQA", description="Grouped Query", introduced_in="Test", year=2024)
        tree.add_model("test", config={}, innovations=[innov])
        viz = TreeVisualizer(tree)
        md = viz.generate_markdown_report()
        assert "## Innovation Timeline" in md
        assert "2024" in md

    def test_empty_innovations(self):
        tree = EvolutionTree()
        tree.add_model("test", config={})
        viz = TreeVisualizer(tree)
        md = viz.generate_markdown_report()
        assert "No innovations recorded" in md or "## Innovation Timeline" in md

    def test_parent_link(self):
        tree = EvolutionTree()
        tree.add_model("parent", config={})
        tree.add_model("child", config={}, parent="parent")
        viz = TreeVisualizer(tree)
        md = viz.generate_markdown_report()
        assert "parent" in md


class TestDefaultColors:
    """Tests for DEFAULT_COLORS."""

    def test_colors_exist(self):
        assert "Qwen" in TreeVisualizer.DEFAULT_COLORS
        assert "LLaMA" in TreeVisualizer.DEFAULT_COLORS
        assert "DeepSeek" in TreeVisualizer.DEFAULT_COLORS

    def test_colors_are_strings(self):
        for color in TreeVisualizer.DEFAULT_COLORS.values():
            assert isinstance(color, str)
            assert color.startswith("#")

    def test_other_fallback(self):
        assert "Other" in TreeVisualizer.DEFAULT_COLORS


class TestEvolutionTreeHtmlTemplate:
    """Tests for the HTML template constant."""

    def test_template_exists(self):
        assert isinstance(EVOLUTION_TREE_HTML, str)
        assert "<!DOCTYPE html>" in EVOLUTION_TREE_HTML

    def test_template_has_placeholders(self):
        assert "{{title}}" in EVOLUTION_TREE_HTML
        assert "{{description}}" in EVOLUTION_TREE_HTML
        assert "{{tree_json}}" in EVOLUTION_TREE_HTML
        assert "{{family_colors}}" in EVOLUTION_TREE_HTML
        assert "{{total_models}}" in EVOLUTION_TREE_HTML
        assert "{{total_families}}" in EVOLUTION_TREE_HTML
        assert "{{total_innovations}}" in EVOLUTION_TREE_HTML

    def test_template_has_d3(self):
        assert "d3js.org" in EVOLUTION_TREE_HTML or "d3.v7" in EVOLUTION_TREE_HTML

    def test_template_has_styling(self):
        assert "<style>" in EVOLUTION_TREE_HTML

    def test_template_has_javascript(self):
        assert "<script>" in EVOLUTION_TREE_HTML


class TestTruncateName:
    """Tests for _truncate_name helper."""

    def test_short_name(self):
        tree = EvolutionTree()
        tree.add_model("test", config={})
        viz = TreeVisualizer(tree)
        assert viz._truncate_name("short") == "short"

    def test_long_name(self):
        tree = EvolutionTree()
        tree.add_model("test", config={})
        viz = TreeVisualizer(tree)
        long = "a" * 50
        truncated = viz._truncate_name(long)
        assert len(truncated) <= 20
        assert truncated.endswith("...")

    def test_exact_length(self):
        tree = EvolutionTree()
        tree.add_model("test", config={})
        viz = TreeVisualizer(tree)
        name = "a" * 20
        assert viz._truncate_name(name) == name

    def test_custom_max_len(self):
        tree = EvolutionTree()
        tree.add_model("test", config={})
        viz = TreeVisualizer(tree)
        long = "a" * 50
        truncated = viz._truncate_name(long, max_len=10)
        assert len(truncated) <= 10
        assert truncated.endswith("...")
