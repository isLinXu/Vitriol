"""Tests for plugins/base.py and evolution/tree_visualizer.py."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vitriol.plugins.base import Plugin, PluginManager, get_plugin_manager, init_plugins
from vitriol.evolution.tree_visualizer import TreeVisualizer


# ─────────────────────────────────────────────────────────────
# Mock Plugin for testing
# ─────────────────────────────────────────────────────────────

class MockPlugin(Plugin):
    name = "mock_plugin"
    version = "1.0.0"
    description = "A mock plugin for testing"
    author = "test"

    def initialize(self, context):
        return True


class FailingPlugin(Plugin):
    name = "failing_plugin"

    def initialize(self, context):
        return False


# ─────────────────────────────────────────────────────────────
# Plugin Tests
# ─────────────────────────────────────────────────────────────

class TestPlugin:
    """Tests for Plugin base class."""

    def test_plugin_abstract(self):
        with pytest.raises(TypeError):
            Plugin()

    def test_mock_plugin_creation(self):
        plugin = MockPlugin()
        assert plugin.name == "mock_plugin"
        assert plugin.version == "1.0.0"

    def test_mock_plugin_initialize(self):
        plugin = MockPlugin()
        assert plugin.initialize({}) is True

    def test_plugin_default_methods(self):
        plugin = MockPlugin()
        assert plugin.get_strategies() == {}
        assert plugin.get_adapters() == {}
        assert plugin.get_analyzers() == {}
        assert plugin.get_cli_commands() == {}
        plugin.shutdown()  # should not raise


class TestPluginManager:
    """Tests for PluginManager."""

    def test_creation(self):
        pm = PluginManager()
        assert pm.plugins == {}
        assert pm.hooks == {}

    def test_default_paths(self):
        pm = PluginManager()
        assert len(pm.plugin_paths) >= 3

    def test_discover_plugins_empty(self):
        pm = PluginManager()
        # Temporarily override paths to non-existent dirs
        pm.plugin_paths = [Path("/nonexistent/plugins")]
        discovered = pm.discover_plugins()
        assert discovered == []

    def test_register_and_trigger_hook(self):
        pm = PluginManager()
        calls = []
        def callback(x):
            calls.append(x)
            return x * 2
        pm.register_hook("test_event", callback)
        results = pm.trigger_hook("test_event", 5)
        assert results == [10]
        assert calls == [5]

    def test_trigger_hook_no_callbacks(self):
        pm = PluginManager()
        results = pm.trigger_hook("nonexistent", 1, 2, 3)
        assert results == []

    def test_trigger_hook_exception_handling(self):
        pm = PluginManager()
        def bad_callback(x):
            raise ValueError("oops")
        def good_callback(x):
            return x + 1
        pm.register_hook("event", bad_callback)
        pm.register_hook("event", good_callback)
        results = pm.trigger_hook("event", 5)
        assert results == [6]  # bad_callback skipped, good_callback returned

    def test_load_plugin_no_plugin_class(self):
        pm = PluginManager()
        with patch.dict("sys.modules", {"fake_plugin": MagicMock()}):
            # Mock import_module to return a module with no Plugin subclass
            with patch("importlib.import_module", return_value=MagicMock()):
                result = pm.load_plugin("fake_plugin")
                assert result is None

    def test_unload_plugin(self):
        pm = PluginManager()
        plugin = MockPlugin()
        pm.plugins["mock"] = plugin
        pm.unload_plugin("mock")
        assert "mock" not in pm.plugins

    def test_get_plugin_info(self):
        pm = PluginManager()
        plugin = MockPlugin()
        pm.plugins["mock_plugin"] = plugin
        info = pm.get_plugin_info()
        assert len(info) == 1
        assert info[0]["name"] == "mock_plugin"
        assert info[0]["version"] == "1.0.0"

    def test_load_all_empty(self):
        pm = PluginManager()
        pm.plugin_paths = [Path("/nonexistent")]
        pm.load_all()
        assert pm.plugins == {}


class TestPluginGlobals:
    """Tests for global plugin functions."""

    def test_get_plugin_manager_singleton(self):
        pm1 = get_plugin_manager()
        pm2 = get_plugin_manager()
        assert pm1 is pm2

    def test_init_plugins(self):
        pm = init_plugins()
        assert isinstance(pm, PluginManager)


# ─────────────────────────────────────────────────────────────
# Tree Visualizer Tests
# ─────────────────────────────────────────────────────────────

class MockArchNode:
    """Mock ArchNode for testing."""

    def __init__(self, model_name, family, parent=None, innovations=None):
        self.model_name = model_name
        self.family = family
        self.parent = parent
        self.innovations = innovations or []

    def get_key_params(self):
        return {"params": "1B"}


class MockInnovation:
    def __init__(self, name, description=""):
        self.name = name
        self.description = description


class MockEvolutionTree:
    """Mock EvolutionTree for testing."""

    def __init__(self, nodes):
        self.nodes = nodes

    def get_innovation_timeline(self):
        return [
            {"year": 2023, "family": "Test", "innovation": "i1", "description": "desc"}
        ]


class TestTreeVisualizer:
    """Tests for TreeVisualizer."""

    def test_creation(self):
        tree = MockEvolutionTree({})
        viz = TreeVisualizer(tree)
        assert viz.tree is tree

    def test_generate_html_string(self):
        nodes = {
            "model1": MockArchNode("Model 1", "FamilyA"),
            "model2": MockArchNode("Model 2", "FamilyA", parent="model1"),
        }
        tree = MockEvolutionTree(nodes)
        viz = TreeVisualizer(tree)
        html = viz.generate_html_string(title="Test Tree")
        assert "Test Tree" in html
        assert "familyColors" in html
        assert "treeData" in html

    def test_generate_html_file(self):
        nodes = {
            "model1": MockArchNode("Model 1", "FamilyA"),
        }
        tree = MockEvolutionTree(nodes)
        viz = TreeVisualizer(tree)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tree.html")
            result = viz.generate_html(path, title="Test")
            assert os.path.exists(result)
            with open(result) as f:
                content = f.read()
            assert "Test" in content

    def test_truncate_name(self):
        tree = MockEvolutionTree({})
        viz = TreeVisualizer(tree)
        assert viz._truncate_name("short") == "short"
        long_name = "a" * 30
        truncated = viz._truncate_name(long_name, max_len=20)
        assert len(truncated) <= 20
        assert "..." in truncated

    def test_generate_markdown_report(self):
        nodes = {
            "model1": MockArchNode("Model 1", "FamilyA", innovations=[MockInnovation("i1")]),
            "model2": MockArchNode("Model 2", "FamilyB", parent="model1"),
        }
        tree = MockEvolutionTree(nodes)
        viz = TreeVisualizer(tree)
        md = viz.generate_markdown_report()
        assert "Architecture Evolution Summary" in md
        assert "Model 1" in md
        assert "FamilyA" in md
        assert "Innovation Timeline" in md

    def test_family_colors_in_html(self):
        nodes = {
            "model1": MockArchNode("Model 1", "Qwen"),
            "model2": MockArchNode("Model 2", "LLaMA"),
        }
        tree = MockEvolutionTree(nodes)
        viz = TreeVisualizer(tree)
        html = viz.generate_html_string()
        assert "#06b6d4" in html  # Qwen color
        assert "#f59e0b" in html  # LLaMA color

    def test_empty_tree(self):
        tree = MockEvolutionTree({})
        viz = TreeVisualizer(tree)
        html = viz.generate_html_string()
        assert "0" in html  # total_models = 0
        md = viz.generate_markdown_report()
        assert "Total Models:** 0" in md

    def test_node_with_innovations(self):
        nodes = {
            "model1": MockArchNode(
                "Model 1", "FamilyA",
                innovations=[MockInnovation("Innov1", "Desc1"), MockInnovation("Innov2")]
            ),
        }
        tree = MockEvolutionTree(nodes)
        viz = TreeVisualizer(tree)
        html = viz.generate_html_string()
        assert "Innov1" in html or "innovations" in html.lower()
