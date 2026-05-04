"""Tests for evolution/timeline module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock


from vitriol.evolution.timeline import InnovationTimeline, TimelineEvent


# ─────────────────────────────────────────────────────────────────────────────
# TimelineEvent Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTimelineEvent:
    """Tests for TimelineEvent dataclass."""

    def test_creation(self):
        event = TimelineEvent(
            year=2023,
            month=6,
            innovation="Test Innovation",
            description="A test description",
            model_id="test-model",
            family="test-family",
            impact="high",
        )
        assert event.year == 2023
        assert event.impact == "high"


# ─────────────────────────────────────────────────────────────────────────────
# InnovationTimeline Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestInnovationTimeline:
    """Tests for InnovationTimeline."""

    def _create_mock_tree(self):
        """Create a mock EvolutionTree with minimal data to avoid network calls."""
        mock_node = MagicMock()
        mock_node.family = "test-family"
        innovation = MagicMock()
        innovation.name = "Test Innovation"
        innovation.description = "A test"
        innovation.introduced_in = "test-model"
        innovation.year = 2023
        mock_node.innovations = [innovation]

        mock_tree = MagicMock()
        mock_tree.families = {"test-family": mock_node}
        mock_tree.nodes = {"test-node": mock_node}
        return mock_tree

    def test_init_loads_families(self):
        mock_tree = self._create_mock_tree()
        timeline = InnovationTimeline(evolution_tree=mock_tree)
        assert timeline.tree is mock_tree
        mock_tree.load_builtin_families.assert_called_once()
        mock_tree.build.assert_called_once()

    def test_build_events(self):
        mock_tree = self._create_mock_tree()
        timeline = InnovationTimeline(evolution_tree=mock_tree)
        events = timeline.build_events()
        assert isinstance(events, list)
        assert len(events) == 1
        assert isinstance(events[0], TimelineEvent)
        assert events[0].year == 2023
        assert events[0].impact in ("high", "medium", "low")

    def test_events_sorted_by_year(self):
        mock_tree = self._create_mock_tree()
        # Add another innovation with different year
        node2 = MagicMock()
        node2.family = "family2"
        innovation2 = MagicMock()
        innovation2.name = "Older Innovation"
        innovation2.description = "Older"
        innovation2.introduced_in = "old-model"
        innovation2.year = 2020
        node2.innovations = [innovation2]
        mock_tree.nodes["node2"] = node2

        timeline = InnovationTimeline(evolution_tree=mock_tree)
        events = timeline.build_events()
        years = [e.year for e in events]
        assert years == sorted(years)

    def test_get_innovation_by_year(self):
        mock_tree = self._create_mock_tree()
        timeline = InnovationTimeline(evolution_tree=mock_tree)
        timeline.build_events()
        by_year = timeline.get_innovation_by_year()
        assert isinstance(by_year, dict)
        assert 2023 in by_year
        assert len(by_year[2023]) == 1

    def test_assess_impact_high(self):
        mock_tree = self._create_mock_tree()
        timeline = InnovationTimeline(evolution_tree=mock_tree)
        assert timeline._assess_impact("MoE") == "high"
        assert timeline._assess_impact("GQA") == "high"
        assert timeline._assess_impact("MLA") == "high"
        assert timeline._assess_impact("SSM") == "high"

    def test_assess_impact_medium(self):
        mock_tree = self._create_mock_tree()
        timeline = InnovationTimeline(evolution_tree=mock_tree)
        assert timeline._assess_impact("RoPE") == "medium"
        assert timeline._assess_impact("SwiGLU") == "medium"

    def test_assess_impact_low(self):
        mock_tree = self._create_mock_tree()
        timeline = InnovationTimeline(evolution_tree=mock_tree)
        assert timeline._assess_impact("Some Unknown Innovation") == "low"

    def test_generate_html_structure(self):
        mock_tree = self._create_mock_tree()
        timeline = InnovationTimeline(evolution_tree=mock_tree)
        html = timeline.generate_html()
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert "timeline" in html.lower()

    def test_generate_html_with_events(self):
        mock_tree = self._create_mock_tree()
        timeline = InnovationTimeline(evolution_tree=mock_tree)
        timeline.build_events()
        html = timeline.generate_html(title="Test Timeline")
        assert "Test Timeline" in html
        assert str(len(timeline.events)) in html

    def test_generate_html_without_building_events(self):
        """generate_html should auto-build events if not already built."""
        mock_tree = self._create_mock_tree()
        timeline = InnovationTimeline(evolution_tree=mock_tree)
        html = timeline.generate_html()
        assert len(timeline.events) > 0
        assert "timeline" in html.lower()

    def test_save_html(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_tree = self._create_mock_tree()
            timeline = InnovationTimeline(evolution_tree=mock_tree)
            output_path = Path(tmpdir) / "timeline.html"
            timeline.save_html(str(output_path))
            assert output_path.exists()
            content = output_path.read_text()
            assert "<!DOCTYPE html>" in content

    def test_event_card_css_classes(self):
        mock_tree = self._create_mock_tree()
        timeline = InnovationTimeline(evolution_tree=mock_tree)
        timeline.build_events()
        html = timeline.generate_html()
        # At least one of these should be present
        assert any(cls in html for cls in ["event-card high", "event-card medium", "event-card low"])

    def test_multiple_families_present(self):
        mock_tree = self._create_mock_tree()
        node2 = MagicMock()
        node2.family = "family2"
        innovation2 = MagicMock()
        innovation2.name = "Innovation2"
        innovation2.description = "Desc"
        innovation2.introduced_in = "model2"
        innovation2.year = 2022
        node2.innovations = [innovation2]
        mock_tree.nodes["node2"] = node2

        timeline = InnovationTimeline(evolution_tree=mock_tree)
        timeline.build_events()
        families = {e.family for e in timeline.events}
        assert len(families) == 2

    def test_event_attributes(self):
        mock_tree = self._create_mock_tree()
        timeline = InnovationTimeline(evolution_tree=mock_tree)
        events = timeline.build_events()
        for event in events:
            assert event.innovation
            assert event.model_id
            assert event.family
            assert event.impact in ("high", "medium", "low")
