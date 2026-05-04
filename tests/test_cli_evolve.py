"""Tests for CLI evolve commands."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from click.testing import CliRunner

from vitriol.cli.commands.evolve import (
    evolve_group,
    build_tree,
    compare_models,
    simulate_model,
    list_families,
    show_timeline,
    recommend_arch,
)


class TestEvolveTree:
    """Tests for evolve tree command."""

    @patch("vitriol.cli.commands.evolve.hf_load_config")
    @patch("vitriol.cli.commands.evolve.EvolutionTree")
    @patch("vitriol.cli.commands.evolve.TreeVisualizer")
    def test_tree_default(self, mock_viz_class, mock_tree_class, mock_hf_load):
        runner = CliRunner()
        mock_tree = MagicMock()
        mock_tree.nodes = {"test": MagicMock()}
        mock_tree.families = {"LLaMA": {}}
        mock_tree_class.return_value = mock_tree

        mock_viz = MagicMock()
        mock_viz.generate_html.return_value = "output/evolution_tree.html"
        mock_viz_class.return_value = mock_viz

        result = runner.invoke(evolve_group, ["tree"])
        assert result.exit_code == 0
        assert "Building architecture evolution tree" in result.output
        assert "Evolution tree saved to" in result.output
        mock_tree.build.assert_called_once()
        mock_viz.generate_html.assert_called_once()

    @patch("vitriol.cli.commands.evolve.hf_load_config")
    @patch("vitriol.cli.commands.evolve.EvolutionTree")
    @patch("vitriol.cli.commands.evolve.TreeVisualizer")
    def test_tree_with_models(self, mock_viz_class, mock_tree_class, mock_hf_load):
        runner = CliRunner()
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {"hidden_size": 128}
        mock_hf_load.return_value = mock_config

        mock_tree = MagicMock()
        mock_tree.nodes = {"org/model": MagicMock()}
        mock_tree.families = {"Test": {}}
        mock_tree_class.return_value = mock_tree

        mock_viz = MagicMock()
        mock_viz.generate_html.return_value = "output/tree.html"
        mock_viz_class.return_value = mock_viz

        result = runner.invoke(evolve_group, ["tree", "org/model"])
        assert result.exit_code == 0
        assert "Adding model: org/model" in result.output
        mock_tree.add_model.assert_called_with("org/model", {"hidden_size": 128})

    @patch("vitriol.cli.commands.evolve.hf_load_config")
    @patch("vitriol.cli.commands.evolve.EvolutionTree")
    @patch("vitriol.cli.commands.evolve.TreeVisualizer")
    def test_tree_load_error(self, mock_viz_class, mock_tree_class, mock_hf_load):
        runner = CliRunner()
        mock_hf_load.side_effect = Exception("Load failed")

        mock_tree = MagicMock()
        mock_tree.nodes = {}
        mock_tree.families = {}
        mock_tree_class.return_value = mock_tree

        mock_viz = MagicMock()
        mock_viz.generate_html.return_value = "output/tree.html"
        mock_viz_class.return_value = mock_viz

        result = runner.invoke(evolve_group, ["tree", "bad/model"])
        assert result.exit_code == 0
        assert "Warning: Could not load bad/model" in result.output

    @patch("vitriol.cli.commands.evolve.EvolutionTree")
    @patch("vitriol.cli.commands.evolve.TreeVisualizer")
    def test_tree_no_build(self, mock_viz_class, mock_tree_class):
        runner = CliRunner()
        mock_tree = MagicMock()
        mock_tree.nodes = {}
        mock_tree.families = {}
        mock_tree_class.return_value = mock_tree

        mock_viz = MagicMock()
        mock_viz.generate_html.return_value = "output/tree.html"
        mock_viz_class.return_value = mock_viz

        result = runner.invoke(evolve_group, ["tree", "--no-build"])
        assert result.exit_code == 0
        mock_tree.build.assert_not_called()

    @patch("vitriol.cli.commands.evolve.EvolutionTree")
    @patch("vitriol.cli.commands.evolve.TreeVisualizer")
    def test_tree_custom_output(self, mock_viz_class, mock_tree_class):
        runner = CliRunner()
        mock_tree = MagicMock()
        mock_tree.nodes = {}
        mock_tree.families = {}
        mock_tree_class.return_value = mock_tree

        mock_viz = MagicMock()
        mock_viz.generate_html.return_value = "custom_tree.html"
        mock_viz_class.return_value = mock_viz

        result = runner.invoke(evolve_group, ["tree", "-o", "custom_tree.html"])
        assert result.exit_code == 0
        mock_viz.generate_html.assert_called_once_with("custom_tree.html", title="Architecture Evolution Tree")


class TestEvolveCompare:
    """Tests for evolve compare command."""

    @patch("vitriol.cli.commands.evolve.hf_load_config")
    @patch("vitriol.cli.commands.evolve.ArchComparator")
    @patch("vitriol.cli.commands.evolve.ComparisonReport")
    def test_compare_markdown(self, mock_report_class, mock_comp_class, mock_hf_load):
        runner = CliRunner()
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {"hidden_size": 128}
        mock_hf_load.return_value = mock_config

        mock_result = MagicMock()
        mock_comp = MagicMock()
        mock_comp.compare_from_ids.return_value = mock_result
        mock_comp_class.return_value = mock_comp

        mock_report_class.to_markdown.return_value = "# Markdown Report"

        result = runner.invoke(evolve_group, ["compare", "model1", "model2"])
        assert result.exit_code == 0
        assert "Comparing model1 vs model2" in result.output
        assert "# Markdown Report" in result.output
        mock_report_class.to_markdown.assert_called_once_with(mock_result)

    @patch("vitriol.cli.commands.evolve.hf_load_config")
    @patch("vitriol.cli.commands.evolve.ArchComparator")
    @patch("vitriol.cli.commands.evolve.ComparisonReport")
    def test_compare_json(self, mock_report_class, mock_comp_class, mock_hf_load):
        runner = CliRunner()
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {"hidden_size": 128}
        mock_hf_load.return_value = mock_config

        mock_result = MagicMock()
        mock_comp = MagicMock()
        mock_comp.compare_from_ids.return_value = mock_result
        mock_comp_class.return_value = mock_comp

        mock_report_class.to_json.return_value = '{"model1": "a"}'

        result = runner.invoke(evolve_group, ["compare", "a", "b", "--format", "json"])
        assert result.exit_code == 0
        mock_report_class.to_json.assert_called_once_with(mock_result)

    @patch("vitriol.cli.commands.evolve.hf_load_config")
    @patch("vitriol.cli.commands.evolve.ArchComparator")
    @patch("vitriol.cli.commands.evolve.ComparisonReport")
    def test_compare_html_output_file(self, mock_report_class, mock_comp_class, mock_hf_load):
        runner = CliRunner()
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {"hidden_size": 128}
        mock_hf_load.return_value = mock_config

        mock_result = MagicMock()
        mock_comp = MagicMock()
        mock_comp.compare_from_ids.return_value = mock_result
        mock_comp_class.return_value = mock_comp

        mock_report_class.to_html.return_value = "<html></html>"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            result = runner.invoke(evolve_group, [
                "compare", "a", "b",
                "--format", "html",
                "-o", str(output_path),
            ])
            assert result.exit_code == 0
            assert "Comparison saved to" in result.output
            assert output_path.exists()
            assert output_path.read_text() == "<html></html>"

    @patch("vitriol.cli.commands.evolve.hf_load_config")
    def test_compare_load_error(self, mock_hf_load):
        runner = CliRunner()
        mock_hf_load.side_effect = Exception("Load failed")

        result = runner.invoke(evolve_group, ["compare", "bad1", "bad2"])
        assert result.exit_code == 0
        assert "Error loading models" in result.output


class TestEvolveSimulate:
    """Tests for evolve simulate command."""

    @patch("vitriol.cli.commands.evolve.hf_load_config")
    @patch("vitriol.cli.commands.evolve.ArchSimulator")
    def test_simulate_model(self, mock_sim_class, mock_hf_load):
        runner = CliRunner()
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {"hidden_size": 128, "num_hidden_layers": 2}
        mock_hf_load.return_value = mock_config

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "total_params": 1000000,
            "active_params_per_token": 1000000,
            "vram_full_model": 2.0,
            "vram_inference": 2.5,
            "vram_training": 8.0,
            "flops_per_token": 1000000,
            "tokens_per_second": 10.0,
            "inference_latency_ms": 100.0,
            "params_per_vram": 500000.0,
        }
        mock_sim = MagicMock()
        mock_sim.simulate.return_value = mock_result
        mock_sim_class.return_value = mock_sim

        result = runner.invoke(evolve_group, ["simulate", "test/model"])
        assert result.exit_code == 0
        assert "Simulating test/model" in result.output
        assert "=== Simulation Results ===" in result.output
        assert "10.0" in result.output

    @patch("vitriol.cli.commands.evolve.ArchSimulator")
    def test_simulate_from_config(self, mock_sim_class):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps({"model_type": "custom", "hidden_size": 128}))

            mock_result = MagicMock()
            mock_result.to_dict.return_value = {
                "total_params": 1000000,
                "active_params_per_token": 1000000,
                "vram_full_model": 2.0,
                "vram_inference": 2.5,
                "vram_training": 8.0,
                "flops_per_token": 1000000,
                "tokens_per_second": 10.0,
                "inference_latency_ms": 100.0,
                "params_per_vram": 500000.0,
            }
            mock_sim = MagicMock()
            mock_sim.simulate.return_value = mock_result
            mock_sim_class.return_value = mock_sim

            result = runner.invoke(evolve_group, [
                "simulate", "--config", str(config_path),
            ])
            assert result.exit_code == 0
            assert "custom" in result.output

    def test_simulate_no_args(self):
        runner = CliRunner()
        result = runner.invoke(evolve_group, ["simulate"])
        assert result.exit_code == 0
        assert "Must provide either model ID or config path" in result.output

    @patch("vitriol.cli.commands.evolve.hf_load_config")
    def test_simulate_load_error(self, mock_hf_load):
        runner = CliRunner()
        mock_hf_load.side_effect = Exception("Load failed")

        result = runner.invoke(evolve_group, ["simulate", "bad/model"])
        assert result.exit_code == 0
        assert "Error loading model" in result.output

    @patch("vitriol.cli.commands.evolve.hf_load_config")
    @patch("vitriol.cli.commands.evolve.ArchSimulator")
    def test_simulate_output_file(self, mock_sim_class, mock_hf_load):
        runner = CliRunner()
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {"hidden_size": 128}
        mock_hf_load.return_value = mock_config

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"total_params": 1000}
        mock_sim = MagicMock()
        mock_sim.simulate.return_value = mock_result
        mock_sim_class.return_value = mock_sim

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "result.json"
            result = runner.invoke(evolve_group, [
                "simulate", "test/model", "-o", str(output_path),
            ])
            assert result.exit_code == 0
            assert output_path.exists()
            data = json.loads(output_path.read_text())
            assert data["total_params"] == 1000


class TestEvolveFamilies:
    """Tests for evolve families command."""

    @patch("vitriol.cli.commands.evolve.EvolutionTree")
    def test_list_families(self, mock_tree_class):
        runner = CliRunner()
        mock_tree = MagicMock()
        mock_tree.families = {
            "LLaMA": {"root": "meta-llama/Llama-2-7b", "members": {"a": {}, "b": {}}},
            "Qwen": {"root": "Qwen/Qwen-7B", "members": {"c": {}}},
        }
        mock_tree_class.return_value = mock_tree

        result = runner.invoke(evolve_group, ["families"])
        assert result.exit_code == 0
        assert "Known Model Families" in result.output
        assert "LLaMA" in result.output
        assert "Qwen" in result.output
        assert "meta-llama/Llama-2-7b" in result.output


class TestEvolveTimeline:
    """Tests for evolve timeline command."""

    @patch("vitriol.cli.commands.evolve.InnovationTimeline")
    def test_show_timeline(self, mock_timeline_class):
        runner = CliRunner()
        mock_timeline = MagicMock()
        mock_timeline.events = [
            MagicMock(impact="high", year=2024, innovation="MoE", family="Test"),
            MagicMock(impact="medium", year=2023, innovation="GQA", family="Test"),
        ]
        mock_tree = MagicMock()
        mock_tree.families = {"Test": {}}
        mock_timeline.tree = mock_tree
        mock_timeline_class.return_value = mock_timeline

        result = runner.invoke(evolve_group, ["timeline"])
        assert result.exit_code == 0
        assert "Building innovation timeline" in result.output
        assert "Timeline saved to" in result.output
        assert "Total innovations: 2" in result.output
        mock_timeline.build_events.assert_called_once()
        mock_timeline.save_html.assert_called_once()

    @patch("vitriol.cli.commands.evolve.InnovationTimeline")
    def test_show_timeline_custom_output(self, mock_timeline_class):
        runner = CliRunner()
        mock_timeline = MagicMock()
        mock_timeline.events = []
        mock_tree = MagicMock()
        mock_tree.families = {}
        mock_timeline.tree = mock_tree
        mock_timeline_class.return_value = mock_timeline

        result = runner.invoke(evolve_group, ["timeline", "-o", "custom.html", "--title", "My Timeline"])
        assert result.exit_code == 0
        mock_timeline.save_html.assert_called_once_with("custom.html")


class TestEvolveRecommend:
    """Tests for evolve recommend command."""

    @patch("vitriol.cli.commands.evolve.ArchitectureRecommender")
    @patch("vitriol.cli.commands.evolve.UseCase")
    def test_recommend_default(self, mock_use_case, mock_rec_class):
        runner = CliRunner()
        mock_rec = MagicMock()
        mock_rec.recommend.return_value = [
            MagicMock(
                model_id="model1",
                family="LLaMA",
                params_b=7.0,
                vram_gb=16.0,
                score=95.0,
                match_reasons=["Fast"],
                innovations=["GQA"],
            ),
        ]
        mock_rec_class.return_value = mock_rec

        result = runner.invoke(evolve_group, ["recommend"])
        assert result.exit_code == 0
        assert "Finding best architectures" in result.output
        assert "model1" in result.output
        assert "7.0B" in result.output
        assert "95.0" in result.output

    @patch("vitriol.cli.commands.evolve.ArchitectureRecommender")
    @patch("vitriol.cli.commands.evolve.UseCase")
    def test_recommend_with_options(self, mock_use_case, mock_rec_class):
        runner = CliRunner()
        mock_rec = MagicMock()
        mock_rec.recommend.return_value = [
            MagicMock(
                model_id="model2",
                family="Qwen",
                params_b=3.0,
                vram_gb=8.0,
                score=90.0,
                match_reasons=["Small"],
                innovations=[],
            ),
        ]
        mock_rec_class.return_value = mock_rec

        result = runner.invoke(evolve_group, [
            "recommend",
            "--max-params", "7",
            "--max-vram", "24",
            "--use-case", "code",
            "--prefer-moe",
            "--require-gqa",
            "--families", "Qwen,LLaMA",
        ])
        assert result.exit_code == 0
        mock_rec.recommend.assert_called_once()
        call_kwargs = mock_rec.recommend.call_args.kwargs
        assert call_kwargs["max_params"] == 7.0
        assert call_kwargs["max_vram"] == 24.0
        assert call_kwargs["prefer_moe"] is True
        assert call_kwargs["require_gqa"] is True
        assert call_kwargs["preferred_families"] == ["Qwen", "LLaMA"]

    @patch("vitriol.cli.commands.evolve.ArchitectureRecommender")
    @patch("vitriol.cli.commands.evolve.UseCase")
    def test_recommend_empty(self, mock_use_case, mock_rec_class):
        runner = CliRunner()
        mock_rec = MagicMock()
        mock_rec.recommend.return_value = []
        mock_rec_class.return_value = mock_rec

        result = runner.invoke(evolve_group, ["recommend"])
        assert result.exit_code == 0
        assert "No matching architectures found" in result.output


class TestEvolveRegister:
    """Test for register function."""

    def test_register(self):
        cli_group = MagicMock()
        from vitriol.cli.commands.evolve import register
        register(cli_group)
        cli_group.add_command.assert_called_once_with(evolve_group)
