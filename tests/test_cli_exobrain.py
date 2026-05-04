"""Tests for vitriol.cli.commands.exobrain module."""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from vitriol.cli.main import cli


class TestExobrainCommandHelp:
    def test_exobrain_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["exobrain", "--help"])
        assert result.exit_code == 0
        assert "exobrain" in result.output.lower()

    def test_exobrain_infer_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["exobrain", "infer", "--help"])
        assert result.exit_code == 0
        assert "infer" in result.output.lower()

    def test_exobrain_distill_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["exobrain", "distill", "--help"])
        assert result.exit_code == 0
        assert "distill" in result.output.lower()


class TestExobrainInferCommand:
    @patch("vitriol.kv.exobrain_inference.ExoBrainInferencePipeline")
    def test_infer_basic(self, mock_pipeline_cls):
        runner = CliRunner()
        mock_result = MagicMock()
        mock_result.prompt = "Hello"
        mock_result.generated_text = "World"
        mock_result.generated_tokens = 1
        mock_result.prompt_tokens = 1
        mock_result.inference_time_s = 0.5
        mock_result.tokens_per_second = 2.0
        mock_result.fusion_mode = "replace"
        mock_result.brain_hit_rate = 0.8
        mock_result.device = "cpu"
        mock_result.error = None

        mock_pipeline = MagicMock()
        mock_pipeline.infer.return_value = mock_result
        mock_pipeline_cls.return_value = mock_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, ["exobrain", "infer", tmpdir, "--prompt", "Hello"])
            assert result.exit_code == 0
            mock_pipeline_cls.assert_called_once()
            mock_pipeline.infer.assert_called_once_with("Hello")

    @patch("vitriol.kv.exobrain_inference.ExoBrainInferencePipeline")
    def test_infer_json_format(self, mock_pipeline_cls):
        runner = CliRunner()
        mock_result = MagicMock()
        mock_result.prompt = "Hello"
        mock_result.generated_text = "World"
        mock_result.generated_tokens = 1
        mock_result.prompt_tokens = 1
        mock_result.inference_time_s = 0.5
        mock_result.tokens_per_second = 2.0
        mock_result.fusion_mode = "replace"
        mock_result.brain_hit_rate = 0.8
        mock_result.device = "cpu"
        mock_result.error = None

        mock_pipeline = MagicMock()
        mock_pipeline.infer.return_value = mock_result
        mock_pipeline_cls.return_value = mock_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, ["exobrain", "infer", tmpdir, "--prompt", "Hello", "--format", "json"])
            assert result.exit_code == 0
            assert "generated_text" in result.output
            assert "World" in result.output

    @patch("vitriol.kv.exobrain_inference.ExoBrainInferencePipeline")
    def test_infer_with_teacher(self, mock_pipeline_cls):
        runner = CliRunner()
        mock_result = MagicMock()
        mock_result.generated_text = "World"
        mock_result.generated_tokens = 1
        mock_result.tokens_per_second = 2.0
        mock_result.brain_hit_rate = 0.8
        mock_result.error = None

        mock_pipeline = MagicMock()
        mock_pipeline.infer.return_value = mock_result
        mock_pipeline_cls.return_value = mock_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, [
                "exobrain", "infer", tmpdir,
                "--teacher", "test/teacher",
                "--prompt", "Hello",
                "--fusion-mode", "residual",
                "--max-new-tokens", "128",
                "--device", "cpu",
                "--dtype", "float32",
                "--retrieval-top-k", "10",
            ])
            assert result.exit_code == 0
            mock_pipeline_cls.assert_called_once()
            call_kwargs = mock_pipeline_cls.call_args.kwargs
            assert call_kwargs["teacher_model_id"] == "test/teacher"
            assert call_kwargs["fusion_mode"] == "residual"
            assert call_kwargs["max_new_tokens"] == 128
            assert call_kwargs["device"] == "cpu"
            assert call_kwargs["retrieval_top_k"] == 10

    @patch("vitriol.kv.exobrain_inference.ExoBrainInferencePipeline")
    def test_infer_with_prompt_file(self, mock_pipeline_cls):
        runner = CliRunner()
        mock_result = MagicMock()
        mock_result.generated_text = "World"
        mock_result.generated_tokens = 1
        mock_result.tokens_per_second = 2.0
        mock_result.brain_hit_rate = 0.8
        mock_result.error = None

        mock_pipeline = MagicMock()
        mock_pipeline.infer.return_value = mock_result
        mock_pipeline_cls.return_value = mock_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "prompt.txt"
            prompt_file.write_text("Hello from file")
            result = runner.invoke(cli, [
                "exobrain", "infer", tmpdir,
                "--prompt", "ignored",
                "--prompt-file", str(prompt_file),
            ])
            assert result.exit_code == 0
            mock_pipeline.infer.assert_called_once_with("Hello from file")

    @patch("vitriol.kv.exobrain_inference.ExoBrainInferencePipeline")
    def test_infer_error_output(self, mock_pipeline_cls):
        runner = CliRunner()
        mock_result = MagicMock()
        mock_result.generated_text = ""
        mock_result.generated_tokens = 0
        mock_result.error = "Inference failed"

        mock_pipeline = MagicMock()
        mock_pipeline.infer.return_value = mock_result
        mock_pipeline_cls.return_value = mock_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, ["exobrain", "infer", tmpdir, "--prompt", "Hello"])
            assert result.exit_code == 0
            assert "Inference failed" in result.output


class TestExobrainDistillCommand:
    @patch("vitriol.kv.exobrain_inference.ExoBrainInferencePipeline")
    @patch("vitriol.kv.exobrain_inference.KnowledgeDistiller")
    def test_distill_basic(self, mock_distiller_cls, mock_pipeline_cls):
        runner = CliRunner()
        mock_result = MagicMock()
        mock_result.num_steps = 3
        mock_result.final_loss = 0.1
        mock_result.parameters_updated = 1000
        mock_result.distill_time_s = 5.0
        mock_result.shell_model_saved = True
        mock_result.loss_history = [0.5, 0.3, 0.1]

        mock_distiller = MagicMock()
        mock_distiller.distill.return_value = mock_result
        mock_distiller_cls.return_value = mock_distiller

        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = str(Path(tmpdir) / "output")
            result = runner.invoke(cli, [
                "exobrain", "distill", tmpdir,
                "--teacher", "test/teacher",
                "--output", out_dir,
            ])
            assert result.exit_code == 0
            mock_distiller_cls.assert_called_once()
            mock_distiller.distill.assert_called_once()

    @patch("vitriol.kv.exobrain_inference.ExoBrainInferencePipeline")
    @patch("vitriol.kv.exobrain_inference.KnowledgeDistiller")
    def test_distill_with_prompts(self, mock_distiller_cls, mock_pipeline_cls):
        runner = CliRunner()
        mock_result = MagicMock()
        mock_result.num_steps = 3
        mock_result.final_loss = 0.1
        mock_result.parameters_updated = 1000
        mock_result.distill_time_s = 5.0
        mock_result.shell_model_saved = True
        mock_result.loss_history = []

        mock_distiller = MagicMock()
        mock_distiller.distill.return_value = mock_result
        mock_distiller_cls.return_value = mock_distiller

        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = str(Path(tmpdir) / "output")
            result = runner.invoke(cli, [
                "exobrain", "distill", tmpdir,
                "--teacher", "test/teacher",
                "--output", out_dir,
                "--prompts", "Hello",
                "--prompts", "World",
            ])
            assert result.exit_code == 0
            call_kwargs = mock_distiller.distill.call_args.kwargs
            assert "Hello" in call_kwargs["prompts"]
            assert "World" in call_kwargs["prompts"]

    @patch("vitriol.kv.exobrain_inference.ExoBrainInferencePipeline")
    @patch("vitriol.kv.exobrain_inference.KnowledgeDistiller")
    def test_distill_with_prompts_file(self, mock_distiller_cls, mock_pipeline_cls):
        runner = CliRunner()
        mock_result = MagicMock()
        mock_result.num_steps = 3
        mock_result.final_loss = 0.1
        mock_result.parameters_updated = 1000
        mock_result.distill_time_s = 5.0
        mock_result.shell_model_saved = True
        mock_result.loss_history = []

        mock_distiller = MagicMock()
        mock_distiller.distill.return_value = mock_result
        mock_distiller_cls.return_value = mock_distiller

        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_file = Path(tmpdir) / "prompts.txt"
            prompts_file.write_text("Hello\nWorld\n")
            out_dir = str(Path(tmpdir) / "output")
            result = runner.invoke(cli, [
                "exobrain", "distill", tmpdir,
                "--teacher", "test/teacher",
                "--output", out_dir,
                "--prompts-file", str(prompts_file),
            ])
            assert result.exit_code == 0
            call_kwargs = mock_distiller.distill.call_args.kwargs
            assert "Hello" in call_kwargs["prompts"]
            assert "World" in call_kwargs["prompts"]

    @patch("vitriol.kv.exobrain_inference.ExoBrainInferencePipeline")
    @patch("vitriol.kv.exobrain_inference.KnowledgeDistiller")
    def test_distill_options(self, mock_distiller_cls, mock_pipeline_cls):
        runner = CliRunner()
        mock_result = MagicMock()
        mock_result.num_steps = 5
        mock_result.final_loss = 0.05
        mock_result.parameters_updated = 2000
        mock_result.distill_time_s = 10.0
        mock_result.shell_model_saved = True
        mock_result.loss_history = []

        mock_distiller = MagicMock()
        mock_distiller.distill.return_value = mock_result
        mock_distiller_cls.return_value = mock_distiller

        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = str(Path(tmpdir) / "output")
            result = runner.invoke(cli, [
                "exobrain", "distill", tmpdir,
                "--teacher", "test/teacher",
                "--output", out_dir,
                "--steps", "5",
                "--lr", "0.001",
                "--loss", "kl",
                "--fusion-mode", "gated",
                "--device", "cuda",
                "--dtype", "float16",
                "--save-format", "pytorch",
                "--gradient-clip", "0.5",
            ])
            assert result.exit_code == 0
            call_kwargs = mock_distiller.distill.call_args.kwargs
            assert call_kwargs["num_steps"] == 5
            assert call_kwargs["learning_rate"] == 0.001
            assert call_kwargs["loss_type"] == "kl"
            assert call_kwargs["save_format"] == "pytorch"
            assert call_kwargs["gradient_clip"] == 0.5

    @patch("vitriol.kv.exobrain_inference.ExoBrainInferencePipeline")
    @patch("vitriol.kv.exobrain_inference.KnowledgeDistiller")
    def test_distill_default_prompts(self, mock_distiller_cls, mock_pipeline_cls):
        runner = CliRunner()
        mock_result = MagicMock()
        mock_result.num_steps = 3
        mock_result.final_loss = 0.1
        mock_result.parameters_updated = 1000
        mock_result.distill_time_s = 5.0
        mock_result.shell_model_saved = True
        mock_result.loss_history = []

        mock_distiller = MagicMock()
        mock_distiller.distill.return_value = mock_result
        mock_distiller_cls.return_value = mock_distiller

        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = str(Path(tmpdir) / "output")
            result = runner.invoke(cli, [
                "exobrain", "distill", tmpdir,
                "--teacher", "test/teacher",
                "--output", out_dir,
            ])
            assert result.exit_code == 0
            call_kwargs = mock_distiller.distill.call_args.kwargs
            assert len(call_kwargs["prompts"]) == 3  # Default prompts
