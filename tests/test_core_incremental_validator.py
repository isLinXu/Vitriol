"""
Tests for vitriol.core.incremental and vitriol.core.validator modules.
"""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from vitriol.core.incremental import IncrementalGenerator
from vitriol.core.validator import ValidationReport, ModelValidator


# ─────────────────────────────────────────────────────────────
# IncrementalGenerator
# ─────────────────────────────────────────────────────────────

class TestIncrementalGenerator:
    @pytest.fixture
    def tmp_output_dir(self, tmp_path):
        return str(tmp_path / "output")

    def test_init_sets_paths(self, tmp_output_dir):
        gen = IncrementalGenerator(tmp_output_dir)
        assert gen.output_dir == tmp_output_dir
        assert gen.checkpoint_file == Path(tmp_output_dir) / ".vitriol_checkpoint.json"

    def test_save_and_load_checkpoint(self, tmp_output_dir):
        gen = IncrementalGenerator(tmp_output_dir)
        state = {"progress": 0.5, "layer": 16}
        gen.save_checkpoint(state)

        loaded = gen.load_checkpoint()
        assert loaded == state

    def test_load_checkpoint_missing_returns_none(self, tmp_output_dir):
        gen = IncrementalGenerator(tmp_output_dir)
        assert gen.load_checkpoint() is None

    def test_clear_checkpoint_removes_file(self, tmp_output_dir):
        gen = IncrementalGenerator(tmp_output_dir)
        gen.save_checkpoint({"done": True})
        assert gen.checkpoint_file.exists()

        gen.clear_checkpoint()
        assert not gen.checkpoint_file.exists()

    def test_clear_checkpoint_no_file_does_not_crash(self, tmp_output_dir):
        gen = IncrementalGenerator(tmp_output_dir)
        gen.clear_checkpoint()  # Should not raise

    def test_save_checkpoint_logs_warning_on_failure(self, tmp_output_dir, caplog):
        gen = IncrementalGenerator(tmp_output_dir)
        gen.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        # Create a directory with the same name as checkpoint file so open() fails
        gen.checkpoint_file.mkdir(parents=True, exist_ok=True)

        with caplog.at_level("WARNING"):
            gen.save_checkpoint({"test": 1})
        assert "Failed to save checkpoint" in caplog.text

    def test_load_checkpoint_logs_warning_on_bad_json(self, tmp_output_dir, caplog):
        gen = IncrementalGenerator(tmp_output_dir)
        gen.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        gen.checkpoint_file.write_text("not-json{")

        with caplog.at_level("WARNING"):
            result = gen.load_checkpoint()
        assert result is None
        assert "Failed to load checkpoint" in caplog.text

    def test_overwrite_checkpoint(self, tmp_output_dir):
        gen = IncrementalGenerator(tmp_output_dir)
        gen.save_checkpoint({"v": 1})
        gen.save_checkpoint({"v": 2})

        loaded = gen.load_checkpoint()
        assert loaded == {"v": 2}


# ─────────────────────────────────────────────────────────────
# ValidationReport
# ─────────────────────────────────────────────────────────────

class TestValidationReport:
    def test_dataclass_defaults(self):
        report = ValidationReport(
            success=True,
            model_loadable=False,
            tokenizer_loadable=False,
            inference_test=False,
        )
        assert report.success is True
        assert report.memory_usage_gb is None
        assert report.errors == []
        assert report.warnings == []

    def test_to_dict(self):
        report = ValidationReport(
            success=True,
            model_loadable=True,
            tokenizer_loadable=True,
            inference_test=True,
            memory_usage_gb=1.5,
            errors=["e1"],
            warnings=["w1"],
        )
        d = report.to_dict()
        assert d["success"] is True
        assert d["model_loadable"] is True
        assert d["memory_usage_gb"] == 1.5
        assert d["errors"] == ["e1"]
        assert d["warnings"] == ["w1"]

    def test_to_dict_with_none_memory(self):
        report = ValidationReport(
            success=False,
            model_loadable=False,
            tokenizer_loadable=False,
            inference_test=False,
        )
        d = report.to_dict()
        assert d["memory_usage_gb"] is None


# ─────────────────────────────────────────────────────────────
# ModelValidator
# ─────────────────────────────────────────────────────────────

class TestModelValidator:
    def setup_method(self):
        # Re-import to survive module eviction by other tests
        import vitriol.core.validator as _v
        self.ModelValidator = _v.ModelValidator

    def test_init_defaults(self):
        validator = self.ModelValidator("/output")
        assert validator.output_dir == "/output"
        assert validator.trust_remote_code is False
        assert validator.report.success is True
        assert validator.report.model_loadable is False

    def test_init_custom_flags(self):
        validator = self.ModelValidator("/output", trust_remote_code=False)
        assert validator.trust_remote_code is False

    @patch("vitriol.core.validator.hf_load_causallm")
    @patch("vitriol.core.validator.hf_load_tokenizer")
    def test_validate_success(self, mock_load_tokenizer, mock_load_causallm):
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.parameters.return_value = []
        mock_model.buffers.return_value = []
        mock_load_causallm.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": MagicMock()}
        mock_load_tokenizer.return_value = mock_tokenizer

        validator = self.ModelValidator("/output")
        report = validator.validate(run_inference=True)

        assert report.model_loadable is True
        assert report.tokenizer_loadable is True
        assert report.success is True

    @patch("vitriol.core.validator.hf_load_causallm")
    def test_validate_model_load_falls_back_to_generic(self, mock_load_causallm):
        mock_load_causallm.side_effect = RuntimeError("size mismatch")

        with patch("vitriol.core.validator.hf_load_model") as mock_load_model:
            mock_model = MagicMock()
            mock_model.device = "cpu"
            mock_model.parameters.return_value = []
            mock_model.buffers.return_value = []
            mock_load_model.return_value = mock_model

            validator = self.ModelValidator("/output")
            report = validator.validate(run_inference=False)

            assert report.model_loadable is True
            assert "AutoModel" in report.warnings[0]

    @patch("vitriol.core.validator.hf_load_causallm")
    def test_validate_model_load_complete_failure(self, mock_load_causallm):
        mock_load_causallm.side_effect = Exception("complete failure")

        with patch("vitriol.core.validator.hf_load_model") as mock_load_model:
            mock_load_model.side_effect = Exception("also fails")

            validator = self.ModelValidator("/output")
            report = validator.validate(run_inference=False)

            assert report.model_loadable is False
            assert report.success is False
            assert len(report.errors) > 0

    @patch("vitriol.core.validator.hf_load_causallm")
    @patch("vitriol.core.validator.hf_load_tokenizer")
    def test_validate_tokenizer_failure(self, mock_load_tokenizer, mock_load_causallm):
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.parameters.return_value = []
        mock_model.buffers.return_value = []
        mock_load_causallm.return_value = mock_model

        mock_load_tokenizer.side_effect = Exception("tokenizer missing")

        validator = self.ModelValidator("/output")
        report = validator.validate(run_inference=True)

        assert report.model_loadable is True
        assert report.tokenizer_loadable is False
        assert report.success is False

    @patch("vitriol.core.validator.hf_load_causallm")
    @patch("vitriol.core.validator.hf_load_tokenizer")
    def test_validate_inference_with_generate(self, mock_load_tokenizer, mock_load_causallm):
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.parameters.return_value = []
        mock_model.buffers.return_value = []
        mock_model.generate.return_value = MagicMock()
        mock_load_causallm.return_value = mock_model

        mock_tok_out = MagicMock()
        mock_tok_out.to.return_value = mock_tok_out
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = mock_tok_out
        mock_load_tokenizer.return_value = mock_tokenizer

        validator = self.ModelValidator("/output")
        report = validator.validate(run_inference=True)

        assert report.inference_test is True
        assert report.success is True

    @patch("vitriol.core.validator.hf_load_causallm")
    @patch("vitriol.core.validator.hf_load_tokenizer")
    def test_validate_inference_without_generate(self, mock_load_tokenizer, mock_load_causallm):
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.parameters.return_value = []
        mock_model.buffers.return_value = []
        del mock_model.generate  # No generate method
        mock_load_causallm.return_value = mock_model

        mock_tok_out = MagicMock()
        mock_tok_out.to.return_value = mock_tok_out
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = mock_tok_out
        mock_load_tokenizer.return_value = mock_tokenizer

        validator = self.ModelValidator("/output")
        report = validator.validate(run_inference=True)

        assert report.inference_test is True

    @patch("vitriol.core.validator.hf_load_causallm")
    def test_validate_run_inference_false_skips_tokenizer(self, mock_load_causallm):
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.parameters.return_value = []
        mock_model.buffers.return_value = []
        mock_load_causallm.return_value = mock_model

        validator = self.ModelValidator("/output")
        report = validator.validate(run_inference=False)

        assert report.tokenizer_loadable is False
        assert "skipped" in report.warnings[0].lower()

    @patch("vitriol.core.validator.hf_load_causallm")
    def test_check_memory_usage(self, mock_load_causallm):
        import torch

        mock_model = MagicMock()
        mock_model.device = "cpu"
        # Create actual tensors for param_size calculation
        p1 = torch.randn(100)
        p2 = torch.randn(50)
        mock_model.parameters.return_value = [p1, p2]
        mock_model.buffers.return_value = []
        mock_load_causallm.return_value = mock_model

        validator = self.ModelValidator("/output")
        report = validator.validate(run_inference=False)

        expected_size = (100 * 4 + 50 * 4) / (1024 ** 3)
        assert report.memory_usage_gb is not None
        assert abs(report.memory_usage_gb - expected_size) < 1e-6

    def test_load_model_for_task_seq2seq(self):
        validator = self.ModelValidator("/output")

        with patch("transformers.AutoModelForSeq2SeqLM") as mock_seq2seq:
            mock_model = MagicMock()
            mock_seq2seq.from_pretrained.return_value = mock_model

            result = validator._load_model_for_task("seq2seq")
            assert result is mock_model

    def test_load_model_for_task_generic(self):
        validator = self.ModelValidator("/output")

        with patch("vitriol.core.validator.hf_load_model") as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model

            result = validator._load_model_for_task("generic")
            assert result is mock_model

    @patch("vitriol.core.validator.hf_load_causallm")
    def test_load_model_for_task_size_mismatch_retry(self, mock_load_causallm):
        mock_load_causallm.side_effect = [
            RuntimeError("size mismatch"),
            MagicMock(),
        ]

        validator = self.ModelValidator("/output")
        result = validator._load_model_for_task("causal_lm")
        assert result is not None
        assert mock_load_causallm.call_count == 2

    @patch("vitriol.core.validator.hf_load_causallm")
    def test_load_model_for_task_oom_retry(self, mock_load_causallm):
        mock_load_causallm.side_effect = [
            MemoryError("out of memory"),
            MagicMock(),
        ]

        validator = self.ModelValidator("/output")
        result = validator._load_model_for_task("causal_lm")
        assert result is not None
        assert mock_load_causallm.call_count == 2

    @patch("vitriol.core.validator.hf_load_causallm")
    def test_load_model_for_task_non_memory_error_reraises(self, mock_load_causallm):
        mock_load_causallm.side_effect = OSError("disk error")

        validator = self.ModelValidator("/output")
        with pytest.raises(OSError):
            validator._load_model_for_task("causal_lm")
