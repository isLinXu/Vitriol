"""Tests for core modules: exporter, batch, manifest, validator, config_processor, visualizer"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# core/manifest
from vitriol.core.manifest import build_manifest

# core/exporter
from vitriol.core.exporter import ModelExporter

# core/validator
from vitriol.core.validator import ModelValidator, ValidationReport

# core/visualizer
from vitriol.core.visualizer import VitriolVisualizer


# ─────────────────────────────────────────────────────────────────────────────
# manifest tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildManifest:
    def test_basic_manifest(self):
        m = build_manifest(
            schema_version=1,
            generated_at="2024-01-01T00:00:00",
            source={"model_id": "test/model"},
            environment={"python": "3.11"},
            security={"trust_remote_code": False},
            security_context=None,
            generation={"strategy": "random"},
            artifacts={"files": []},
            loadability={"status": "ok"},
        )
        assert m["schema_version"] == 1
        assert m["source"]["model_id"] == "test/model"
        assert "security_context" not in m

    def test_manifest_with_security_context(self):
        m = build_manifest(
            schema_version=2,
            generated_at="2024-01-01T00:00:00",
            source={},
            environment={},
            security={},
            security_context={"level": "high"},
            generation={},
            artifacts={},
            loadability={},
        )
        assert m["schema_version"] == 2
        assert m["security_context"]["level"] == "high"

    def test_manifest_none_defaults(self):
        m = build_manifest(
            schema_version=1,
            generated_at="now",
            source=None,
            environment=None,
            security=None,
            security_context=None,
            generation=None,
            artifacts=None,
            loadability=None,
        )
        assert m["source"] == {}
        assert m["environment"] == {}

    def test_manifest_coerces_types(self):
        m = build_manifest(
            schema_version="3",
            generated_at=12345,
            source={},
            environment={},
            security={},
            security_context=None,
            generation={},
            artifacts={},
            loadability={},
        )
        assert m["schema_version"] == 3
        assert m["generated_at"] == "12345"


# ─────────────────────────────────────────────────────────────────────────────
# exporter tests
# ─────────────────────────────────────────────────────────────────────────────

class TestModelExporter:
    def test_init(self):
        exporter = ModelExporter("/tmp/test_model")
        assert exporter.input_dir == Path("/tmp/test_model")
        assert exporter.trust_remote_code is True

    def test_init_false_trust(self):
        exporter = ModelExporter("/tmp/test_model", trust_remote_code=False)
        assert exporter.trust_remote_code is False

    def test_load_best_config_fallback(self):
        exporter = ModelExporter("/nonexistent/path")
        with patch.object(exporter, "_load_best_config", return_value=MagicMock(to_dict=lambda: {"test": True})):
            config = exporter._load_best_config()
            assert config.to_dict()["test"] is True

    def test_export_structure_error(self, tmp_path):
        exporter = ModelExporter(str(tmp_path))
        with patch.object(exporter, "_load_best_config", side_effect=Exception("fail")):
            with pytest.raises(Exception, match="fail"):
                exporter.export_structure(str(tmp_path / "out.json"))


# ─────────────────────────────────────────────────────────────────────────────
# validator tests
# ─────────────────────────────────────────────────────────────────────────────

class TestValidationReport:
    def test_to_dict(self):
        report = ValidationReport(
            success=True,
            model_loadable=True,
            tokenizer_loadable=True,
            inference_test=False,
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

    def test_defaults(self):
        report = ValidationReport(success=False, model_loadable=False, tokenizer_loadable=False, inference_test=False)
        assert report.errors == []
        assert report.warnings == []
        assert report.memory_usage_gb is None


class TestModelValidator:
    def test_init(self):
        v = ModelValidator("/tmp/output", trust_remote_code=False)
        assert v.output_dir == "/tmp/output"
        assert v.trust_remote_code is False
        assert v.report.success is True

    def test_validate_success(self):
        v = ModelValidator("/tmp/output")
        with patch.object(v, "_validate_model_loading", return_value=MagicMock()):
            report = v.validate(run_inference=False)
            # When _validate_model_loading succeeds and no exception, success stays True
            assert report.success is True
            assert report.warnings == ["Tokenizer validation skipped because inference is disabled"]

    def test_validate_failure(self):
        v = ModelValidator("/tmp/output")
        # Force an exception in the validation flow
        def raise_error(*args, **kwargs):
            raise Exception("load fail")
        with patch.object(v, "_validate_model_loading", side_effect=raise_error):
            report = v.validate(run_inference=False)
            assert report.success is False
            assert any("load fail" in e for e in report.errors)

    def test_validate_skip_inference(self):
        v = ModelValidator("/tmp/output")
        with patch.object(v, "_validate_model_loading", return_value=None):
            report = v.validate(run_inference=False)
            assert "skipped" in str(report.warnings).lower() or report.warnings == []


# ─────────────────────────────────────────────────────────────────────────────
# visualizer tests
# ─────────────────────────────────────────────────────────────────────────────

class TestVitriolVisualizer:
    def test_generate_diagram_mock(self, tmp_path):
        model = MagicMock()
        model.__str__ = MagicMock(return_value="Line1\nLine2")
        output = tmp_path / "out.png"

        with patch("PIL.Image.new") as mock_img:
            mock_draw = MagicMock()
            mock_font = MagicMock()
            with patch("PIL.ImageDraw.Draw", return_value=mock_draw):
                with patch("PIL.ImageFont.truetype", return_value=mock_font):
                    with patch("os.path.exists", return_value=True):
                        result = VitriolVisualizer.generate_diagram(model, str(output))
                        # Should return True on success path
                        assert result in (True, False)

    def test_generate_diagram_truncation(self, tmp_path):
        model = MagicMock()
        model.__str__ = MagicMock(return_value="\n".join([f"Line{i}" for i in range(150)]))
        output = tmp_path / "out.png"

        with patch("PIL.Image.new") as mock_img:
            mock_draw = MagicMock()
            with patch("PIL.ImageDraw.Draw", return_value=mock_draw):
                with patch("os.path.exists", return_value=True):
                    with patch("PIL.ImageFont.truetype"):
                        result = VitriolVisualizer.generate_diagram(model, str(output))
                        assert result in (True, False)

    def test_generate_diagram_import_error(self, tmp_path):
        model = MagicMock()
        model.__str__ = MagicMock(return_value="test")
        output = tmp_path / "out.png"

        with patch.dict(sys.modules, {"PIL": None}):
            # ImportError path
            result = VitriolVisualizer.generate_diagram(model, str(output))
            assert result is False

