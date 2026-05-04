"""Tests for tools/comparator module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock


from vitriol.tools.comparator import format_number, format_params, ModelComparator


# ─────────────────────────────────────────────────────────────────────────────
# Utility Function Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatParams:
    """Tests for format_params helper."""

    def test_billions(self):
        assert format_params(7_200_000_000) == "7.20B"
        assert format_params(1_000_000_000) == "1.00B"

    def test_millions(self):
        assert format_params(500_000_000) == "500.00M"
        assert format_params(1_000_000) == "1.00M"

    def test_thousands(self):
        assert format_params(50_000) == "50.00K"
        assert format_params(1_000) == "1.00K"

    def test_small_numbers(self):
        assert format_params(500) == "500"
        assert format_params(0) == "0"


class TestFormatNumber:
    """Tests for format_number helper."""

    def test_large_number(self):
        assert format_number(4096) == "4,096"
        assert format_number(1_000_000) == "1,000,000"

    def test_zero(self):
        assert format_number(0) == "0"


# ─────────────────────────────────────────────────────────────────────────────
# ModelComparator Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestModelComparator:
    """Tests for ModelComparator."""

    def _create_mock_architecture(self, model_type="test", total_params=7e9, total_layers=32):
        """Create a mock Architecture object."""
        arch = MagicMock()
        arch.model_type = model_type
        arch.arch_type = "transformer"
        arch.total_params = total_params
        arch.total_layers = total_layers
        arch.parameters = {
            "hidden_size": 4096,
            "num_heads": 32,
            "num_kv_heads": 8,
            "vocab_size": 32000,
        }
        arch.features = ["RoPE", "GQA"]
        return arch

    def test_init(self):
        comparator = ModelComparator(["model1", "model2"])
        assert comparator.model_ids == ["model1", "model2"]
        assert comparator.analyses == {}

    def test_compare_table_empty(self):
        comparator = ModelComparator(["model1", "model2"])
        table = comparator.compare_table()
        assert table is not None
        # Table should have model columns even without analyses

    def test_compare_table_with_analyses(self):
        comparator = ModelComparator(["org/model1", "org/model2"])
        comparator.analyses["org/model1"] = self._create_mock_architecture("llama", 7e9, 32)
        comparator.analyses["org/model2"] = self._create_mock_architecture("qwen", 14e9, 48)

        table = comparator.compare_table()
        assert table is not None

    def test_compare_memory_footprint(self):
        comparator = ModelComparator(["org/model1"])
        comparator.analyses["org/model1"] = self._create_mock_architecture(total_params=7e9)

        table = comparator.compare_memory_footprint()
        assert table is not None
        # 7B params * 2 bytes = 14GB FP16

    def test_compare_memory_footprint_empty(self):
        comparator = ModelComparator(["model1"])
        table = comparator.compare_memory_footprint()
        assert table is not None
        # Should be empty but not crash

    def test_generate_diff_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            comparator = ModelComparator(["org/model1", "org/model2"])
            comparator.analyses["org/model1"] = self._create_mock_architecture("llama", 7e9, 32)
            comparator.analyses["org/model2"] = self._create_mock_architecture("qwen", 14e9, 48)

            output_path = Path(tmpdir) / "report.md"
            comparator.generate_diff_report(str(output_path))

            assert output_path.exists()
            content = output_path.read_text()
            assert "Model Architecture Comparison Report" in content
            assert "org/model1" in content
            assert "org/model2" in content

    def test_generate_diff_report_same_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            comparator = ModelComparator(["org/model1", "org/model2"])
            # Both models have same hidden_size
            arch1 = self._create_mock_architecture("llama", 7e9, 32)
            arch2 = self._create_mock_architecture("qwen", 14e9, 48)
            arch2.parameters = arch1.parameters.copy()
            comparator.analyses["org/model1"] = arch1
            comparator.analyses["org/model2"] = arch2

            output_path = Path(tmpdir) / "report.md"
            comparator.generate_diff_report(str(output_path))

            content = output_path.read_text()
            assert "All models have Hidden Size = 4096" in content

    def test_print_summary_no_crash(self):
        comparator = ModelComparator(["org/model1"])
        comparator.analyses["org/model1"] = self._create_mock_architecture()
        # Should not raise
        comparator.print_summary()

    def test_short_name_extraction(self):
        comparator = ModelComparator(["org/model1", "model2"])
        comparator.analyses["org/model1"] = self._create_mock_architecture()
        comparator.analyses["model2"] = self._create_mock_architecture()

        table = comparator.compare_table()
        # Column headers should use short names (last part after /)
        assert "model1" in str(table.columns)
        assert "model2" in str(table.columns)
