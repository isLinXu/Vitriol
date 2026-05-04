from vitriol.compat.family_matrix import FAMILY_MATRIX
from vitriol.core.validator import ModelValidator


def test_family_matrix_has_required_fields() -> None:
    required = {
        "family",
        "model_id",
        "task_type",
        "target_tier",
        "trust_remote_code",
        "expected_adapter",
        "notes",
    }

    assert FAMILY_MATRIX
    for row in FAMILY_MATRIX:
        assert required.issubset(row.keys())
        assert row["task_type"] in {"causal_lm", "seq2seq", "generic"}
        assert row["target_tier"] in {"tier1", "tier2", "tier3"}


def test_validator_prefers_seq2seq_loader_for_seq2seq_models(monkeypatch) -> None:
    from vitriol.core.validator import ModelValidator
    calls: list[str] = []

    class FakeSeq2SeqLoader:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            calls.append(cls.__name__)
            return object()

    monkeypatch.setattr("vitriol.core.validator.AutoModelForSeq2SeqLM", FakeSeq2SeqLoader)
    validator = ModelValidator("/tmp/out", trust_remote_code=False)
    validator._load_model_for_task("seq2seq")
    assert calls == ["FakeSeq2SeqLoader"]


def test_validator_no_inference_skips_tokenizer_loading(monkeypatch) -> None:
    validator = ModelValidator("/tmp/out", trust_remote_code=False)

    monkeypatch.setattr(validator, "_validate_model_loading", lambda task_type="causal_lm": object())

    def fail_if_called():
        raise AssertionError("tokenizer should not be required when inference is disabled")

    monkeypatch.setattr(validator, "_validate_tokenizer_loading", fail_if_called)

    report = validator.validate(run_inference=False)

    assert report.success is True
    assert report.tokenizer_loadable is False
    assert report.inference_test is False
    assert any("Tokenizer validation skipped" in warning for warning in report.warnings)


def test_model_family_coverage_doc_exists() -> None:
    from pathlib import Path

    assert Path("docs/model-family-coverage.md").exists()
