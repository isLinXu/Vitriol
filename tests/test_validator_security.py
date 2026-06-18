"""Validator security propagation tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vitriol.core.validator import ModelValidator


class TestValidatorSecurityPropagation:
    def test_seq2seq_load_passes_trust_remote_code(self) -> None:
        validator = ModelValidator("/tmp/model", trust_remote_code=True)
        mock_model = MagicMock()

        with patch("vitriol.core.validator.AutoModelForSeq2SeqLM", create=True) as mock_cls:
            mock_cls.from_pretrained.return_value = mock_model
            model = validator._load_model_for_task("seq2seq")

        assert model is mock_model
        _, kwargs = mock_cls.from_pretrained.call_args
        assert kwargs["trust_remote_code"] is True
        assert kwargs["local_files_only"] is True

    def test_security_context_defaults_false(self) -> None:
        validator = ModelValidator("/tmp/model")
        ctx = validator._security_context()
        assert ctx["trust_remote_code"] is False
        assert ctx["allow_network"] is False
        assert ctx["local_files_only"] is True
