from __future__ import annotations


def test_hf_loading_sets_offline_env_when_allow_network_false(monkeypatch) -> None:
    """
    End-to-end offline semantics: whenever allow_network=False,
    besides enforcing local_files_only=True, we must also set OFFLINE env vars to prevent
    underlying libraries from "silently going online".
    """
    import os

    # Clear env to avoid cross-test contamination.
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    from vitriol.utils.hf_loading import hf_kwargs

    kwargs = hf_kwargs({"allow_network": False, "local_files_only": False, "trust_remote_code": False})
    assert kwargs["local_files_only"] is True

    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def test_offline_env_cannot_be_bypassed_by_allow_network_true(monkeypatch) -> None:
    """
    A stronger invariant: once the process is marked OFFLINE (via env vars),
    subsequent hf_kwargs calls must not be able to flip local_files_only back to False.
    """

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    from vitriol.utils.hf_loading import hf_kwargs

    kwargs = hf_kwargs({"allow_network": True, "local_files_only": False, "trust_remote_code": False})
    assert kwargs["local_files_only"] is True
