from __future__ import annotations


def test_build_generation_config_exposes_security_context_with_provenance(monkeypatch) -> None:
    """
    P4: auditability — build_generation_config must expose security_context (with provenance),
    so jobs/manifests/reports can trace where each security field comes from.
    """
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    from vitriol.config.manager import build_generation_config

    cfg = build_generation_config(
        overrides={
            "trust_remote_code": False,
            "allow_network": False,
            # Do not explicitly set local_files_only; let the resolver infer it.
        }
    )

    sec = cfg.security_context
    assert sec["trust_remote_code"] is False
    assert sec["allow_network"] is False
    assert sec["local_files_only"] is True
    assert sec["provenance"]["trust_remote_code"] == "explicit"
    assert sec["provenance"]["allow_network"] == "explicit"
    # allow_network=False => infer local_files_only=True
    assert sec["provenance"]["local_files_only"] in {"inferred_offline", "explicit"}
