from __future__ import annotations


def test_security_context_precedence_and_provenance(monkeypatch) -> None:
    """
    P3: Single Source of Truth.

    Rules (highest → lowest priority):
    1) env OFFLINE (HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE) forces allow_network=False + local_files_only=True
    2) explicit overrides (call-site explicit input)
    3) base/defaults

    Additional requirement: the resolver must return provenance (the source of each field).
    """
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    from vitriol.security.context import resolve_security_context

    ctx = resolve_security_context(
        base={"trust_remote_code": True, "allow_network": True, "local_files_only": False},
        explicit={"trust_remote_code": False},
    )
    assert ctx.trust_remote_code is False
    assert ctx.allow_network is True
    assert ctx.local_files_only is False
    assert ctx.provenance["trust_remote_code"] == "explicit"
    assert ctx.provenance["allow_network"] == "base"

    # env OFFLINE has the highest priority (non-bypassable).
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    ctx2 = resolve_security_context(
        base={"trust_remote_code": True, "allow_network": True, "local_files_only": False},
        explicit={"allow_network": True, "local_files_only": False},
    )
    assert ctx2.allow_network is False
    assert ctx2.local_files_only is True
    assert ctx2.provenance["allow_network"] == "env_offline"
    assert ctx2.provenance["local_files_only"] == "env_offline"


def test_build_generation_config_uses_security_context(monkeypatch) -> None:
    """
    build_generation_config must follow the same SecurityContext semantics (especially env OFFLINE).
    """
    from vitriol.config.manager import build_generation_config

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    cfg = build_generation_config(overrides={"allow_network": True, "local_files_only": False})
    assert cfg.security.allow_network is False
    assert cfg.security.local_files_only is True
