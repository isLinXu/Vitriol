from __future__ import annotations


def test_build_manifest_includes_security_context_and_provenance() -> None:
    """
    P5: artifact-level traceability. The manifest must carry security_context (with provenance),
    not just a flat trust_remote_code/allow_network/local_files_only set.
    """
    from vitriol.core.manifest import build_manifest

    manifest = build_manifest(
        schema_version=2,
        generated_at="2026-01-01T00:00:00Z",
        source={"model_id": "demo/model"},
        environment={"python": "3.10.0"},
        security={"trust_remote_code": False, "allow_network": False, "local_files_only": True},
        security_context={
            "trust_remote_code": False,
            "allow_network": False,
            "local_files_only": True,
            "provenance": {"allow_network": "explicit", "local_files_only": "inferred_offline"},
        },
        generation={"strategy": "compact"},
        artifacts={},
        loadability={"checked": False},
    )

    assert manifest["schema_version"] == 2
    assert manifest["security_context"]["allow_network"] is False
    assert manifest["security_context"]["provenance"]["local_files_only"] == "inferred_offline"
