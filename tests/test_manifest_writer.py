"""Smoke tests for vitriol.core.manifest_writer.

Verifies the manifest writer remains importable and exposes its public API
after being extracted from core/generator.py.
"""

from vitriol.core import manifest_writer


def test_manifest_writer_exposes_public_api() -> None:
    assert callable(manifest_writer.write_manifest)


def test_hash_file_returns_none_for_missing_path() -> None:
    assert manifest_writer._hash_file("/nonexistent/path/that/cannot/exist") is None


def test_load_index_data_returns_none_for_missing() -> None:
    assert manifest_writer._load_index_data("/nonexistent/index.json") is None


def test_extract_active_dims_returns_none_when_config_missing() -> None:
    assert manifest_writer._extract_active_dims("/nonexistent/config.json") is None


def test_extract_reconcile_info_returns_none_when_missing() -> None:
    assert manifest_writer._extract_reconcile_info("/nonexistent/reconcile.json") is None


def test_hash_file_hashes_real_file(tmp_path) -> None:
    p = tmp_path / "sample.bin"
    p.write_bytes(b"vitriol")
    digest = manifest_writer._hash_file(str(p))
    assert isinstance(digest, str)
    assert len(digest) == 64  # SHA-256 hex


def test_extract_active_dims_reads_real_config(tmp_path) -> None:
    import json

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "vocab_size": 32000,
        "hidden_size": 256,
        "num_hidden_layers": 2,
        "unrelated_field": "ignored",
    }))
    dims = manifest_writer._extract_active_dims(str(cfg))
    assert dims is not None
    assert dims["vocab_size"] == 32000
    assert dims["hidden_size"] == 256
    assert dims["num_hidden_layers"] == 2
    # Unrelated fields are not present in the projection
    assert "unrelated_field" not in dims


def test_extract_reconcile_info_reads_real_payload(tmp_path) -> None:
    import json

    rec = tmp_path / "vitriol-reconcile.json"
    rec.write_text(json.dumps({
        "patched": True,
        "diff": {"vocab_size": {"before": 0, "after": 32000}, "hidden_size": {"before": 0, "after": 256}},
    }))
    info = manifest_writer._extract_reconcile_info(str(rec))
    assert info == {
        "patched": True,
        "diff_keys": ["hidden_size", "vocab_size"],
        "diff_count": 2,
    }
