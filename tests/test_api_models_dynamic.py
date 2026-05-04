"""Tests for the dynamic /models API endpoint (M3: Gap Closure).

Verifies that:
1. /models returns dynamically generated data (not hardcoded)
2. /models/families returns evolution tree families
3. /models/adapters returns registered adapters
4. Response structure matches spec in gap-closure-design.md
5. NAS job has os/json imports available (M2 regression guard)
"""

import pytest


# ---------------------------------------------------------------------------
# FastAPI test client fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def api_client():
    """Create a test client for the Vitriol API."""
    from fastapi.testclient import TestClient
    from vitriol.api.server import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# /models — dynamic model listing
# ---------------------------------------------------------------------------

class TestModelsEndpoint:
    """Tests for GET /models."""

    def test_models_returns_200(self, api_client):
        resp = api_client.get("/models")
        assert resp.status_code == 200

    def test_models_has_families_key(self, api_client):
        data = api_client.get("/models").json()
        assert "families" in data
        assert isinstance(data["families"], list)

    def test_models_has_adapters_key(self, api_client):
        data = api_client.get("/models").json()
        assert "adapters" in data
        assert isinstance(data["adapters"], list)

    def test_models_has_strategies_key(self, api_client):
        data = api_client.get("/models").json()
        assert "strategies" in data
        assert isinstance(data["strategies"], list)

    def test_models_has_notes_key(self, api_client):
        data = api_client.get("/models").json()
        assert "notes" in data
        notes = data["notes"]
        assert "source" in notes
        assert "Dynamically" in notes["source"]

    def test_models_has_legacy_models_key(self, api_client):
        """Old 'models' key is kept for backward compatibility."""
        data = api_client.get("/models").json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_models_families_not_empty(self, api_client):
        data = api_client.get("/models").json()
        families = data["families"]
        assert len(families) > 0, "Families should not be empty (DEFAULT_FAMILIES has 16+)"

    def test_family_entry_has_required_fields(self, api_client):
        data = api_client.get("/models").json()
        for fam in data["families"]:
            assert "name" in fam, f"Family missing 'name': {fam}"
            assert "root" in fam, f"Family missing 'root': {fam}"
            assert "members_count" in fam, f"Family missing 'members_count': {fam}"
            assert "members" in fam, f"Family missing 'members': {fam}"

    def test_adapters_include_known_entries(self, api_client):
        """Check that at least LlamaAdapter and DefaultAdapter are present."""
        data = api_client.get("/models").json()
        adapter_names = [a["name"] for a in data["adapters"]]
        assert "LlamaAdapter" in adapter_names, f"LlamaAdapter not found in: {adapter_names}"
        assert "DefaultAdapter" in adapter_names, f"DefaultAdapter not found in: {adapter_names}"

    def test_strategies_include_core_entries(self, api_client):
        """Core strategies should be listed."""
        data = api_client.get("/models").json()
        assert "compact" in data["strategies"]
        assert "ultra" in data["strategies"]
        assert "hybrid_ultra" in data["strategies"]

    def test_known_models_derived_from_fallback_params(self, api_client):
        """Models in the legacy 'models' key come from FALLBACK_PARAMS."""
        data = api_client.get("/models").json()
        model_ids = [m["model_id"] for m in data["models"]]
        # Spot-check a few well-known models
        assert any("llama" in mid.lower() for mid in model_ids), \
            f"No LLaMA model found in: {model_ids[:5]}..."


# ---------------------------------------------------------------------------
# /models/families — family-specific endpoint
# ---------------------------------------------------------------------------

class TestModelsFamiliesEndpoint:
    """Tests for GET /models/families."""

    def test_families_returns_200(self, api_client):
        resp = api_client.get("/models/families")
        assert resp.status_code == 200

    def test_families_has_total(self, api_client):
        data = api_client.get("/models/families").json()
        assert "total" in data
        assert data["total"] >= 16  # DEFAULT_FAMILIES has 16 families

    def test_families_each_has_innovations(self, api_client):
        data = api_client.get("/models/families").json()
        for fam in data["families"]:
            assert "key_innovations" in fam


# ---------------------------------------------------------------------------
# /models/adapters — adapter-specific endpoint
# ---------------------------------------------------------------------------

class TestModelsAdaptersEndpoint:
    """Tests for GET /models/adapters."""

    def test_adapters_returns_200(self, api_client):
        resp = api_client.get("/models/adapters")
        assert resp.status_code == 200

    def test_adapters_has_total(self, api_client):
        data = api_client.get("/models/adapters").json()
        assert "total" in data
        assert data["total"] >= 5  # At minimum: Llama, QwenMoe, Qwen35Moe, DeepSeek, Default

    def test_default_adapter_marked_fallback(self, api_client):
        data = api_client.get("/models/adapters").json()
        default = [a for a in data["adapters"] if a["name"] == "DefaultAdapter"]
        assert len(default) == 1
        assert default[0].get("is_fallback") is True


# ---------------------------------------------------------------------------
# Regression guard: os/json imports in server.py
# ---------------------------------------------------------------------------

class TestServerImports:
    """Ensure server.py has required stdlib imports for NAS job processing."""

    def test_os_import_available(self):
        from vitriol.api import server
        assert hasattr(server, "os"), "server.py must import os for NAS artifacts"

    def test_json_import_available(self):
        from vitriol.api import server
        assert hasattr(server, "json"), "server.py must import json for NAS artifacts"
