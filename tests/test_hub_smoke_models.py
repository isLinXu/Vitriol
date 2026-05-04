import json
import os
from pathlib import Path

import pytest
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils.hub import cached_file

from vitriol.compat.family_matrix import FAMILY_MATRIX
from vitriol.config.manager import GenerationConfig, SecurityOptions
from vitriol.core.generator import MinimalWeightGenerator


def hub_smoke_rows() -> list[dict[str, object]]:
    return list(FAMILY_MATRIX)


def tier1_rows() -> list[dict[str, object]]:
    return [row for row in FAMILY_MATRIX if row["target_tier"] == "tier1"]


def _load_generated_model(out_dir: Path, task_type: str, trust_remote_code: bool):
    if task_type == "causal_lm":
        return AutoModelForCausalLM.from_pretrained(
            str(out_dir),
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
    if task_type == "seq2seq":
        return AutoModelForSeq2SeqLM.from_pretrained(
            str(out_dir),
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
    return AutoModel.from_pretrained(
        str(out_dir),
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )


def test_hub_smoke_rows_are_derived_from_family_matrix() -> None:
    rows = hub_smoke_rows()
    assert [row["model_id"] for row in rows] == [row["model_id"] for row in FAMILY_MATRIX]


def test_tier1_rows_are_filtered_from_family_matrix() -> None:
    rows = tier1_rows()
    assert rows
    assert all(row["target_tier"] == "tier1" for row in rows)


@pytest.mark.parametrize("row", hub_smoke_rows(), ids=[str(row["family"]) for row in hub_smoke_rows()])
def test_hub_smoke_generate_minimal_weights(tmp_path: Path, row: dict[str, object]) -> None:
    if os.getenv("VITRIOL_RUN_HUB_SMOKE", "").lower() not in ("1", "true", "yes"):
        pytest.skip("Hub smoke disabled (set VITRIOL_RUN_HUB_SMOKE=1)")

    model_id = str(row["model_id"])
    task_type = str(row["task_type"])
    trust_remote_code = bool(row["trust_remote_code"])

    hub_cfg_path = cached_file(
        model_id,
        "config.json",
        _raise_exceptions_for_missing_entries=False,
        local_files_only=False,
    )
    if not hub_cfg_path:
        pytest.skip(f"Hub config.json not available for {model_id}")

    root = os.getenv("VITRIOL_HUB_SMOKE_OUTPUT", "")
    out_dir = (Path(root) if root else tmp_path) / model_id.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = GenerationConfig(strategy="ultra")
    cfg.security = SecurityOptions(
        trust_remote_code=trust_remote_code,
        allow_network=True,
        local_files_only=False,
    )

    g = MinimalWeightGenerator(
        model_id=model_id,
        output_dir=str(out_dir),
        config=cfg,
        shrink_config=True,
    )
    g.generate()

    assert (out_dir / "config.json").exists()
    assert (out_dir / "meta-config.json").exists()
    assert any(out_dir.glob("*.index.json"))
    assert (out_dir / "architecture.html").exists()
    assert (out_dir / "vitriol-manifest.json").exists()

    meta_cfg = AutoConfig.from_pretrained(str(out_dir), local_files_only=True, trust_remote_code=cfg.security.trust_remote_code)
    assert getattr(meta_cfg, "model_type", None)

    hub_meta = Path(hub_cfg_path).read_text()
    out_meta = (out_dir / "meta-config.json").read_text()
    assert hub_meta == out_meta

    manifest = json.loads((out_dir / "vitriol-manifest.json").read_text())
    assert manifest.get("source", {}).get("meta_config_equals_source_config") is True
    assert manifest.get("loadability", {}).get("checked") is True

    loaded = _load_generated_model(
        out_dir,
        task_type,
        cfg.security.trust_remote_code,
    )
    assert loaded is not None

    manifest = json.loads((out_dir / "vitriol-manifest.json").read_text())
    assert manifest.get("loadability", {}).get("ok") is True


@pytest.mark.parametrize("row", tier1_rows(), ids=[str(row["family"]) for row in tier1_rows()])
def test_hub_tier1_generated_model_runs_inference(tmp_path: Path, row: dict[str, object]) -> None:
    if os.getenv("VITRIOL_RUN_HUB_SMOKE", "").lower() not in ("1", "true", "yes"):
        pytest.skip("Hub smoke disabled (set VITRIOL_RUN_HUB_SMOKE=1)")

    model_id = str(row["model_id"])
    task_type = str(row["task_type"])
    trust_remote_code = bool(row["trust_remote_code"])

    hub_cfg_path = cached_file(
        model_id,
        "config.json",
        _raise_exceptions_for_missing_entries=False,
        local_files_only=False,
    )
    if not hub_cfg_path:
        pytest.skip(f"Hub config.json not available for {model_id}")

    root = os.getenv("VITRIOL_HUB_SMOKE_OUTPUT", "")
    out_dir = (Path(root) if root else tmp_path) / f"{model_id.replace('/', '__')}_tier1"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = GenerationConfig(strategy="ultra")
    cfg.security = SecurityOptions(
        trust_remote_code=trust_remote_code,
        allow_network=True,
        local_files_only=False,
    )

    g = MinimalWeightGenerator(
        model_id=model_id,
        output_dir=str(out_dir),
        config=cfg,
        shrink_config=True,
    )
    g.generate()

    tok = AutoTokenizer.from_pretrained(
        str(out_dir),
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    loaded_cfg = AutoConfig.from_pretrained(
        str(out_dir),
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    assert getattr(loaded_cfg, "model_type", None)

    model = _load_generated_model(out_dir, task_type, trust_remote_code)
    inputs = tok("hello", return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=4)
    assert outputs is not None
