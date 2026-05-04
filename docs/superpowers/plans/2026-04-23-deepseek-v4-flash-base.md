# DeepSeek-V4-Flash-Base Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a complete `DeepSeek-V4-Flash-Base` phase-one support loop in Vitriol: `ultra` export, `validate --no-inference`, and architecture plus weight visualization support.

**Architecture:** Reuse the current DeepSeek family path first, then patch only the first real failure in the correct layer. Keep the proof grounded in real CLI execution, and add only local deterministic regressions for any compatibility fix exposed by the run.

**Tech Stack:** Python, Click CLI commands, Vitriol generator/validator/arch-viz stack, pytest

---

## File Structure

- `docs/superpowers/specs/2026-04-23-deepseek-v4-flash-base-design.md`
  - Approved design spec and scope boundary for this work.
- `src/vitriol/utils/hf_loading.py`
  - Raw/Hugging Face config loading fallback if the upstream V4 config shape is not fully recognized.
- `src/vitriol/core/generator.py`
  - `ultra` export path, shard/index generation, tokenizer save logic, and family-sensitive export handling.
- `src/vitriol/patches/model_family_patches.py`
  - DeepSeek-family config normalization or export-time invariants if the model requires them.
- `src/vitriol/arch_viz/analyzers.py`
  - Family detection and architecture metadata used by `arch-viz`.
- `src/vitriol/arch_viz/parser.py`
  - Config parsing path used by architecture visualization.
- `tests/test_hf_loading.py`
  - Deterministic config loading fallback regressions.
- `tests/test_more_regressions.py`
  - Export-path regressions such as tokenizer metadata, RoPE cleanup, or non-persistent buffer handling.
- `tests/test_smoke_vitriol.py`
  - Lightweight architecture/HTML visualization regressions.

### Task 1: Run The Real Export Path

**Files:**
- Modify: `docs/superpowers/plans/2026-04-23-deepseek-v4-flash-base.md`
- Test: real CLI run only

- [ ] **Step 1: Inspect the upstream config with a dry read**

Run:

```bash
python3 - <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path
import json
cfg = hf_hub_download("deepseek-ai/DeepSeek-V4-Flash-Base", "config.json")
data = json.loads(Path(cfg).read_text(encoding="utf-8"))
for key in ["model_type", "architectures", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "num_experts", "rope_scaling"]:
    print(f"{key}: {data.get(key)}")
PY
```

Expected: prints the primary config surface so the first run is not blind.

- [ ] **Step 2: Attempt the real `ultra` export**

Run:

```bash
PYTHONPATH=src python3 -m vitriol.cli.main --trust-remote-code generate deepseek-ai/DeepSeek-V4-Flash-Base --strategy ultra -o output/deepseek_v4_flash_base_ultra
```

Expected: either the export completes or the command fails with a concrete first compatibility error.

- [ ] **Step 3: Capture the first failure boundary**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
out = Path("output/deepseek_v4_flash_base_ultra")
print("exists:", out.exists())
if out.exists():
    for name in ["config.json", "meta-config.json", "pytorch_model.bin.index.json", "vitriol-manifest.json"]:
        print(name, (out / name).exists())
PY
```

Expected: identify whether the export failed before config save, during sharding, or after writing partial artifacts.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/plans/2026-04-23-deepseek-v4-flash-base.md
git commit -m "docs: add deepseek v4 flash base implementation plan"
```

### Task 2: Patch The First Real Failure

**Files:**
- Modify: `src/vitriol/utils/hf_loading.py`
- Modify: `src/vitriol/core/generator.py`
- Modify: `src/vitriol/patches/model_family_patches.py`
- Modify: `src/vitriol/arch_viz/analyzers.py`
- Modify: `src/vitriol/arch_viz/parser.py`
- Test: `tests/test_hf_loading.py`
- Test: `tests/test_more_regressions.py`
- Test: `tests/test_smoke_vitriol.py`

- [ ] **Step 1: Write the failing regression that matches the real failure**

Example shape if config fallback fails:

```python
def test_load_config_or_raw_handles_deepseek_v4_shape(tmp_path: Path) -> None:
    config_dir = tmp_path / "deepseek_v4"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "architectures": ["DeepseekV4ForCausalLM"],
                "text_config": {
                    "hidden_size": 5120,
                    "num_hidden_layers": 61,
                    "num_attention_heads": 128,
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = load_config_or_raw(
        str(config_dir),
        security={"trust_remote_code": False, "allow_network": False, "local_files_only": True},
    )

    assert loaded.model_type == "deepseek_v4"
    assert loaded.text_config.hidden_size == 5120
```

- [ ] **Step 2: Run the focused failing test**

Run:

```bash
PYTHONPATH=src python3 -m pytest tests/test_hf_loading.py -k deepseek_v4 -v
```

Expected: FAIL on the exact missing compatibility surface.

- [ ] **Step 3: Implement the minimal fix in the correct layer**

Possible minimal shapes:

```python
def load_config_or_raw(...):
    ...
```

```python
class ModelAnalyzerRegistry:
    _analyzers = {
        "deepseek_v4": DeepSeekAnalyzer(),
    }
```

```python
def patch_deepseek_family(...):
    ...
```

The exact code should match the real failure from Task 1 rather than applying all three examples blindly.

- [ ] **Step 4: Re-run the focused regression**

Run:

```bash
PYTHONPATH=src python3 -m pytest tests/test_hf_loading.py tests/test_more_regressions.py tests/test_smoke_vitriol.py -k "deepseek_v4 or deepseek" -v
```

Expected: PASS for the new regression and no breakage in nearby DeepSeek-family tests.

- [ ] **Step 5: Re-run the real export**

Run:

```bash
PYTHONPATH=src python3 -m vitriol.cli.main --trust-remote-code generate deepseek-ai/DeepSeek-V4-Flash-Base --strategy ultra -o output/deepseek_v4_flash_base_ultra
```

Expected: completes successfully and writes the full export artifact set.

- [ ] **Step 6: Commit**

```bash
git add src/vitriol/utils/hf_loading.py src/vitriol/core/generator.py src/vitriol/patches/model_family_patches.py src/vitriol/arch_viz/analyzers.py src/vitriol/arch_viz/parser.py tests/test_hf_loading.py tests/test_more_regressions.py tests/test_smoke_vitriol.py
git commit -m "feat: support deepseek v4 flash base export"
```

### Task 3: Validate And Visualize The Artifact

**Files:**
- Modify: `output/deepseek_v4_flash_base_ultra/architecture.html`
- Modify: `output/deepseek_v4_flash_base_ultra/architecture.png`
- Modify: `output/deepseek_v4_flash_base_ultra/architecture_detail.png`
- Test: real CLI validation only

- [ ] **Step 1: Run no-inference validation**

Run:

```bash
PYTHONPATH=src python3 -m vitriol.cli.main --trust-remote-code validate output/deepseek_v4_flash_base_ultra --no-inference
```

Expected: `Success: True`, `Model Loadable: True`, and `Tokenizer Loadable: True`.

- [ ] **Step 2: Generate static architecture outputs**

Run:

```bash
PYTHONPATH=src python3 -m vitriol.cli.main --trust-remote-code arch-viz output/deepseek_v4_flash_base_ultra --block --output output/deepseek_v4_flash_base_ultra/architecture.png
PYTHONPATH=src python3 -m vitriol.cli.main --trust-remote-code arch-viz output/deepseek_v4_flash_base_ultra --detail --output output/deepseek_v4_flash_base_ultra/architecture_detail.png
PYTHONPATH=src python3 -m vitriol.cli.main --trust-remote-code arch-viz output/deepseek_v4_flash_base_ultra --html --output output/deepseek_v4_flash_base_ultra/architecture.html
```

Expected: all three files are created without parser or analyzer failure.

- [ ] **Step 3: Start and verify `weight-viz`**

Run:

```bash
PYTHONPATH=src python3 -m vitriol.cli.main weight-viz -m output/deepseek_v4_flash_base_ultra --port 8782 --no-open
```

Expected: server starts and reports a local URL plus layer statistics availability.

- [ ] **Step 4: Verify final artifact set**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
out = Path("output/deepseek_v4_flash_base_ultra")
required = [
    "config.json",
    "meta-config.json",
    "pytorch_model.bin.index.json",
    "vitriol-manifest.json",
    "architecture.html",
    "architecture.png",
    "architecture_detail.png",
]
missing = [name for name in required if not (out / name).exists()]
print("missing:", missing)
assert not missing, missing
print("ok")
PY
```

Expected: prints `missing: []` and `ok`.

- [ ] **Step 5: Commit**

```bash
git add output/deepseek_v4_flash_base_ultra
git commit -m "chore: refresh deepseek v4 flash base artifacts"
```

### Task 4: Final Regression Sweep

**Files:**
- Modify: `README.md`
- Modify: `README_CN.md`
- Test: `tests/test_hf_loading.py`
- Test: `tests/test_more_regressions.py`
- Test: `tests/test_smoke_vitriol.py`

- [ ] **Step 1: Run the regression bundle**

Run:

```bash
PYTHONPATH=src python3 -m pytest tests/test_hf_loading.py tests/test_more_regressions.py tests/test_smoke_vitriol.py tests/test_cli_infer.py
```

Expected: PASS, with no DeepSeek-family regressions introduced by the V4 support work.

- [ ] **Step 2: Update docs only if the path required a reusable compatibility fix**

Example text to add:

```md
- `deepseek-ai/DeepSeek-V4-Flash-Base`: verified `ultra` export + `validate --no-inference` + `arch-viz` + `weight-viz`
```

- [ ] **Step 3: Re-run diagnostics on edited files**

Run:

```bash
python3 -m pytest tests/test_hf_loading.py tests/test_more_regressions.py tests/test_smoke_vitriol.py -q
```

Expected: PASS and no newly introduced local regressions.

- [ ] **Step 4: Commit**

```bash
git add README.md README_CN.md tests/test_hf_loading.py tests/test_more_regressions.py tests/test_smoke_vitriol.py
git commit -m "docs: record deepseek v4 flash base support"
```
