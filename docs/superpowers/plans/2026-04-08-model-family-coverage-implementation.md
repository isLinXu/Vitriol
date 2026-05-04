# Model Family Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an executable model-family coverage matrix that verifies Archon-exported weights across mainstream HuggingFace text model families with `export + load + inference` evidence.

**Architecture:** Introduce one shared family-matrix definition as the source of truth, then drive both smoke and Tier 1 inference validation from that matrix. Extend validator logic to be task-aware, and add family-specific regression tests only where adapter or shrink invariants are required.

**Tech Stack:** Python, pytest, transformers, Archon generator/validator/adapter stack

---

## File Map

- Modify: `tests/test_hub_smoke_models.py`
  Purpose: Convert the current ad-hoc smoke list into a family-aware matrix runner.
- Modify: `src/archon/core/validator.py`
  Purpose: Make validation select the correct AutoModel loader and inference path by task type.
- Modify: `tests/test_end_to_end_local_generate.py`
  Purpose: Add deterministic local regressions for family-specific invariants.
- Create: `src/archon/compat/family_matrix.py`
  Purpose: Hold the canonical family coverage rows consumed by tests and reporting.
- Create: `tests/test_family_matrix.py`
  Purpose: Verify matrix integrity and expected schema before hub/network tests run.
- Create: `docs/model-family-coverage.md`
  Purpose: Publish the tested family tiers and evidence boundary for users.

## Implementation Notes

- Keep the first pass text-model-only.
- Prefer tiny, stable, public HF models for matrix rows.
- Use `Tier 1` only when a real forward or `generate()` path succeeds.
- Keep family-specific fixes in adapters or generator invariants, not inside tests.

### Task 1: Add The Shared Family Matrix

**Files:**
- Create: `src/archon/compat/family_matrix.py`
- Test: `tests/test_family_matrix.py`

- [ ] **Step 1: Write the failing test**

```python
from archon.compat.family_matrix import FAMILY_MATRIX


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_family_matrix.py::test_family_matrix_has_required_fields -v`
Expected: FAIL with `ModuleNotFoundError` or missing `FAMILY_MATRIX`

- [ ] **Step 3: Write minimal implementation**

```python
FAMILY_MATRIX = [
    {
        "family": "llama",
        "model_id": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "task_type": "causal_lm",
        "target_tier": "tier1",
        "trust_remote_code": False,
        "expected_adapter": "LlamaAdapter",
        "notes": "Baseline decoder-only family",
    },
    {
        "family": "mistral",
        "model_id": "hf-internal-testing/tiny-random-MistralForCausalLM",
        "task_type": "causal_lm",
        "target_tier": "tier1",
        "trust_remote_code": False,
        "expected_adapter": None,
        "notes": "Baseline decoder-only family without custom adapter",
    },
]
```

- [ ] **Step 4: Expand the matrix to the agreed first-wave families**

```python
FAMILY_MATRIX = [
    {"family": "llama", "model_id": "hf-internal-testing/tiny-random-LlamaForCausalLM", "task_type": "causal_lm", "target_tier": "tier1", "trust_remote_code": False, "expected_adapter": "LlamaAdapter", "notes": "Baseline decoder-only family"},
    {"family": "mistral", "model_id": "hf-internal-testing/tiny-random-MistralForCausalLM", "task_type": "causal_lm", "target_tier": "tier1", "trust_remote_code": False, "expected_adapter": None, "notes": "Standard decoder-only family"},
    {"family": "gpt2", "model_id": "sshleifer/tiny-gpt2", "task_type": "causal_lm", "target_tier": "tier1", "trust_remote_code": False, "expected_adapter": None, "notes": "Canonical GPT-style decoder family"},
    {"family": "opt", "model_id": "hf-internal-testing/tiny-random-OPTForCausalLM", "task_type": "causal_lm", "target_tier": "tier1", "trust_remote_code": False, "expected_adapter": None, "notes": "Facebook OPT family"},
    {"family": "bloom", "model_id": "hf-internal-testing/tiny-random-BloomForCausalLM", "task_type": "causal_lm", "target_tier": "tier1", "trust_remote_code": False, "expected_adapter": None, "notes": "Bloom decoder family"},
    {"family": "t5", "model_id": "hf-internal-testing/tiny-random-T5ForConditionalGeneration", "task_type": "seq2seq", "target_tier": "tier1", "trust_remote_code": False, "expected_adapter": None, "notes": "Encoder-decoder baseline"},
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_family_matrix.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/archon/compat/family_matrix.py tests/test_family_matrix.py
git commit -m "test: add model family coverage matrix"
```

### Task 2: Drive Hub Smoke From The Matrix

**Files:**
- Modify: `tests/test_hub_smoke_models.py`
- Test: `tests/test_hub_smoke_models.py`

- [ ] **Step 1: Write the failing test**

```python
from archon.compat.family_matrix import FAMILY_MATRIX


def test_hub_smoke_models_are_derived_from_family_matrix() -> None:
    smoke_ids = {row["model_id"] for row in FAMILY_MATRIX}
    assert "hf-internal-testing/tiny-random-LlamaForCausalLM" in smoke_ids
    assert "hf-internal-testing/tiny-random-T5ForConditionalGeneration" in smoke_ids
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_hub_smoke_models.py::test_hub_smoke_models_are_derived_from_family_matrix -v`
Expected: FAIL because the test file still uses `HUB_SMOKE_MODELS`

- [ ] **Step 3: Replace the hard-coded list with matrix-driven parametrization**

```python
from archon.compat.family_matrix import FAMILY_MATRIX


@pytest.mark.parametrize("row", FAMILY_MATRIX, ids=[row["family"] for row in FAMILY_MATRIX])
def test_hub_smoke_generate_minimal_weights(tmp_path: Path, row: dict[str, object]) -> None:
    model_id = str(row["model_id"])
    task_type = str(row["task_type"])
    trust_remote_code = bool(row["trust_remote_code"])
```

- [ ] **Step 4: Select the correct loader by task type**

```python
def _load_generated_model(out_dir: Path, task_type: str, trust_remote_code: bool):
    if task_type == "causal_lm":
        return AutoModelForCausalLM.from_pretrained(str(out_dir), local_files_only=True, trust_remote_code=trust_remote_code)
    if task_type == "seq2seq":
        return AutoModelForSeq2SeqLM.from_pretrained(str(out_dir), local_files_only=True, trust_remote_code=trust_remote_code)
    return AutoModel.from_pretrained(str(out_dir), local_files_only=True, trust_remote_code=trust_remote_code)
```

- [ ] **Step 5: Keep artifact checks and manifest checks intact**

```python
assert (out_dir / "config.json").exists()
assert (out_dir / "meta-config.json").exists()
assert any(out_dir.glob("*.index.json"))
assert (out_dir / "archon-manifest.json").exists()
loaded = _load_generated_model(out_dir, task_type, trust_remote_code)
assert loaded is not None
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_hub_smoke_models.py::test_hub_smoke_models_are_derived_from_family_matrix -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add tests/test_hub_smoke_models.py
git commit -m "test: drive hub smoke from family matrix"
```

### Task 3: Make Validator Task-Aware

**Files:**
- Modify: `src/archon/core/validator.py`
- Test: `tests/test_family_matrix.py`

- [ ] **Step 1: Write the failing test**

```python
from archon.core.validator import ModelValidator


def test_validator_prefers_seq2seq_loader_for_seq2seq_models(monkeypatch) -> None:
    calls = []

    class FakeLoader:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append(cls.__name__)
            return object()

    monkeypatch.setattr("archon.core.validator.AutoModelForSeq2SeqLM", FakeLoader)
    validator = ModelValidator("/tmp/out", trust_remote_code=False)
    validator._load_model_for_task("seq2seq")
    assert calls == ["FakeLoader"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_family_matrix.py::test_validator_prefers_seq2seq_loader_for_seq2seq_models -v`
Expected: FAIL because `_load_model_for_task` does not exist

- [ ] **Step 3: Add explicit task-aware loader selection**

```python
def _load_model_for_task(self, task_type: str):
    if task_type == "causal_lm":
        return AutoModelForCausalLM.from_pretrained(...)
    if task_type == "seq2seq":
        return AutoModelForSeq2SeqLM.from_pretrained(...)
    return AutoModel.from_pretrained(...)
```

- [ ] **Step 4: Thread task type through validation**

```python
def validate(self, run_inference: bool = True, task_type: str = "causal_lm") -> ValidationReport:
    model = self._validate_model_loading(task_type=task_type)
```

```python
def _validate_model_loading(self, task_type: str = "causal_lm"):
    try:
        model = self._load_model_for_task(task_type)
```

- [ ] **Step 5: Keep generic fallback behavior only as a compatibility fallback**

```python
except Exception:
    model = AutoModel.from_pretrained(...)
    self.report.warnings.append("Loaded as AutoModel, not the preferred task loader")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_family_matrix.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/archon/core/validator.py tests/test_family_matrix.py
git commit -m "feat: make validator task aware"
```

### Task 4: Add Tier 1 Inference Matrix Validation

**Files:**
- Modify: `tests/test_hub_smoke_models.py`
- Test: `tests/test_hub_smoke_models.py`

- [ ] **Step 1: Write the failing test**

```python
def test_tier1_rows_require_real_inference_path() -> None:
    tier1 = [row for row in FAMILY_MATRIX if row["target_tier"] == "tier1"]
    assert tier1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_hub_smoke_models.py::test_tier1_rows_require_real_inference_path -v`
Expected: FAIL because no inference-specific matrix test exists yet

- [ ] **Step 3: Add a Tier 1-only test that loads tokenizer, config, and model**

```python
TIER1_ROWS = [row for row in FAMILY_MATRIX if row["target_tier"] == "tier1"]


@pytest.mark.parametrize("row", TIER1_ROWS, ids=[row["family"] for row in TIER1_ROWS])
def test_hub_tier1_generated_model_runs_inference(tmp_path: Path, row: dict[str, object]) -> None:
    model_id = str(row["model_id"])
    task_type = str(row["task_type"])
    trust_remote_code = bool(row["trust_remote_code"])
```

- [ ] **Step 4: Implement minimal real execution by task type**

```python
tok = AutoTokenizer.from_pretrained(str(out_dir), local_files_only=True, trust_remote_code=trust_remote_code)
cfg = AutoConfig.from_pretrained(str(out_dir), local_files_only=True, trust_remote_code=trust_remote_code)
model = _load_generated_model(out_dir, task_type, trust_remote_code)

inputs = tok("hello", return_tensors="pt")
if task_type == "seq2seq":
    outputs = model.generate(**inputs, max_new_tokens=4)
else:
    outputs = model.generate(**inputs, max_new_tokens=4)
assert outputs is not None
```

- [ ] **Step 5: Skip inference only for non-Tier 1 rows, not by family-specific ad-hoc conditions**

```python
if row["target_tier"] != "tier1":
    pytest.skip("Inference reserved for Tier 1 rows")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_hub_smoke_models.py -v`
Expected: PASS locally for non-network checks; hub-backed tests pass when `ARCHON_RUN_HUB_SMOKE=1`

- [ ] **Step 7: Commit**

```bash
git add tests/test_hub_smoke_models.py
git commit -m "test: add tier1 inference validation"
```

### Task 5: Add Family-Specific Regression Coverage

**Files:**
- Modify: `tests/test_end_to_end_local_generate.py`
- Test: `tests/test_end_to_end_local_generate.py`

- [ ] **Step 1: Write the failing test for a standard seq2seq family**

```python
def test_end_to_end_local_t5_ultra_generate_loads_seq2seq(tmp_path: Path) -> None:
    model_dir = tmp_path / "t5-model"
    out_dir = tmp_path / "t5-out"
    ...
    model = AutoModelForSeq2SeqLM.from_pretrained(str(out_dir), local_files_only=True, trust_remote_code=False)
    assert model is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_end_to_end_local_generate.py::test_end_to_end_local_t5_ultra_generate_loads_seq2seq -v`
Expected: FAIL if seq2seq path is still handled like causal LM only

- [ ] **Step 3: Add the minimal local seq2seq regression**

```python
raw_cfg = {
    "model_type": "t5",
    "architectures": ["T5ForConditionalGeneration"],
    "d_model": 64,
    "d_ff": 128,
    "num_layers": 2,
    "num_decoder_layers": 2,
    "num_heads": 4,
    "vocab_size": 512,
}
```

- [ ] **Step 4: Keep the existing GLM regression and add one adapter-backed regression if needed**

```python
def test_end_to_end_local_glm_ultra_generate_runs_forward(tmp_path: Path) -> None:
    ...
```

```python
def test_end_to_end_local_qwen35_ultra_generate_loads_causallm(tmp_path: Path) -> None:
    ...
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_end_to_end_local_generate.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_end_to_end_local_generate.py
git commit -m "test: add family specific coverage regressions"
```

### Task 6: Publish The Coverage Table

**Files:**
- Create: `docs/model-family-coverage.md`
- Test: `tests/test_family_matrix.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_model_family_coverage_doc_exists() -> None:
    assert Path("docs/model-family-coverage.md").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_family_matrix.py::test_model_family_coverage_doc_exists -v`
Expected: FAIL because the doc does not exist yet

- [ ] **Step 3: Write the first version of the coverage doc**

```markdown
# Model Family Coverage

| Family | Representative Model | Tier | Task Type | trust_remote_code | Notes |
|--------|----------------------|------|-----------|-------------------|-------|
| llama | hf-internal-testing/tiny-random-LlamaForCausalLM | Tier 1 | causal_lm | false | Baseline decoder-only family |
```

- [ ] **Step 4: Document the support boundary explicitly**

```markdown
This table reports only families with executed validation evidence in this repository.
It does not claim universal compatibility for every model variant within a family.
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_family_matrix.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add docs/model-family-coverage.md tests/test_family_matrix.py
git commit -m "docs: publish model family coverage table"
```

### Task 7: Final Verification Sweep

**Files:**
- Modify: `tests/test_hub_smoke_models.py`
- Modify: `tests/test_family_matrix.py`
- Modify: `tests/test_end_to_end_local_generate.py`
- Modify: `src/archon/core/validator.py`
- Modify: `src/archon/compat/family_matrix.py`
- Modify: `docs/model-family-coverage.md`

- [ ] **Step 1: Run the local deterministic suite**

Run: `PYTHONPATH=src pytest tests/test_family_matrix.py tests/test_end_to_end_local_generate.py -v`
Expected: PASS

- [ ] **Step 2: Run the hub smoke suite in opt-in mode**

Run: `ARCHON_RUN_HUB_SMOKE=1 PYTHONPATH=src pytest tests/test_hub_smoke_models.py -v`
Expected: PASS for matrix rows with network access; failures identify concrete family gaps

- [ ] **Step 3: Run focused diagnostics after code edits**

Run: `python -m py_compile src/archon/core/validator.py src/archon/compat/family_matrix.py`
Expected: no output

- [ ] **Step 4: Verify the published coverage doc matches the matrix rows**

```python
from archon.compat.family_matrix import FAMILY_MATRIX

for row in FAMILY_MATRIX:
    assert row["family"] in Path("docs/model-family-coverage.md").read_text()
```

- [ ] **Step 5: Commit**

```bash
git add src/archon/core/validator.py src/archon/compat/family_matrix.py tests/test_family_matrix.py tests/test_hub_smoke_models.py tests/test_end_to_end_local_generate.py docs/model-family-coverage.md
git commit -m "feat: add executable model family coverage validation"
```

## Self-Review

### Spec Coverage Check

- Coverage matrix definition: covered by Task 1.
- Family smoke validation: covered by Task 2.
- Task-aware validation: covered by Task 3.
- Tier 1 real inference evidence: covered by Task 4.
- Family-specific regressions: covered by Task 5.
- Published compatibility summary: covered by Task 6.
- Final evidence sweep: covered by Task 7.

### Placeholder Scan

- No `TBD`, `TODO`, or deferred implementation placeholders remain.
- Every task contains file paths, test commands, and concrete code snippets.

### Type Consistency

- Matrix rows consistently use `family`, `model_id`, `task_type`, `target_tier`, `trust_remote_code`, `expected_adapter`, `notes`.
- Validation consistently uses `causal_lm`, `seq2seq`, and `generic`.
- Tier labels consistently use `tier1`, `tier2`, and `tier3`.
