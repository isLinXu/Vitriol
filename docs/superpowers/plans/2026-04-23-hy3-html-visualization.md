# Hy3 HTML Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strengthen `hy_v3` interactive HTML visualization so the page clearly exposes Hy3-specific structure in both the top-level overview and the layer/detail browsing surfaces.

**Architecture:** Reuse the existing `HTMLRenderer` entry points and extend them with small Hy3-specific helpers rather than introducing a separate renderer. Keep the analyzer as the source of truth for structure, then use renderer helpers and HTML regression tests to surface `Dense Prefix`, `MoE`, `MTP`, `GQA`, and long-context metadata consistently.

**Tech Stack:** Python, existing `vitriol.arch_viz` renderer stack, pytest

---

### Task 1: Hy3 HTML Summary

**Files:**
- Modify: `src/vitriol/arch_viz/renderers/html.py`
- Test: `tests/test_smoke_vitriol.py`

- [ ] **Step 1: Write the failing test**

```python
assert "Dense Prefix" in html
assert "MoE top-8 / 192" in html
assert "256K Context" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m pytest tests/test_smoke_vitriol.py -k hy3 -v`
Expected: FAIL because the current HTML only shows generic badges and router text.

- [ ] **Step 3: Write minimal implementation**

```python
def _render_hy3_summary(self, arch: Architecture) -> str:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m pytest tests/test_smoke_vitriol.py -k hy3 -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/vitriol/arch_viz/renderers/html.py tests/test_smoke_vitriol.py docs/superpowers/plans/2026-04-23-hy3-html-visualization.md
git commit -m "feat: enhance hy3 html overview"
```

### Task 2: Hy3 Layer Grouping

**Files:**
- Modify: `src/vitriol/arch_viz/renderers/html.py`
- Test: `tests/test_smoke_vitriol.py`

- [ ] **Step 1: Write the failing test**

```python
assert "Dense Prefix · 1 layer" in html
assert "MoE Blocks · 79 layers" in html
assert "MTP Head · 1 layer" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m pytest tests/test_smoke_vitriol.py -k hy3 -v`
Expected: FAIL because the current HTML shows only a generic `Layers` group.

- [ ] **Step 3: Write minimal implementation**

```python
def _render_hy3_layer_groups(self, arch: Architecture) -> str:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m pytest tests/test_smoke_vitriol.py -k hy3 -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/vitriol/arch_viz/renderers/html.py tests/test_smoke_vitriol.py
git commit -m "feat: add hy3 html layer grouping"
```

### Task 3: Re-render And Verify

**Files:**
- Modify: `output/hy3_preview_ultra_final/architecture.html`
- Test: `tests/test_cli_infer.py`, `tests/test_smoke_vitriol.py`, `tests/test_more_regressions.py`

- [ ] **Step 1: Run focused regression suite**

```bash
PYTHONPATH=src python3 -m pytest tests/test_cli_infer.py tests/test_smoke_vitriol.py tests/test_more_regressions.py
```

- [ ] **Step 2: Re-generate Hy3 HTML artifact**

```bash
PYTHONPATH=src python3 -m vitriol.cli.main --trust-remote-code arch-viz output/hy3_preview_ultra_final --html --output output/hy3_preview_ultra_final/architecture.html
```

- [ ] **Step 3: Spot check generated HTML**

```bash
python3 - <<'PY'
from pathlib import Path
html = Path("output/hy3_preview_ultra_final/architecture.html").read_text(encoding="utf-8")
for token in ["Dense Prefix", "MoE Blocks", "MTP Head", "top-8 of 192 active"]:
    assert token in html, token
print("ok")
PY
```

- [ ] **Step 4: Commit**

```bash
git add output/hy3_preview_ultra_final/architecture.html
git commit -m "chore: refresh hy3 html visualization"
```
