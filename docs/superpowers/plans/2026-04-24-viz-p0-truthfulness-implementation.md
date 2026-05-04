# Viz P0 Truthfulness Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复可视化系统 P0 真实性问题：禁止隐式 Demo/硬编码回退、修正参数量字段语义、让采样统计可复现并输出来源元数据。

**Architecture:** 以“来源可追溯”为主线：前端在输入缺失时进入 BLOCK 错误态并仅提供显式 Demo 开关；后端统计输出拆分 `model_total_params` 与 `display_params_estimate` 并携带 `params_source/sampling` 元信息；采样统一引入 seed，默认 42，可 CLI 覆盖。

**Tech Stack:** Python (click/pytest), HTML/JS, torch(可选), vitriol 内部模块

---

## Files touched (expected)

**Modify**
- `src/vitriol/viz/model_3d_visualizer.html`
- `src/vitriol/viz/weight_inspector.py`
- `src/vitriol/visualization/visualizer.py`
- `src/vitriol/cli/commands/weight_viz.py`
- `src/vitriol/cli/commands/viz.py`（`collect_weight_stats` 调用签名同步）

**Create/Modify Tests**
- `tests/test_viz_p0_truthfulness.py`（新增）

---

### Task 1: 3D 模型可视化严格阻断隐式回退（BLOCK + 显式 DEMO）

**Files:**
- Modify: `src/vitriol/viz/model_3d_visualizer.html`
- Test: `tests/test_viz_p0_truthfulness.py`

- [ ] **Step 1: 写一个失败的测试，确保不存在隐式回退到 getDefaultConfigForPath**

```python
from pathlib import Path


def test_model_3d_visualizer_does_not_fallback_to_default_config() -> None:
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    assert "return getDefaultConfigForPath(path)" not in html
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_viz_p0_truthfulness.py::test_model_3d_visualizer_does_not_fallback_to_default_config -q`  
Expected: FAIL（当前代码存在隐式回退）

- [ ] **Step 3: 修改前端：加载失败进入 BLOCK 错误态，并提供显式 demo=1 开关**

实现要点（代码以实际文件为准）：
1) 增加 `isDemoEnabled()`：解析 URL hash（`#?demo=1`）  
2) `loadModelConfig()` 的 catch 分支：
   - 若 demo=1：返回 `getDemoConfig()`（且 demo config 的关键数值不要使用 397B/7B 硬编码）
   - 否则：抛错或返回 `{ blocked: true, error: ... }` 并让 UI 显示 `N/A`
3) UI 增加显式标识：
   - Demo 模式：显示 `DEMO` badge
   - Block 模式：显示 `BLOCKED` 提示（并引导用户检查路径/服务）

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_viz_p0_truthfulness.py::test_model_3d_visualizer_does_not_fallback_to_default_config -q`  
Expected: PASS

- [ ] **Step 5: 补充测试：demo=1 开关存在且为显式入口**

```python
def test_model_3d_visualizer_has_explicit_demo_switch() -> None:
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    assert "demo=1" in html
```

- [ ] **Step 6: Commit**

```bash
git add src/vitriol/viz/model_3d_visualizer.html tests/test_viz_p0_truthfulness.py
git commit -m "fix(viz): block implicit demo fallback in 3d visualizer"
```

---

### Task 2: 权重统计字段语义修正（total_params 不再误导）+ 来源元数据

**Files:**
- Modify: `src/vitriol/viz/weight_inspector.py`
- Modify: `src/vitriol/cli/commands/weight_viz.py`
- Modify: `src/vitriol/cli/commands/viz.py`（调用签名同步）
- Test: `tests/test_viz_p0_truthfulness.py`

- [ ] **Step 1: 写失败测试：generate_viz_data 输出必须包含 P0 字段**

```python
import json
from pathlib import Path

from vitriol.viz.weight_inspector import generate_viz_data


def test_weight_inspector_viz_data_has_p0_metadata(tmp_path: Path) -> None:
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    cfg = {
        "model_type": "llama",
        "vocab_size": 100,
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "intermediate_size": 64,
        "tie_word_embeddings": False,
    }
    (model_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (model_dir / "meta-config.json").write_text(json.dumps(cfg), encoding="utf-8")

    data = generate_viz_data(str(model_dir), max_layers=2)
    assert "model_total_params" in data
    assert "display_params_estimate" in data
    assert "params_source" in data
    assert "sampling" in data
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_viz_p0_truthfulness.py::test_weight_inspector_viz_data_has_p0_metadata -q`  
Expected: FAIL（当前未输出这些字段）

- [ ] **Step 3: 修改 weight_inspector.generate_viz_data 输出**

要求：
1) 新增字段：
   - `model_total_params`
   - `display_params_estimate`
   - `params_source`
   - `sampling`（含 seed/sample_size 等）
2) `total_params` 兼容字段：
   - 若能通过 `ArchitectureAnalyzer` 得到模型总参数量，则 `total_params = model_total_params`
   - 否则 `total_params = display_params_estimate`

- [ ] **Step 4: 同步调用方**

1) `src/vitriol/cli/commands/weight_viz.py`：保持展示使用 `total_params`，但同时可打印 `params_source`  
2) `src/vitriol/cli/commands/viz.py`：`collect_weight_stats()` 调用 `generate_viz_data()` 的签名变更（如新增 seed/sample_size 时需同步）

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/test_viz_p0_truthfulness.py::test_weight_inspector_viz_data_has_p0_metadata -q`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/vitriol/viz/weight_inspector.py src/vitriol/cli/commands/weight_viz.py src/vitriol/cli/commands/viz.py tests/test_viz_p0_truthfulness.py
git commit -m "fix(viz): clarify weight stats param semantics and add provenance metadata"
```

---

### Task 3: 采样可复现（默认 seed=42，可 CLI 覆盖）

**Files:**
- Modify: `src/vitriol/viz/weight_inspector.py`
- Modify: `src/vitriol/visualization/visualizer.py`
- Modify: `src/vitriol/cli/commands/weight_viz.py`
- Test: `tests/test_viz_p0_truthfulness.py`

- [ ] **Step 1: 写失败测试：同 seed 下 stats 可复现（若 torch 可用）**

```python
import pytest


def test_weight_inspector_sampling_is_deterministic_with_seed() -> None:
    torch = pytest.importorskip("torch")
    from vitriol.viz.weight_inspector import _compute_tensor_stats

    t = torch.arange(0, 1000, dtype=torch.float32).reshape(100, 10)
    a = _compute_tensor_stats(t, seed=42, sample_size=128)
    b = _compute_tensor_stats(t, seed=42, sample_size=128)
    assert a["mean"] == b["mean"]
    assert a["std"] == b["std"]
    assert a["sparsity"] == b["sparsity"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_viz_p0_truthfulness.py::test_weight_inspector_sampling_is_deterministic_with_seed -q`  
Expected: FAIL（当前无 seed 入参或采样非确定）

- [ ] **Step 3: 实现确定性采样**

`weight_inspector._compute_tensor_stats()`：
- 参数新增：`seed: int = 42, sample_size: int = 1_000_000`
- 当 `numel > sample_size` 时，使用 `torch.Generator(device="cpu")` + 固定 seed；
- 采样用 `torch.randint(0, numel, (sample_size,), generator=gen)`（允许重复，避免 `randperm` 超大开销）

`visualization/visualizer.py` 的 `_flatten_weights()`：
- 采样同样使用固定 seed（类属性或函数参数），默认 42。

`weight_viz` CLI：
- 新增 `--seed`（默认 42），透传到 `generate_viz_data()`。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_viz_p0_truthfulness.py::test_weight_inspector_sampling_is_deterministic_with_seed -q`  
Expected: PASS（torch 可用时）

- [ ] **Step 5: 跑一遍核心测试集**

Run: `pytest -q`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/vitriol/viz/weight_inspector.py src/vitriol/visualization/visualizer.py src/vitriol/cli/commands/weight_viz.py tests/test_viz_p0_truthfulness.py
git commit -m "fix(viz): make sampling deterministic with seed and expose CLI option"
```

