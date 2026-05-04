# Attention Lens UI 专业化重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将当前 3D trace 回放页面的 UI/动效/排版重构为“更专业更美观”的折中风格：控件集中到底部 Playback Bar，分析集中到右侧 Lens Drawer（Tokens/Attention/Meta），画布区域保持干净；并保持现有 trace 回放能力不回归。

**Architecture:** 在 `model_3d_visualizer.html` 中做“布局层重构”：把散落的 fixed 面板合并为两大区域（底部 bar + 右侧 drawer），新增统一的 Design Tokens（颜色/间距/圆角/阴影/字体）；并对 attention lens（热力/连线/热条）做视觉一致化（同一色阶、同一线宽策略、同一动画节奏）。

**Tech Stack:** HTML/CSS/JS（现有单文件）、pytest（结构性回归测试）、本地 trace demo

---

## 0) 文件清单

**Modify:**
- `src/vitriol/viz/model_3d_visualizer.html`
- `tests/test_viz_trace_injection_markers.py`（新增结构断言，防止 UI 回归到“散落面板”）

**Outputs (verification):**
- `verification/tinyllama_hybrid_ultra_trace/trace_3d.html`（更新后的 demo）

---

## Task 1: 建立 UI Design Tokens（颜色/字体/阴影/圆角）并给关键容器挂统一 class

**Files:**
- Modify: `src/vitriol/viz/model_3d_visualizer.html`
- Test: `tests/test_viz_trace_injection_markers.py`

- [ ] **Step 1: 写 failing test：要求存在 design token 变量与关键容器 id**

Append to `tests/test_viz_trace_injection_markers.py`:

```python
from pathlib import Path


def test_ui_polish_has_design_tokens_and_new_layout_ids() -> None:
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    assert "--ui-panel" in html
    assert "--ui-stroke" in html
    assert 'id="lensDrawer"' in html
    assert 'id="playbackBar"' in html
```

- [ ] **Step 2: 运行测试确认失败**

Run:
```bash
PYTHONPATH=src python -m pytest -q tests/test_viz_trace_injection_markers.py::test_ui_polish_has_design_tokens_and_new_layout_ids
```
Expected: FAIL

- [ ] **Step 3: 最小实现：加入 :root UI tokens**

在 `<style>` 的 `:root` 增加一组变量：
- `--ui-bg`、`--ui-panel`、`--ui-panel-2`
- `--ui-stroke`、`--ui-stroke-2`
- `--ui-text`、`--ui-muted`
- `--ui-brand`、`--ui-brand-2`
- `--ui-shadow`
- `--ui-radius`、`--ui-radius-2`

- [ ] **Step 4: 新增容器骨架**

新增两个容器（先不迁移内容，只做骨架）：
- `div#playbackBar`（底部）
- `aside#lensDrawer`（右侧抽屉）

- [ ] **Step 5: 运行测试确认通过**

Run:
```bash
PYTHONPATH=src python -m pytest -q tests/test_viz_trace_injection_markers.py::test_ui_polish_has_design_tokens_and_new_layout_ids
```

- [ ] **Step 6: Commit**

```bash
git add src/vitriol/viz/model_3d_visualizer.html tests/test_viz_trace_injection_markers.py
git commit -m "feat(viz): add design tokens and new layout containers"
```

---

## Task 2: 重构回放控件：迁移到 Playback Bar（集中、对齐、响应式）

**Files:**
- Modify: `src/vitriol/viz/model_3d_visualizer.html`
- Test: `tests/test_viz_trace_injection_markers.py`

- [ ] **Step 1: failing test：原 toolbar 不应再是 fixed-left 堆叠**

Add:
```python
def test_playback_is_centralized_into_bar() -> None:
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    assert 'id="playPauseBtn"' in html
    assert 'id="tokenSlider"' in html
    # 新版：控件在 playbackBar 内
    assert 'id="playbackBar"' in html
```

- [ ] **Step 2: 实现迁移**
把 `#playbackToolbar` 的内容迁移到 `#playbackBar` 内：
- Play/Pause, Step, Speed
- Token slider（时间轴）
- Follow/Auto focus 开关

保留原有 JS 绑定（id 不变），只改 DOM 位置与 CSS。

- [ ] **Step 3: 运行测试 + 冒烟**

Run:
```bash
PYTHONPATH=src python -m pytest -q tests/test_viz_trace_injection_markers.py
```
Manual:
- 打开 trace_3d.html，确认按钮/slider 可用且不遮挡

- [ ] **Step 4: Commit**

```bash
git add src/vitriol/viz/model_3d_visualizer.html tests/test_viz_trace_injection_markers.py
git commit -m "refactor(viz): move playback controls into bottom bar"
```

---

## Task 3: 重构 Tokens + Attention Lens：迁移到右侧 Drawer（更像专业工具）

**Files:**
- Modify: `src/vitriol/viz/model_3d_visualizer.html`
- Test: `tests/test_viz_trace_injection_markers.py`

- [ ] **Step 1: failing test：tokens/lens canvas 应在 lensDrawer 内**

Add:
```python
def test_tokens_lens_drawer_exists() -> None:
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    assert 'id="lensDrawer"' in html
    assert 'id="tokenList"' in html
    assert 'id="attnCanvas"' in html
    assert 'id="attnHistogramCanvas"' in html
```

- [ ] **Step 2: 迁移 tokenListPanel 到 drawer**
- Drawer header：标题 + tabs（Tokens / Lens / Meta）
- Drawer body：histogram bar + token chips + arcs overlay
- Drawer footer（可选）：top-k、layer、mean-heads 等提示

- [ ] **Step 3: 视觉优化**
- token chip：统一 padding/字号/对比度；active 用更克制的高亮
- heat：统一色阶（brand→brand2），背景不做“满底色”，改为轻量内阴影 + 细边
- arcs：线宽/透明度更稳定，减少跳动；只在 hover 或 follow+attn 时展示

- [ ] **Step 4: 运行测试通过**

Run:
```bash
PYTHONPATH=src python -m pytest -q tests/test_viz_trace_injection_markers.py
```

- [ ] **Step 5: Commit**
```bash
git add src/vitriol/viz/model_3d_visualizer.html tests/test_viz_trace_injection_markers.py
git commit -m "feat(viz): move tokens and attention lens into right drawer"
```

---

## Task 4: “专业感”细节：对齐、留白、动效节奏、降噪策略

**Files:**
- Modify: `src/vitriol/viz/model_3d_visualizer.html`

- [ ] **Step 1: 统一对齐栅格**
- 所有 panel padding 统一为 12/14/16 体系
- 字号统一：12（正文）/ 11（注释）/ 10（meta）
- monospace 只用于 meta/数值

- [ ] **Step 2: 动效节奏**
- camera follow 的 duration 与 speed 映射更克制（避免忽快忽慢）
- arcs 的绘制使用 requestAnimationFrame 缓动（可选）

- [ ] **Step 3: 视觉噪声控制**
- 仅当 `nodeId` 是 `*:attn` 且 trace 有数据时才显示 histogram/arcs
- 非 attn 步隐藏 histogram/arcs（减少无意义占位）

- [ ] **Step 4: 手工冒烟**
打开 trace_3d.html：
- 拖动 token timeline
- hover token 看 lens
- 播放时 lens 自动出现/消失

- [ ] **Step 5: Commit**
```bash
git add src/vitriol/viz/model_3d_visualizer.html
git commit -m "chore(viz): polish spacing alignment and motion"
```

---

## Task 5: 更新 demo 产物（tinyllama trace_3d.html）

**Files/Outputs:**
- Update: `verification/tinyllama_hybrid_ultra_trace/trace.json`
- Update: `verification/tinyllama_hybrid_ultra_trace/trace_3d.html`

- [ ] **Step 1: 重跑 trace 生成（确保 attention_histogram/topk 仍在）**
```bash
PYTHONPATH=src python -m vitriol.cli.main trace --model-path output/tinyllama-hybrid-ultra-test --prompt "hello" --max-new-tokens 8 --out verification/tinyllama_hybrid_ultra_trace/trace.json --device cpu
```

- [ ] **Step 2: 重新生成注入版 HTML**
```bash
PYTHONPATH=src python /sessions/69eedf2298277f65d17a1d8a/work/generate_tinyllama_trace_html.py
```

- [ ] **Step 3: 跑回归测试**
```bash
PYTHONPATH=src python -m pytest -q tests/test_viz_trace_injection_markers.py tests/test_viz_token_playback_markers.py tests/test_trace_schema_v1.py tests/test_trace_cli_outputs_token_fields.py
```

---

## Execution Handoff
Plan complete and saved to `docs/superpowers/plans/2026-04-28-attention-lens-ui-polish.md`.

你希望我继续用哪种方式执行？
1) Subagent-Driven（推荐）
2) Inline Execution

