# 结构驱动 Token 推理回放（2D/3D）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Vitriol 的 3D 可视化中加入“结构驱动 Token 推理回放”动画（播放/暂停/单步/倍速/跳转），并在 2D 中同步展示当前 token/layer 信息，提升交互与讲解能力。

**Architecture:** 使用统一的 Playback Engine（JS）生成/消费结构驱动事件流（trace），3D 负责粒子与模块高亮，2D 负责读数面板；两者共享同一状态结构以保证同步与可复现。

**Tech Stack:** 原生 HTML/JS（现有 vitriol viz HTML），Three.js（3D 已在使用），pytest（结构性回归测试）

---

## 0) 代码结构映射（将要修改/新增的文件）

**Modify:**
- `src/vitriol/viz/model_3d_visualizer.html`  
  - 新增 playback toolbar（UI）
  - 新增 token 粒子与高亮逻辑
  - 建立 nodeId → 3D object/position 的索引

- `src/vitriol/viz/model_visualizer.html`  
  - 新增同步信息面板（当前 token/layer/step）
  - 提供可选：同一页面内模拟回放（2D-only 模式）

**Add:**
- `src/vitriol/viz/playback/playback_engine.js`（纯前端模块，以 `<script>` 内联方式引入或复制到 HTML 内）
- `src/vitriol/viz/playback/trace_generator.js`（结构驱动 trace 生成器）

**Test (Add):**
- `tests/test_viz_token_playback_markers.py`（确保关键 DOM id/标识/文案存在，防止“伪真/缺失控制条/回退逻辑”回归）

> 说明：现有项目以“单文件 HTML”组织较多。本计划按最小入侵原则推进：优先在 HTML 内新增可隔离的模块代码块；若文件增长过快，再抽到 `src/vitriol/viz/playback/*.js` 并用注入方式拼进 HTML。

---

## Task 1: 为 3D 增加 Playback Toolbar 的最小骨架（DOM + 样式 + badge）

**Files:**
- Modify: `src/vitriol/viz/model_3d_visualizer.html`
- Test: `tests/test_viz_token_playback_markers.py`

- [ ] **Step 1: 写 failing test（结构性检查）**

Create `tests/test_viz_token_playback_markers.py`:

```python
from pathlib import Path


def test_3d_has_token_playback_toolbar_and_demo_badge() -> None:
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    assert 'id="playbackToolbar"' in html
    assert 'id="playPauseBtn"' in html
    assert 'id="stepBtn"' in html
    assert 'id="speedSelect"' in html
    assert 'id="tokenSlider"' in html
    assert "Structure-driven" in html  # 防误导：必须标识为演示
```

- [ ] **Step 2: 运行测试，确认失败**

Run:
```bash
PYTHONPATH=src python -m pytest -q tests/test_viz_token_playback_markers.py
```
Expected: FAIL（缺少上述 DOM id/文案）

- [ ] **Step 3: 最小实现：在 3D HTML 顶部加入 toolbar**

在 `model_3d_visualizer.html` header 区域附近加入：
- `div#playbackToolbar`
- buttons: `#playPauseBtn`, `#stepBtn`
- select: `#speedSelect`
- range: `#tokenSlider`
- badge 文案：`🧪 DEMO (Structure-driven)`

- [ ] **Step 4: 运行测试，确认通过**

Run:
```bash
PYTHONPATH=src python -m pytest -q tests/test_viz_token_playback_markers.py
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/vitriol/viz/model_3d_visualizer.html tests/test_viz_token_playback_markers.py
git commit -m "feat(viz): add 3d token playback toolbar skeleton"
```

---

## Task 2: 建立 nodeId → 3D 节点位置索引（为动画定位做准备）

**Files:**
- Modify: `src/vitriol/viz/model_3d_visualizer.html`
- Test: `tests/test_viz_token_playback_markers.py`（追加 marker）

- [ ] **Step 1: 扩展 failing test，要求存在 node index 构建函数**

Append to `tests/test_viz_token_playback_markers.py`:

```python
def test_3d_has_node_index_builder() -> None:
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    assert "function buildNodeIndex" in html
    assert "nodeIndex" in html
```

- [ ] **Step 2: 运行测试，确认失败**

Run:
```bash
PYTHONPATH=src python -m pytest -q tests/test_viz_token_playback_markers.py
```
Expected: FAIL

- [ ] **Step 3: 最小实现：在 3D JS 中新增 buildNodeIndex()**

要求：
- 输出 `nodeIndex: Map<string, { object3d, worldPos: THREE.Vector3 }>`
- 建立 key 命名：`embed`, `lm_head`, `block:${i}:attn`, `block:${i}:ffn|moe`
- 先覆盖最常见节点：embed、每层 attn、每层 ffn/moe、lm_head

- [ ] **Step 4: 运行测试，确认通过**

- [ ] **Step 5: Commit**

```bash
git add src/vitriol/viz/model_3d_visualizer.html tests/test_viz_token_playback_markers.py
git commit -m "feat(viz): build 3d node index for playback"
```

---

## Task 3: 实现 PlaybackEngine（状态机 + RAF 驱动）

**Files:**
- Modify: `src/vitriol/viz/model_3d_visualizer.html`（内联 engine）
- (Optional Add): `src/vitriol/viz/playback/playback_engine.js`
- Test: `tests/test_viz_token_playback_markers.py`

- [ ] **Step 1: failing test：检查存在 PlaybackEngine 类与关键方法**

Append:
```python
def test_3d_has_playback_engine_class() -> None:
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    assert "class PlaybackEngine" in html
    assert "play()" in html
    assert "pause()" in html
    assert "step()" in html
    assert "setSpeed" in html
```

- [ ] **Step 2: 运行测试，确认失败**

- [ ] **Step 3: 最小实现 PlaybackEngine**

要求：
- 使用 `requestAnimationFrame` 作为 tick
- 状态字段：`paused, speed, tokenIndex, layerIndex`
- 支持：
  - `play() / pause()`
  - `step()`：推进 layerIndex（到末尾则 tokenIndex+1 并 layerIndex=0）
  - `setSpeed(x)`：改变每层推进时间
  - `onChange(cb)`：状态变化回调（用于 UI 与 3D 渲染）

- [ ] **Step 4: 运行测试，确认通过**

- [ ] **Step 5: Commit**

```bash
git add src/vitriol/viz/model_3d_visualizer.html tests/test_viz_token_playback_markers.py
git commit -m "feat(viz): add structure-driven playback engine"
```

---

## Task 4: 3D 粒子 token + 节点高亮（Phase 1 最小可用）

**Files:**
- Modify: `src/vitriol/viz/model_3d_visualizer.html`
- Test: `tests/test_viz_token_playback_markers.py`

- [ ] **Step 1: failing test：检查存在 token particle 创建与高亮函数**

Append:
```python
def test_3d_has_token_particle_and_highlight_helpers() -> None:
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    assert "function createTokenParticle" in html
    assert "function highlightNode" in html
```

- [ ] **Step 2: 实现 createTokenParticle()**

要求：
- 创建 `THREE.Mesh(SphereGeometry, MeshStandardMaterial)`
- token 颜色与 tokenIndex 映射（可复现）
- 位置从 `nodeIndex.get('embed')` 初始化

- [ ] **Step 3: 实现 highlightNode(nodeId)**

要求：
- 对当前节点 object3d 做 emissive/scale 动效（最小可用：改变材质 emissive/颜色）
- 上一个高亮应复位，避免累积

- [ ] **Step 4: 绑定 PlaybackEngine.onChange()**

逻辑：
- 根据当前 `tokenIndex/layerIndex` 计算目标 nodeId：
  - layerIndex == -1 → `embed`
  - 0..N-1 → `block:${i}:attn`（第一阶段只走主干）
  - 最后 → `lm_head`
- 粒子在两个 nodeId 的 worldPos 之间线性插值（以时间参数 t）

- [ ] **Step 5: 运行测试并做手工冒烟**

Run:
```bash
PYTHONPATH=src python -m pytest -q tests/test_viz_token_playback_markers.py
```

Manual:
- 打开 3D HTML
- 点击 Play：粒子应移动与高亮
- Pause/Step 生效

- [ ] **Step 6: Commit**

```bash
git add src/vitriol/viz/model_3d_visualizer.html tests/test_viz_token_playback_markers.py
git commit -m "feat(viz): animate token particle through layers in 3d"
```

---

## Task 5: UI 控件与 PlaybackEngine 绑定（Play/Pause/Step/Speed/Slider）

**Files:**
- Modify: `src/vitriol/viz/model_3d_visualizer.html`
- Test: `tests/test_viz_token_playback_markers.py`

- [ ] **Step 1: failing test：要求存在 bindPlaybackControls()**

Append:
```python
def test_3d_binds_playback_controls() -> None:
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    assert "function bindPlaybackControls" in html
```

- [ ] **Step 2: 实现 bindPlaybackControls(engine)**

要求：
- `#playPauseBtn` 点击切换 `play/pause`，按钮文本同步
- `#stepBtn` 调用 `engine.step()`
- `#speedSelect` 调用 `engine.setSpeed(parseFloat(value))`
- `#tokenSlider` 设置 `engine.setTokenIndex(n)`（需要新增方法）

- [ ] **Step 3: 运行测试通过**

- [ ] **Step 4: Commit**

```bash
git add src/vitriol/viz/model_3d_visualizer.html tests/test_viz_token_playback_markers.py
git commit -m "feat(viz): bind playback toolbar controls"
```

---

## Task 6: 2D 同步面板（只读读数，Phase 1）

**Files:**
- Modify: `src/vitriol/viz/model_visualizer.html`
- Test: `tests/test_viz_token_playback_markers.py`

- [ ] **Step 1: failing test：检查 2D 有同步面板 DOM**

Append:
```python
def test_2d_has_playback_status_panel() -> None:
    html = Path("src/vitriol/viz/model_visualizer.html").read_text(encoding="utf-8")
    assert 'id="playbackStatusPanel"' in html
    assert 'id="currentTokenText"' in html
    assert 'id="currentLayerText"' in html
```

- [ ] **Step 2: 最小实现：在 2D 添加 panel + update 函数**

要求：
- panel 不遮挡主 SVG（放在右侧或底部）
- 提供 `window.__VITRIOL_PLAYBACK_STATE__` 接口（3D 写入 / 2D 读取）
- 若 2D 独立打开，则显示“未连接到播放引擎”

- [ ] **Step 3: 运行测试通过**

- [ ] **Step 4: Commit**

```bash
git add src/vitriol/viz/model_visualizer.html tests/test_viz_token_playback_markers.py
git commit -m "feat(viz): add 2d playback status panel"
```

---

## Task 7: 端到端冒烟与示例输出（生成 Qwen3.5 demo HTML）

**Files:**
- Modify: `/sessions/.../work/generate_qwen35_viz_html.py`（仅用于生成验证 HTML，不进入仓库或放到 docs）
- Output: `workspace/verification/qwen35_vitriol_ultra_dummy/qwen35_3d.html`（可直接打开）

- [ ] **Step 1: 重新生成 2D/3D HTML**

Run:
```bash
PYTHONPATH=src python /sessions/69eedf2298277f65d17a1d8a/work/generate_qwen35_viz_html.py
```

- [ ] **Step 2: 手工验证**
- 打开 3D，点击 Play/Pause/Step/Speed/Slider，确认动画与高亮正常
- 打开 2D，确认 panel 显示状态（若实现了共享 state 则同步）

---

## 自检（Plan Self-Review）
### Spec coverage
- Toolbar + badge（Task1）
- nodeId 定位（Task2）
- PlaybackEngine（Task3）
- 粒子与高亮（Task4）
- 控件绑定（Task5）
- 2D 同步面板（Task6）
- demo 验证（Task7）

### Placeholder scan
无 TBD/TODO；每个步骤给出明确路径与命令。

---

## Execution Handoff
Plan complete and saved to `docs/superpowers/plans/2026-04-28-structure-driven-token-playback.md`.

两种执行方式：
1) **Subagent-Driven（推荐）**：我按 Task 分派子代理逐个实现并复核  
2) **Inline Execution**：我在当前会话里按任务逐步实现（带检查点）  

你希望我用哪一种？

