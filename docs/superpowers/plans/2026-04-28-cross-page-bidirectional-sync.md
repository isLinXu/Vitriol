# 跨页面双向同步（2D ⇄ 3D）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个可开关的跨页面双向同步模式：3D 回放页发布播放状态给 2D 架构图页；2D 点击节点可命令 3D 跳转/聚焦并默认暂停。

**Architecture:** 复用同源 `BroadcastChannel("vitriol_playback_v1")` 作为主通道，localStorage (`VITRIOL_PLAYBACK_STATE`) 作为 fallback。消息分为 `state`（3D→2D）与 `command`（2D→3D）。3D 在 Sync 开启时发布 state，并监听 command 执行 `pause + highlight + followCamera`。2D 订阅 state 同步高亮，并在 SVG 节点 click 时发布 command。

**Tech Stack:** 单文件 HTML/JS（`model_3d_visualizer.html` / `model_visualizer.html`），pytest（静态标记/结构性回归测试）

---

## 文件结构（将修改/新增的文件）

**Modify:**
- `src/vitriol/viz/model_3d_visualizer.html`（接收 command + 执行聚焦/暂停；完善 payload.type；可选 command fallback）
- `src/vitriol/viz/model_visualizer.html`（发布 command；保证子节点 click 的 stopPropagation；可选 UI 提示）

**Create/Modify Tests:**
- `tests/test_viz_cross_page_sync_markers.py`（扩充：要求存在 command publish/subscribe 标记）

**(Optional) Docs:**
- `docs/superpowers/specs/2026-04-28-cross-page-bidirectional-sync-design.md`（已存在，无需改）

---

## Task 1: RED — 新增双向同步（command）相关测试

**Files:**
- Modify: `tests/test_viz_cross_page_sync_markers.py`

- [ ] **Step 1: 写 failing test（3D 必须有 command handler；2D 必须有 publishCommand）**

在 `tests/test_viz_cross_page_sync_markers.py` 中补充断言：

```python
def test_3d_has_cross_page_command_handler_markers() -> None:
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    assert "type\": \"command\"" in html or "type:'command'" in html or "type: 'command'" in html
    assert "focus_node" in html
    assert "followCameraToNode" in html
    assert ".pause(" in html or "pause()" in html


def test_2d_has_cross_page_command_publish_markers() -> None:
    html = Path("src/vitriol/viz/model_visualizer.html").read_text(encoding="utf-8")
    assert "type\": \"command\"" in html or "type:'command'" in html or "type: 'command'" in html
    assert "focus_node" in html
    assert "BroadcastChannel" in html
```

- [ ] **Step 2: 运行测试确认失败**

Run:
```bash
PYTHONPATH=src python -m pytest -q tests/test_viz_cross_page_sync_markers.py
```
Expected: FAIL

---

## Task 2: GREEN — 3D 端接收 command 并执行 pause+focus

**Files:**
- Modify: `src/vitriol/viz/model_3d_visualizer.html`
- Test: `tests/test_viz_cross_page_sync_markers.py`

- [ ] **Step 1: 在 3D 的 setupCrossPageSync3D 中补齐 payload.type 并暴露 subscribe**

把 state payload 统一为：
```js
{ ts, source:'3d', type:'state', state }
```

新增 `subscribe(onCommand)`：
- 使用 `BroadcastChannel.onmessage` 监听 `{source:'2d', type:'command'}`  
- 解析 `command.name === 'focus_node'` 时执行：
  - `playbackEngine.pause()`（默认 pause:true）
  - 规范化 nodeId：`block:i:attn:*` → `block:i:attn`（若 nodeIndex 无该 key）；`ffn` 同理
  - `highlightNode(nodeId)` + `followCameraToNode(nodeId)`
  - 将 `engine.nodeId = nodeId; engine._emit()`，让 state 同步回 2D

（注意：`playbackEngine` 在 `setupPlaybackControls()` 里初始化，需要确保 command handler 能访问到它。若 handler 收到消息时 engine 尚未 ready，先缓存最后一个 command，engine 初始化后再应用。）

- [ ] **Step 2: 可选 localStorage command fallback（YAGNI 可跳过）**

若要做 fallback，则监听：
- key：`VITRIOL_PLAYBACK_COMMAND`

但本轮优先只做 BroadcastChannel（更实时且更少干扰）。

- [ ] **Step 3: 运行 tests 验证 GREEN**

Run:
```bash
PYTHONPATH=src python -m pytest -q tests/test_viz_cross_page_sync_markers.py
```

- [ ] **Step 4: Commit**

```bash
git add src/vitriol/viz/model_3d_visualizer.html tests/test_viz_cross_page_sync_markers.py
git commit -m "feat(viz): handle cross-page focus command in 3d"
```

---

## Task 3: GREEN — 2D 端发布 command（点击节点 → 3D 暂停并聚焦）

**Files:**
- Modify: `src/vitriol/viz/model_visualizer.html`
- Test: `tests/test_viz_cross_page_sync_markers.py`

- [ ] **Step 1: 增加 publishCommand()（BroadcastChannel + localStorage 选一）**

实现一个函数：
```js
function publishCommand(cmd) {
  const payload = { ts: Date.now(), source:'2d', type:'command', command: cmd };
  if (bc) bc.postMessage(payload);
  // optional fallback:
  // localStorage.setItem('VITRIOL_PLAYBACK_COMMAND', JSON.stringify(payload));
}
```

- [ ] **Step 2: 在 SVG 的 click handler 中发送 focus_node**

两处需要覆盖：
1) 主组件：`group.addEventListener('click', ...)`  
2) 子组件：`_mkSub(...).addEventListener('click', (evt) => { evt.stopPropagation(); ... })`

命令内容：
```js
publishCommand({ name:'focus_node', nodeId: subIdOrLayerKey, pause: true });
```

同时保持现有 `showDetails(...)` 行为不变。

- [ ] **Step 3: 运行 tests**

Run:
```bash
PYTHONPATH=src python -m pytest -q tests/test_viz_cross_page_sync_markers.py
```

- [ ] **Step 4: Commit**

```bash
git add src/vitriol/viz/model_visualizer.html tests/test_viz_cross_page_sync_markers.py
git commit -m "feat(viz): publish focus command from 2d nodes"
```

---

## Task 4: 验证与使用说明（demo + 回归）

**Files:**
- Update (generated): `verification/tinyllama_hybrid_ultra_trace/trace_3d.html`

- [ ] **Step 1: 重新生成注入版 3D demo HTML**

Run:
```bash
PYTHONPATH=src python /sessions/69eedf2298277f65d17a1d8a/work/generate_tinyllama_trace_html.py
```

- [ ] **Step 2: 跑全量相关回归**

Run:
```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_viz_cross_page_sync_markers.py \
  tests/test_viz_trace_injection_markers.py \
  tests/test_viz_token_playback_markers.py \
  tests/test_viz_2d_fine_grained_nodes.py
```

- [ ] **Step 3: 手工冒烟（说明写在 PR/提交描述即可）**
1) 打开 3D：勾选 `Sync 2D`  
2) 打开 2D：点击 `Q/K/V/O` 或 `GATE/UP/DOWN`  
3) 验证 3D 暂停并聚焦到对应节点（高亮+镜头移动），2D 同步显示状态

---

## Self-Review（执行前自检）
- [ ] 协议字段一致：`source/type/state/command` 命名统一
- [ ] 2D 点击子节点不会触发父节点 click（stopPropagation）
- [ ] 3D 接收 command 时引擎未初始化的情况有兜底（缓存最后命令）
- [ ] nodeId 不存在时回退到父节点，不抛异常

---

## Execution Handoff
Plan complete and saved to `docs/superpowers/plans/2026-04-28-cross-page-bidirectional-sync.md`.

两种执行方式：
1) Subagent-Driven（推荐）
2) Inline Execution

你选哪一种？

