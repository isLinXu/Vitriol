# NodeId 接口字典与对齐契约（trace ⇄ 3D ⇄ 2D）

**日期：** 2026-04-28  
**适用范围：** 本仓库 `vitriol` 的离线 trace 回放体系与 2D/3D 可视化  
**目标：** 用一份“接口字典”把 **Trace JSON**、**3D nodeIndex**、**2D SVG data-id**、**跨页同步协议** 的字段/命名/回退规则固定下来，避免“看起来能播但语义错位”的问题。

---

## 1. 读者与使用方式

**面向读者：**
- 扩展 trace 采集（新增/修改 hook 节点）
- 扩展 2D/3D 架构图节点与高亮逻辑
- 扩展跨页面同步协议（state/command）
- 给新模型类型（非 Llama）接入回放/可视化

**使用方式：**
1) 先看「NodeId 语法」与「NodeId 列表」  
2) 再看「三方对齐：Trace / 3D / 2D」  
3) 最后看「同步协议」与「file:// 握手探测」  

---

## 2. 核心概念

### 2.1 NodeId（唯一标识）
NodeId 是一个字符串 key，用于标识“推理/结构中的一个节点”。它必须在以下位置保持一致（或可回退一致）：
- Trace：`events[].node_path[]`
- 3D：`nodeIndex` 的 key（用于定位 worldPos 与高亮）
- 2D：SVG element 的 `data-id`（用于点击/高亮）
- Sync：state/command payload 的 `nodeId`

### 2.2 三类粒度
1) **全局节点**：`embed`、`lm_head`  
2) **Block 粗粒度节点**：`block:{i}:attn`、`block:{i}:ffn`  
3) **Block 细粒度节点（子模块）**：`block:{i}:attn:q_proj` 等

> 细粒度节点可能只在 Trace 中真实存在（hook 到线性层），而 3D mesh 可能只有粗粒度锚点。此时 3D 必须通过 **synthetic subnodes** 提供“可落点”的 worldPos，并在高亮时回退到父 mesh。

---

## 3. NodeId 语法（Grammar）

### 3.1 全局节点
- `embed`
- `lm_head`

### 3.2 Block 节点
通用格式：
- `block:{layer_index}:{kind}`

其中：
- `layer_index`：从 0 开始的整数
- `kind`：`norm1 | norm2 | attn | ffn | moe` 等

### 3.3 子模块节点
子模块通过 `:` 继续向下细分：
- `block:{i}:attn:q_proj`
- `block:{i}:attn:k_proj`
- `block:{i}:attn:v_proj`
- `block:{i}:attn:o_proj`
- `block:{i}:ffn:gate_proj`
- `block:{i}:ffn:up_proj`
- `block:{i}:ffn:down_proj`

---

## 4. Llama-like 模型的“标准 NodeId 列表”

### 4.1 每个 block（layer i）最小应出现的节点
粗粒度（必须至少存在）：
- `block:{i}:norm1`
- `block:{i}:attn`
- `block:{i}:norm2`
- `block:{i}:ffn`（或 `:moe`）

细粒度（推荐；用于 1:1 对齐解释）：
- `block:{i}:attn:q_proj`
- `block:{i}:attn:k_proj`
- `block:{i}:attn:v_proj`
- `block:{i}:attn:o_proj`
- `block:{i}:ffn:gate_proj`
- `block:{i}:ffn:up_proj`
- `block:{i}:ffn:down_proj`

### 4.2 兼容命名（历史字段）
Trace 端仍可能产生：
- `block:{i}:mlp`

**归一化规则（canonical）：**
- `:mlp` → `:ffn`

（见 3D `normalizeTraceNodeId()`）

---

## 5. 三方对齐契约（Trace ⇄ 3D ⇄ 2D）

### 5.1 Trace（source of truth for execution order）
位置：`trace.v1` 的 `events[]`

关键字段：
- `events[].node_path`：**顺序敏感**，表示一次 token 计算中“经过哪些节点”
- `events[].attention_topk["block:{i}:attn"]`：每层 attention 的 top-k（mean over heads, last tgt）
- `events[].attention_histogram["block:{i}:attn"]`：每层桶化分布（bins/values）

约束：
- `attention_topk` 与 `attention_histogram` 的 key **必须是粗粒度** `block:{i}:attn`  
  - 即使当前 `nodeId` 是 `block:{i}:attn:q_proj`，也要通过回退映射到 `block:{i}:attn` 取 attention 数据（避免语义错配）。

### 5.2 3D（定位与高亮）
位置：`src/vitriol/viz/model_3d_visualizer.html`

核心结构：
- `nodeIndex: Map<string, { object3d, worldPos }>`

对齐策略：
1) **粗粒度节点**（有真实 mesh）：
   - 通过模块 mesh name（如 `Attn_{i}`, `Down_{i}` 等）注册
2) **细粒度节点**（无真实 mesh）：
   - 通过 `synthetic subnodes` 注册 `worldPos = parent.worldPos + offset`
   - `object3d` 指向 parent anchor mesh（保证 highlight 仍可用）

回退规则（必须实现）：
- 若 `nodeIndex` 不包含 `block:{i}:attn:*` 子节点：回退到 `block:{i}:attn`
- 若 `nodeIndex` 不包含 `block:{i}:ffn:*` 子节点：回退到 `block:{i}:ffn`

Attention Lens 的 key 回退（必须实现）：
- 当 `currentNode = block:{i}:attn:q_proj`：
  - `attnKey = block:{i}:attn`
  - 从 `attention_topk[attnKey]` / `attention_histogram[attnKey]` 取数据

### 5.3 2D（可点击节点与高亮）
位置：`src/vitriol/viz/model_visualizer.html`

对齐策略：
- 每个 layer 结构块都有：
  - `layer.trace_id`（优先，若无则 fallback `layer.id`）
  - SVG `data-id = layer.trace_id`
- 子模块（Q/K/V/O、GATE/UP/DOWN）：
  - SVG `data-id = <subNodeId>`

高亮策略：
- `selectByNodeId(nodeId)`：
  - 优先找精确的 `data-id=nodeId`
  - 找不到则回退到父节点 `block:{i}:attn` / `block:{i}:ffn`

点击行为（双向同步）：
- 点击主块 / 子块：
  - 保持 `showDetails(...)`
  - 同时发布 sync command（见 §6）

---

## 6. 跨页面同步协议（2D ⇄ 3D）

通道：
- `BroadcastChannel("vitriol_playback_v1")`（优先）
- localStorage fallback：
  - state key：`VITRIOL_PLAYBACK_STATE`
  - command key：`VITRIOL_PLAYBACK_COMMAND`

### 6.1 state（3D → 2D）
```json
{
  "ts": 1714330000000,
  "source": "3d",
  "type": "state",
  "state": {
    "paused": true,
    "speed": 1,
    "tokenIndex": 3,
    "tokenGlobalIndex": 12,
    "layerIndex": 0,
    "mode": "trace",
    "nodeId": "block:0:attn:q_proj"
  }
}
```

2D 收到后必须：
- 写入 `window.__VITRIOL_PLAYBACK_STATE__`
- 调用 `updatePlaybackStatus(state)`（进而驱动 `selectByNodeId(state.nodeId)`）

### 6.2 command（2D → 3D）
```json
{
  "ts": 1714330000000,
  "source": "2d",
  "type": "command",
  "command": {
    "name": "focus_node",
    "nodeId": "block:7:ffn:gate_proj",
    "pause": true
  }
}
```

3D 收到后默认行为（已确认）：
- `pause + highlight + followCamera`
- 如果 3D 引擎未 ready：缓存最后 command，ready 后立即 apply
- nodeId 若不存在：回退到父节点（attn/ffn）

---

## 7. file:// 场景：端口探测握手（可用性保障）

问题：
- 用户用 `file://` 打开 2D，会导致 origin 不透明、无法与 `http://localhost:*` 通信 → 同步必失败。

解决：
- 2D(file://) 提示条里做自动探测：
  - 创建隐藏 iframe 扫候选端口（8002/8765/…）加载：
    - `http://localhost:{port}/trace_3d.html#?handshake=1`
  - 监听 `window.message`：
    - 收到 `{type:'vitriol_handshake', source:'3d'}` 后用 `event.origin` 锁定正确 `http://localhost:PORT`
  - 用该 origin 生成“一键打开/复制链接”的 HTTP URL
  - 超时回退到 8765

3D 端握手响应（handshake=1）：
- `window.parent.postMessage({type:'vitriol_handshake', source:'3d', origin: location.origin}, '*')`

---

## 8. 测试护栏（哪些文件会锁定这份契约）

> 本仓库大量采用“静态标记测试”锁定关键能力，重构 HTML/JS 时需同步更新。

推荐从这些测试入手理解契约边界：
- Trace schema：
  - `tests/test_trace_schema_v1.py`
  - `tests/test_trace_cli_outputs_token_fields.py`（含细粒度节点必须出现）
- 3D viz 注入/回放护栏：
  - `tests/test_viz_trace_injection_markers.py`
  - `tests/test_viz_token_playback_markers.py`
- 2D 真实性护栏：
  - `tests/test_viz_model_visualizer_truthfulness.py`
- 跨页同步护栏：
  - `tests/test_viz_cross_page_sync_markers.py`
- file:// 可用性与握手护栏：
  - `tests/test_viz_file_protocol_hint.py`
  - `tests/test_viz_3d_handshake_markers.py`

---

## 9. 给“新增模型类型”的接入清单（Checklist）

当接入非 Llama-like 模型时，建议按顺序做：
1) **Trace**：新增/扩展 hook，产出可用 `node_path`（至少 embed/attn/ffn/head）
2) **NodeId 命名**：保证同一套 grammar（block:i:* 或另定义但要文档化）
3) **3D**：buildNodeIndex 能注册粗粒度锚点；必要时 synthetic subnodes 补细粒度落点
4) **2D**：layer.trace_id 产出与 nodeId 对齐；子模块可选
5) **Sync**：state/command 的 nodeId 能被 3D/2D 理解与回退
6) **Tests**：增加最小护栏（至少 schema + marker）

