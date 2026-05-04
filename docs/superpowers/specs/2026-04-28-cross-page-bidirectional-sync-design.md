# 跨页面双向同步（2D ⇄ 3D）设计稿

**日期：** 2026-04-28  
**范围：** Vitriol 的 3D 回放页（`model_3d_visualizer.html`）与 2D 架构图页（`model_visualizer.html`）  
**目标：** 提供一个“模式/开关”实现跨页面双向同步：  
1) **3D → 2D：** 3D 播放/拖动时，2D 实时高亮到当前 `nodeId`（细粒度优先）。  
2) **2D → 3D：** 2D 点击任意节点（含细粒度子模块）时，3D **跳转并聚焦**到对应 `nodeId`，并且**默认自动暂停**（便于讲解/分析）。

---

## 1. 用户体验（UX）与交互约定

### 1.1 3D 页面
- Playback Bar 增加开关：`Sync 2D`  
  - 开启后：3D 将持续对外发布回放状态，同时接收来自 2D 的控制命令。  
  - 关闭后：3D 不发布、不接收（避免干扰用户单独使用页面）。  
- 同时支持 URL 参数：`#?sync=1`（用于一键开启同步，方便分享/脚本化启动）。

### 1.2 2D 页面
- 无需新增复杂 UI：默认自动监听同步通道。  
- 用户点击：
  - 组件主块（如 `block:i:attn` / `block:i:ffn` / `block:i:norm1`）  
  - 组件子块（如 `block:i:attn:q_proj`、`block:i:ffn:gate_proj`）  
  → 发送 **focus_node** 命令到 3D 页面。  
- 2D 自身仍保持“本地点击展示 details”的能力，不被同步模式破坏。

### 1.3 默认“暂停”策略（已确认）
- 2D → 3D 的 focus 命令默认触发：`pause + focus`  
  - pause：确保用户看到聚焦结果，不被播放继续推进覆盖。  
  - focus：镜头跟随（若开启 Follow camera/Auto focus，则采用现有策略）。

---

## 2. 数据与协议（CrossPageSync v1）

### 2.1 通道选择（优先级）
1) **BroadcastChannel**（实时、同源多标签页适用）  
2) **localStorage + storage event**（fallback；BroadcastChannel 不可用时仍能同步）

### 2.2 统一通道名/键
- BroadcastChannel name: `vitriol_playback_v1`  
- localStorage key: `VITRIOL_PLAYBACK_STATE`（state payload）  
- localStorage key（命令建议）：`VITRIOL_PLAYBACK_COMMAND`（command payload，可选；若只用 BroadcastChannel 也可不落盘）

### 2.3 消息结构

#### A) state（3D → all）
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

#### B) command（2D → 3D）
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

> 约束：`nodeId` 采用 trace 的命名约定；3D 端若不支持该细粒度节点，必须回退到父节点（例如 `block:7:ffn` / `block:7:attn`）。

---

## 3. 3D 端实现要点

### 3.1 发布 state
- 在 PlaybackEngine 的 `_emit()` 中：
  - 仍写入 `window.__VITRIOL_PLAYBACK_STATE__`（供同页读取）
  - 若 Sync 开启：发布 state 到 BroadcastChannel，并写入 localStorage（去噪/节流）

### 3.2 接收 command
- 监听 BroadcastChannel message：收到 `{type:"command", source:"2d"}`：
  - 若 `command.name === "focus_node"`：
    - 若 `pause === true`：调用 `engine.pause()`  
    - `engine.nodeId = normalizedNodeId` 并 `_emit()`（使 2D/3D 状态一致）  
    - 调用现有 `highlightNode(nodeId)` + `followCameraToNode(nodeId)`  
- 兼容回退：
  - `block:i:attn:*` → 回退到 `block:i:attn`
  - `block:i:ffn:*` → 回退到 `block:i:ffn`

---

## 4. 2D 端实现要点

### 4.1 订阅 state（已有轮询，增强为事件驱动）
- BroadcastChannel：收到 `{type:"state", source:"3d"}` → `updatePlaybackStatus(state)`  
- localStorage fallback：监听 `storage` 事件（key=VITRIOL_PLAYBACK_STATE）→ 同上  

### 4.2 发布 command（新增）
- 在 SVG 节点（主块/子块）的 click handler 中：
  - 保持原 `showDetails()` 的行为
  - 额外调用 `publishCommand({name:"focus_node", nodeId, pause:true})`
- 注意 stopPropagation：避免父组件 click 重复触发。

---

## 5. 测试策略（TDD）

### 5.1 静态标记测试（已存在/新增）
- 3D：包含 `BroadcastChannel`、`VITRIOL_PLAYBACK_STATE`、`syncToggle`
- 2D：包含 `BroadcastChannel`、`storage` listener、`selectByNodeId`

### 5.2 行为测试（建议新增，非强制）
由于目前测试框架是对 HTML/JS 做静态断言为主，可通过：
- 断言存在 `publishCommand`/`onmessage` 处理函数名
- 断言 click handler 中包含 command 构造与 pause:true

---

## 6. 非目标（本轮不做）
- 2D 中显示 token-level attention heat（2D 目前作为结构图，保持轻量）
- 多房间/多模型并行的命名空间（未来可加入 roomId）
- 跨不同 origin 的同步（只支持同源标签页）

---

## 7. 风险与降级策略
- BroadcastChannel 在部分环境不可用：fallback 到 localStorage。
- 多个 3D 页同时开启同步：以“最后写入/最后发出”为准；可后续加入 `sessionId` 做隔离。

