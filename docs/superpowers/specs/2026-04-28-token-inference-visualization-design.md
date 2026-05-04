# Token 推理流动可视化（结构驱动）设计稿
Generated: 2026-04-28  
Scope: Vitriol 可视化系统（`src/vitriol/viz/model_3d_visualizer.html` + `src/vitriol/viz/model_visualizer.html`）  

## 1. 背景与目标
现有 2D/3D 可视化主要用于展示模型结构与统计信息，但交互与“推理过程解释性”不足。用户希望：

1) **增强交互能力与可视化效果**（更好看、更可读、更可操作）。  
2) **模拟实时看到每一个 token 如何通过模型架构推理**。  

本阶段选择“**动画演示（结构驱动）**”，即：不跑真实 forward/decoding 张量计算，但用真实结构（config/analyzer）驱动可复现的 token 流动动画，用于讲解/演示/对齐结构理解。

## 2. 非目标（明确边界）
- 不追求真实 attention 权重、真实专家选择概率、真实激活值（本阶段为结构驱动模拟）。
- 不做在线加载超大权重的逐步推理可视化。
- 不做分布式/多机推理跟踪。

> 后续可升级到“离线 trace 回放”或“真实推理”模式，但不在本阶段范围。

## 3. 用户体验（UX）概述
### 3.1 3D 作为权威主视图
3D 页面新增 **Inference Playback（推理回放）** 模式，展示：
- token 粒子沿结构路径移动（Embedding → Blocks → Head）。
- 当前所在模块高亮（glow/outline）。
- 视角跟随（可选）：镜头自动对准当前模块（可关闭）。
- 播放控制：播放/暂停、单步、倍速、跳转 token、跳转层。

### 3.2 2D 作为讲解/读数辅助
2D 页面新增一个轻量面板（不抢主图）：
- 当前 token 信息（索引、文本、颜色标识）。
- 当前 layer 进度条（0..N）。
- step 说明（进入 attention / 进入 MoE / 退出 block 等）。

2D 与 3D 使用**同一套播放引擎与事件流（trace）**，保证一致性。

## 4. 核心概念与数据模型
### 4.1 PlaybackState（统一播放状态）
```
PlaybackState = {
  tokens: Array<{ id: number; text: string; color: string }>,
  currentTokenIndex: number,
  currentLayerIndex: number,
  paused: boolean,
  speed: number, // e.g. 0.5, 1, 2, 4
  loop: boolean,
  mode: 'structure_demo',
}
```

### 4.2 TraceEvent（结构驱动事件流）
事件流由“结构”生成，用于驱动 2D/3D 动画。
```
TraceEvent = {
  t: number, // ms offset
  type: 'enter' | 'exit',
  tokenIndex: number,
  layerIndex: number,
  nodeId: string, // 3D/2D可定位的节点id
  meta?: {
    // MoE
    expertsChosen?: number[],
    topK?: number,
    // UI hints
    label?: string,
  }
}
```

### 4.3 NodeId 约定（可定位性）
需要在 3D/2D 内部建立稳定 nodeId：
- `embed`
- `block:{i}:attn`
- `block:{i}:ffn`（dense）
- `block:{i}:moe`（MoE）
- `lm_head`
- `moe:{i}:expert:{j}`（可选：用于 fan-out 专家展示）

## 5. 动画与交互设计
### 5.1 Token 粒子表现（3D）
- 每个 token 对应一个粒子（sphere / sprite）。
- token 颜色由 tokenIndex 映射（可复现）。
- 粒子沿 pathPoints（节点位置序列）插值移动。
- 当前节点高亮（材质 emissive 或 outline）。

### 5.2 MoE fan-out（可选，第二阶段）
当某层为 MoE：
- 粒子在 `moe` 节点处分裂为 top-k 子粒子，分别流向 `expert` 子节点，之后回到主干。
- expertsChosen 为伪随机但可复现（固定 seed = hash(modelId, tokenIndex, layerIndex)）。

### 5.3 控件（最小可用）
- Play / Pause
- Step（单步：推进到下一层或下一事件）
- Speed（0.5x/1x/2x/4x）
- Token slider（跳转 token）
- Layer slider（跳转层）
- Follow camera（开关）

快捷键（可选）：
- Space：Play/Pause
- →：Step
- +/-：Speed

## 6. 结构来源与“真实性”标识
本阶段结构驱动的“真实性”要求：
- 层数、MoE 层分布、专家数量等结构信息应来自 `meta-config.json`/`ArchitectureAnalyzer`/`__INLINE_*` 注入。
- 动画轨迹是演示（simulation），需显示 badge：`🧪 DEMO (Structure-driven)`。

## 7. 实施拆解（分阶段）
### Phase 1（最小可用：单 token 全层流动）
1) 在 3D 中建立稳定 nodeId 与节点坐标索引（用于动画定位）。
2) 新增 Playback 引擎（requestAnimationFrame 驱动）。
3) 渲染单 token 粒子，按层推进并高亮节点。
4) 添加基础控制条（Play/Pause/Step/Speed）。
5) 2D 同步显示当前 token/layer（简易面板）。

### Phase 2（MoE fan-out + 视角跟随）
1) MoE 节点展开专家子节点（或使用虚拟目标点）。
2) fan-out 动画 + expertsChosen 可复现算法。
3) camera follow（可开关）。

## 8. 测试与回归策略
由于是 HTML/JS，可采用“结构性测试”：
- 单元测试（Python/文本）：确保关键 marker、DOM id、badge 文案存在（防误导回归）。
- 端到端（可选）：Playwright 对控制条可点击、状态可切换进行冒烟。

## 9. 风险与缓解
- 性能：大模型节点多 → Phase 1 只做单 token；后续增加 LOD/批量粒子。
- 误解风险：用户把演示当真实推理 → 强制 badge + tooltip + 文案说明。
- 结构不全：某些模型字段缺失 → BLOCKED 或降级为 estimated（并显式标识）。

---

## 10. 验收标准（可验证）
1) 3D 中可一键启动播放：token 粒子从 embed 逐层移动至 head，且当前模块高亮。  
2) 可暂停、单步、倍速、跳转 token/layer。  
3) 在 DEMO 模式下页面明确标注 `Structure-driven`，不产生“伪真推理”误导。  
4) 2D/3D 同步：2D 面板显示的 tokenIndex/layerIndex 与 3D 当前一致。  

