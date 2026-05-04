# 离线 Trace 生成与回放（Token-by-Token）设计稿
Generated: 2026-04-28  
Rules-Ver: 3.0 (project)  
Context-ID: VIZ-TRACE-REPLAY  

## 1. 目标
在现有 2D/3D 模型结构可视化基础上，新增“**离线 Trace 回放**”能力：
- 使用 **hybrid-ultra / ultra 策略生成的最小模型权重**（你指定：`output/tinyllama-hybrid-ultra-test`）进行一次真实推理（CPU 也可）。
- 将推理过程导出为结构化 Trace（JSON），包含 **prompt tokens + 生成 tokens**，以及每个 token 在模型结构中的“经过节点序列（layer/attn/mlp）”。
- 3D 作为权威主视图：加载 Trace 后按 token 顺序回放（粒子移动/节点高亮），并提供与当前 playback toolbar 兼容的控制。
- 2D 作为辅助读数：展示 token 文本、layer 进度、当前事件信息。

> 这是真实推理驱动的“离线回放”（可复现），不是纯结构随机动画；但仍不追求展示真实 attention 矩阵数值（除非未来扩展）。

## 2. 约束与注意事项
1) `tinyllama-hybrid-ultra-test` README 明确提示“minimal/dummy weights，NOT intended for inference”。  
   - 但该模型仍可被 transformers 加载并运行，用于 **打通 trace 链路**。
   - v1 默认只跑很短：`max_new_tokens=8`，避免耗时/数值不稳定。
2) v1 只保证 “模块级路径”真实（哪些层/子模块被执行），不保证：
   - top-k MoE 专家选择（TinyLlama 无 MoE）
   - attention 权重、激活值、KV cache 数值细节

## 3. Trace Schema（v1）
文件：`trace.json`

```json
{
  "schema_version": "trace.v1",
  "generated_at": "2026-04-28T00:00:00Z",
  "model_path": "output/tinyllama-hybrid-ultra-test",
  "prompt": "hello",
  "max_new_tokens": 8,
  "device": "cpu",
  "tokenizer": {
    "name_or_path": "...",
    "type": "AutoTokenizer"
  },
  "tokens": {
    "prompt_token_ids": [ ... ],
    "prompt_tokens": [ "...", "..." ],
    "generated_token_ids": [ ... ],
    "generated_tokens": [ "...", "..." ]
  },
  "events": [
    {
      "token_index": 0,
      "phase": "prefill|decode",
      "node_path": ["embed", "block:0:attn", "block:0:mlp", "...", "lm_head"]
    }
  ]
}
```

### 3.1 node_path 约定
为了能直接复用现有 3D nodeIndex：
- `embed`
- `block:{i}:attn`
- `block:{i}:mlp`（Llama 系）
- `lm_head`

> 注意：当前 3D 的索引 key 里是 `ffn`/`moe`，因此我们会 **兼容映射**：`mlp → ffn`（对 Llama）。

## 4. Trace 生成方式（真实推理，离线）
v1 使用一个“最小可控”的 greedy decode 循环，而不是直接 `model.generate()`，以便：
- 每步 decode 明确对应一个新 token
- 方便对齐 hook 事件到当前 step

实现要点：
1) 用 transformers 加载 tokenizer/config/model（参考 output README 的用法）。
2) 注册 forward hooks：
   - embedding（可选：用固定事件 `embed` 代替）
   - 每个 transformer layer 的 `self_attn` 与 `mlp` 子模块（Llama 模型结构）
   - lm_head（可用固定事件 `lm_head` 代替）
3) 每次 forward（每个 decode step）只记录一次 `node_path`：
   - `["embed"] + 逐层(["block:i:attn","block:i:mlp"]) + ["lm_head"]`

## 5. 3D 回放集成
### 5.1 注入方式（与现有 viz CLI 对齐）
扩展 `vitriol cli viz` 注入逻辑：
- 在 HTML 中新增 marker：`// INLINE_TRACE_MARKER`
- CLI 读取 trace.json 并注入：
  - `window.__VITRIOL_TRACE__ = {...}`

### 5.2 PlaybackEngine 增强
现有 PlaybackEngine 增加：
- `loadTrace(trace)`：将 trace.events 转成可回放 path（nodeId 列表）
- `setMode('structure'|'trace')`

3D 渲染保持不变：依然使用粒子插值 + highlightNode，只是 path 来源从“结构默认路径”切换为“trace node_path”。

## 6. 2D 同步增强
在现有 `playbackStatusPanel` 基础上补充：
- 当前 token 文本（prompt+generated 合并视角）
- 当前 phase（prefill/decode）
- 当前 nodeId（便于讲解）

## 7. 验收标准（v1）
1) 能对 `output/tinyllama-hybrid-ultra-test` 运行一次推理并产出 `trace.json`。  
2) 3D 可视化能加载该 trace，并可播放/暂停/单步/倍速/seek。  
3) 回放时粒子移动与节点高亮与 trace 的 node_path 一致。  
4) 页面明确标识来源：`TRACE (offline replay)`，避免误导。  

