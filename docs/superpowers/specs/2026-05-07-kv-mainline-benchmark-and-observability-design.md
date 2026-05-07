# 设计文档：KV 主链路收敛（压测基准）→ 观测闭环 → Hermes-Agent RL 适配性核查

日期：2026-05-07  
状态：Draft（待你确认后进入实现）  

## 1. 背景与问题陈述

Vitriol 当前在 **KV-Cache 推理/压缩** 与 **可视化/观测** 方面已经具备大量组件，但存在以下系统性问题：

1. **KV 推理有多条路径并存**：  
   - Store/Hook 路径（`CacheHookPatcher + UniversalAttentionPatcher + KVStoreBackend + KVCacheStore`）具备“packed 存储 + decode 侧读取”的完整能力。  
   - Runtime patch 路径（`KVRuntimePatcher` patch `torch.nn.functional.scaled_dot_product_attention`）更偏“算子扰动/加速实验”。  
   - 模型专用 cache patch（如 `Qwen35CachePatcher`）多为 qdq（量化再反量化）并不天然带来“存储压缩”。
2. **压测与回归口径不统一**：有 `infer/bench/ppl_evaluator`，但输出字段/统计来源分散，且“qdq vs packed”的含义容易混淆。
3. **观测闭环不完整**：存在 dashboard/metrics/trace 等模块，但缺少统一 run_id，以及统一指标注册/导出与关联方式。

我们希望按顺序完成三件事：

1) **先把 KV 主链路收敛 + 压测基准固化**（成为默认推荐路径，可回归）  
2) **再把 run_id + metrics/trace/dashboard 串成闭环**  
3) **最后核查 hermes-agent-rl（当前 workspace 中为 `output/.../hermes-agentic-rl`）是否满足“基于 hermes-agent 做 RL 优化”的要求**

---

## 2. 总体目标（Goals）

### 2.1 Phase 1：KV 主链路收敛 + 压测基准（必须）

**G1**：明确“默认推荐 KV 推理/压缩路径”为 **Store/Hook** 路径，并在 CLI/bench 中可一键启用。  
**G2**：形成“质量-性能-存储”三指标的基准输出，且可在 CI 中做回归比较。  
**G3**：清晰区分：  
- qdq（扰动/近似）≠ 存储压缩  
- packed（子字节/packed 表示）= 存储压缩  
避免用户用错路径导致“显存省了/没省”的误判。

### 2.2 Phase 2：观测闭环（必须）

**G4**：所有关键运行（infer/bench/ppl/trace）统一引入 `run_id`，并可关联到：
- 统一 metrics 导出（Prometheus text）  
- trace 文件（trace.v1）  
- dashboard（SSE/HTTP）  

### 2.3 Phase 3：Hermes-Agent RL 适配性核查（必须）

**G5**：输出一份“准入清单（pass/blocker）+ 风险 + 建议改动优先级”的审查报告，判断当前 hermes-agentic-rl 是否满足：
- 基于 NousResearch/hermes-agent 的真实 runtime 训练  
- 面向工具调用/多轮对话/分层记忆/子 agent 的 RL 优化要求  

---

## 3. 非目标（Non-goals）

Phase 1/2 的非目标：
- 不在本轮实现中追求所有 KV 创新模块（spectral/predictive/cross-layer/dict 等）都进入默认链路；它们保留为“可选策略”。  
- 不把 ExoBrain 推理/蒸馏链路强行纳入 KV 默认基准（除非后续明确要求）。  
- 不在本轮做深度 GPU kernel 性能调优（如 pack/unpack 全面 Triton 化），仅保证统计口径与链路可用性。

Phase 3 的非目标：
- 不在本轮直接重写 hermes-agentic-rl 的算法实现；只做“是否满足要求”的审查与差距分析（必要时给出后续改造建议/子任务拆解）。

---

## 4. 方案选择（Approach）

我们采用 **方案 A（推荐）**：

### 方案 A：收敛到 Store/Hook 主路径 + 基准三指标回归

**主路径**：  
`CacheHookPatcher`（cache.update hook）  
→ `KVStoreBackend.write_kv/read_attention`（每层一个 `KVCacheStore`）  
→ `UniversalAttentionPatcher`（decode q_len==1 尝试走 store attention）  
→ `KVCacheStore`（packed/residual proxy/compute_skip/zero-copy decode/sliding window 等）

**实验/扰动路径**（保留但降级为非默认）：  
- `KVRuntimePatcher`：算子级 patch（qdq/扰动/预处理缓存）  
- `Qwen35CachePatcher`：模型专用 cache.update 的 qdq  

**关键原则**：任何对外输出中明确标记 `kv_compute_path` 与 `kv_storage_path`，避免把 qdq 当成“节省存储/显存”。

---

## 5. Phase 1 详细设计：KV 主链路收敛 + 压测基准

### 5.1 统一运行结构：RunSummary（JSON 可序列化）

定义一份稳定的、可扩展的输出结构（infer/bench/ppl 均可复用）：

```json
{
  "run_id": "20260507-...uuid...",
  "model_id": "Qwen/Qwen2.5-0.5B",
  "preset": {"name": "balanced", "params": {}},
  "device": "cuda",
  "dtype": "float16",
  "kv": {
    "mode": "tuned|baseline",
    "compute_path": "store_hook|runtime_patch|vendor_cache_patch|none",
    "storage_path": "packed|raw|qdq_only|unknown",
    "estimated_kv_bytes": 12345678,
    "layer_stats": {
      "0": {"seq_len": 1024, "estimated_kv_bytes": 1234}
    }
  },
  "perf": {
    "prefill_s": 0.12,
    "decode_s": 0.34,
    "decode_toks_per_s": 88.1
  },
  "quality": {
    "exact": true,
    "prefix_match": [12, 16, 75.0],
    "ppl_baseline": 12.3,
    "ppl_tuned": 13.1,
    "ppl_degradation_pct": 6.5
  },
  "stats": {
    "cache_hooks": {"cache_update_calls": 1},
    "turboquant": {"calls": 42},
    "kv_runtime_patch": {"preprocess_cache_hit_rate": 0.9}
  }
}
```

> 说明：Phase 1 只需保证 `infer/bench/ppl` 至少能输出 `perf + kv(estimated bytes) + quality(token match 或 ppl)`；其余字段逐步补齐。

### 5.2 CLI 默认行为

#### `vitriol infer`
- 默认 preset：`balanced`（保持现状）  
- 增加/强化：明确输出 `kv_storage_path` 与 `kv_compute_path`（若当前实现已隐含，则补齐显示/JSON 字段）。  
- `--format json` 输出使用 RunSummary 结构（向后兼容：保留现有 text/summary 输出）。

#### `vitriol bench ...`
- bench runner 作为“基准输出的权威来源”：  
  - baseline（safe/exact） vs tuned（preset）  
  - 输出 chosen_n（已存在 `chosen_v_quantize_only_first_n`）  
  - 输出 kv_store_stats（来自 `KVStoreBackend.stats()`）  

### 5.3 “显存节省口径”纠偏：raw cache 常驻问题

当前 `KVCacheStore` 同时保留 `_k_raw/_v_raw` 和 `_k_enc/_v_enc`，会导致：
- `estimated_kv_bytes()` 只统计 encoded，  
- 但如果 raw 仍常驻，真实显存/内存并不等于 estimated。

因此引入一个明确的策略开关（命名待实现时按项目风格确定）：

- `keep_raw_cache: bool`（默认：True 以保守/兼容；benchmark/长上下文可设 False）  
- 当 `keep_raw_cache=False` 时：  
  - 完成 prefill 后可丢弃 raw（或只保留必要元信息）  
  - 如需重建（eviction/策略变化）则走“受控回退”路径（可选择禁止重建、或保留滚动窗口 raw）

**验收标准**：  
当 `keep_raw_cache=False` 且走 packed 存储时，“estimated_kv_bytes 与峰值内存”差距显著缩小（至少不再理论上被 raw 抵消）。

### 5.4 基准（Benchmark）与回归（Regression）设计

**最小可回归套件（CPU 可跑）**：
- 模型：小模型（如 Qwen 0.5B 或 tinyllama）  
- 提示：短 prompt suite（`bench.autokv.default_prompt_suite` 或 PPL evaluator 的 DEFAULT_PROMPTS）  
- 断言：  
  - safe 模式 `exact == True`  
  - tuned 模式 `prefix_match` 达到阈值（例如 ≥60%）  
  - `estimated_kv_bytes(tuned) < estimated_kv_bytes(baseline)`（仅当 storage_path=packed 才适用）

**扩展套件（GPU 运行，可选）**：
- 追加 decode tok/s、长上下文（`ultra-long`）稳定性指标。

---

## 6. Phase 2 详细设计：观测闭环（run_id + metrics + trace + dashboard）

### 6.1 RunContext 与 run_id 生成

定义 `RunContext`（可放在 `vitriol/telemetry` 或 `vitriol/utils`）：
- `run_id`：默认生成（时间戳 + uuid 短码）  
- `model_id/preset/device/dtype`：从 CLI/bench/ppl 传入  
- 可选：`git_sha`（若可获取）  

所有对外 JSON 输出必须包含 run_id。

### 6.2 统一指标注册（MetricsCollector）

将以下来源的 stats 统一写入 `telemetry.metrics`：

- `cache_hooks.get_cache_hook_stats()`  
- `KVStoreBackend.stats()`（layers/seq_lens/estimated_kv_bytes）  
- `turboquant.get_turboquant_stats()`  
- `KVRuntimePatcher.stats()`（如果该路径启用）  

导出：Prometheus text（沿用 `to_prometheus_format()`），并在 CLI 提供 `--metrics-out <path>` 或 `--print-metrics`（具体形式在实现阶段定）。

### 6.3 Trace 与 run_id 关联

`cli trace` 的 trace.v1 JSON 结构增加：
- `run_id`  
- `kv_summary`（可选，默认关闭；以避免 trace 过大）  

### 6.4 Dashboard 与 run_id 关联

`viz.dashboard` 增加：
- 展示当前 active run_id  
- events/metrics 按 run_id 分组（或至少可过滤）

---

## 7. Phase 3：hermes-agentic-rl（基于 hermes-agent 的 RL 优化）适配性核查设计

### 7.1 核查对象

workspace 中现有路径（以实际存在为准）：
- `output/.../hermes-agentic-rl/`（含 docs、scripts、hermes_rl 包等）

### 7.2 准入清单（Audit Checklist）

输出报告将按以下维度给出：**Pass / Blocker / Risk / Recommendation**：

1. **Hermes-Agent 桥接真实性**：是否对接真实 hermes-agent runtime（工具调用/记忆/子 agent），而非玩具 env。  
2. **MDP 定义完整性**：state encoder 是否覆盖对话历史、tool registry、分层记忆、scratchpad；action space 是否可表达 tool-call 参数化、memory op、delegate、respond/terminate。  
3. **奖励与约束**：是否具备分层奖励与约束优化（Lagrangian/RCPO/CPO 等），并与 Hermes-Agent 的安全策略一致。  
4. **算法闭环**：在线（PPO/GRPO/IMPALA）+ 离线（BC/AWR/DPO）+ replay buffer + checkpoint/registry。  
5. **评测与 A/B**：eval harness、benchmark runner、A/B 显著性与报告产物是否齐全。  
6. **部署与推理**：导出/推理 backend（HF/vLLM/TensorRT-LLM/LoRA 热切换）是否可用且接口稳定。  
7. **可复现性**：config/seed/版本化/manifest 是否完整；是否能 CPU smoke 跑通。

### 7.3 报告产物

生成文件：
- `docs/superpowers/reports/2026-05-07-hermes-agentic-rl-readiness-audit.md`

并包含：
- 总结（是否满足要求）  
- blockers 列表（必须修复项）  
- 风险项与建议（按 P0/P1/P2）  
- 最小可复现实验（smoke 命令、期望输出）

---

## 8. 实现计划的输入（为下一步 writing-plans 准备）

当你确认本设计文档后，实现计划（tasks）将拆为三阶段，每阶段都有可验证验收点：

1) Phase 1：KV 主链路收敛 + 基准输出稳定（CI 可跑）  
2) Phase 2：run_id + metrics/trace/dashboard 闭环  
3) Phase 3：hermes-agentic-rl readiness audit 报告

---

## 9. 风险与对策

1. **“显存节省”误判风险（raw 常驻）**：通过 `keep_raw_cache` 或受控回退策略降低。  
2. **transformers patch 兼容性风险**：`UniversalAttentionPatcher` 依赖 `ALL_ATTENTION_FUNCTIONS.get_interface`，需提供 graceful fallback 并记录 stats。  
3. **指标/输出格式变更风险**：通过新增 JSON format（不破坏现有 text/summary）与版本化 schema 规避。  

---

## 10. 验收标准（Acceptance Criteria）

### Phase 1
- `infer/bench` 在默认推荐路径上可稳定输出：perf + kv estimated bytes + 质量（prefix match 或 ppl）  
- 基准输出可在 tests/CI 中以阈值断言通过  
- 输出中明确标记 compute_path 与 storage_path

### Phase 2
- run_id 贯通 infer/bench/ppl/trace  
- metrics 可导出（Prometheus text），且能覆盖 KV 关键统计  
- dashboard 能展示/过滤 run_id（最低限度：显示当前 run_id）

### Phase 3
- 产出 readiness audit 报告，给出“是否满足要求”的结论与 blockers

