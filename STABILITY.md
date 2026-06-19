# Vitriol 功能稳定性矩阵

> **版本**: v0.3.1  
> **最后更新**: 2026-06-18  
> **目的**: 明确每个功能的稳定性级别，为贡献者和用户提供清晰的 API 契约预期。

---

## 稳定性等级定义

| 等级 | 标识 | 语义 | API 变更策略 | 测试要求 |
|------|------|------|-------------|----------|
| **Stable** | 🟢 | 生产就绪，API 在 semver 内保持不变 | 仅向后兼容 | 100% 覆盖，CI 门禁 |
| **Beta** | 🟡 | 可用，接口可能微调 | 保留 1 个 minor 版本的弃用期 | 核心路径覆盖 |
| **Experimental** | 🔴 | 研究原型，无稳定性保证 | 随时可能重构或删除 | 仅 smoke test |
| **Deprecated** | ⚫ | 已废弃，将在下个大版本移除 | 不再接受新功能 | 仅回归测试 |

---

## 功能稳定性矩阵

### 核心引擎 (Core Engine)

| 功能 | 模块 | 等级 | 稳定化条件 | 预计稳定版本 |
|------|------|------|-----------|-------------|
| MinimalWeightGenerator | `core.generator` | 🟢 | 已稳定 | v0.3.0 |
| ConfigShrinker | `core.shrinker` | 🟢 | 已稳定 | v0.3.0 |
| Pipeline System | `core.pipeline` | 🟢 | 已稳定 | v0.3.0 |
| IncrementalGenerator | `core.incremental` | 🟢 | 已稳定 | v0.3.0 |
| ModelValidator | `core.validator` | 🟢 | 已稳定 | v0.3.0 |
| ModelAnalyzer | `core.analyzer` | 🟢 | 已稳定 | v0.3.0 |
| AdaptiveSharder | `core.adaptive_sharder` | 🟢 | 已稳定 | v0.3.0 |
| SmartInitializer | `core.smart_initializer` | 🟢 | 已稳定 | v0.3.0 |
| ParallelGenerator | `core.parallel_generator` | 🟢 | 已稳定 | v0.3.0 |
| ModelHasher | `core.hasher` | 🟢 | 已稳定 | v0.3.0 |

### 权重生成策略 (Strategies)

| 功能 | 模块 | 等级 | 稳定化条件 | 预计稳定版本 |
|------|------|------|-----------|-------------|
| Random | `strategies.random` | 🟢 | 已稳定 | v0.3.0 |
| Compact | `strategies.compact` | 🟢 | 已稳定 | v0.3.0 |
| Ultra | `strategies.ultra` | 🟢 | 已稳定 | v0.3.0 |
| Sparse | `strategies.sparse` | 🟢 | 已稳定 | v0.3.0 |
| StructuredSparse | `strategies.structured_sparse` | 🟢 | 已稳定 | v0.3.0 |
| Ternary | `strategies.ternary` | 🟢 | 已稳定 | v0.3.0 |
| Binary | `strategies.binary` | 🟢 | 已稳定 | v0.3.0 |
| Quantized | `strategies.quantized` | 🟢 | 已稳定 | v0.3.0 |
| LowRank | `strategies.lowrank` | 🟢 | 已稳定 | v0.3.0 |
| Learned | `strategies.learned` | 🟡 | 需要更多训练兼容性测试 | v0.4.0 |
| HybridLearned | `strategies.hybrid_learned` | 🟡 | 需要更多训练兼容性测试 | v0.4.0 |
| HybridUltra | `strategies.hybrid_ultra` | 🟢 | 已稳定 | v0.3.0 |
| Quantum | `strategies.quantum` | 🔴 | 需要论文级验证和量子计算社区反馈 | v0.6.0+ |

### 架构可视化 (Architecture Visualization)

| 功能 | 模块 | 等级 | 稳定化条件 | 预计稳定版本 |
|------|------|------|-----------|-------------|
| 3D Viewer (viewer.html) | `viz` | 🟢 | 已稳定 | v0.3.0 |
| Block Renderer | `arch_viz.renderers.block` | 🟢 | 已稳定 | v0.3.0 |
| Detail Renderer | `arch_viz.renderers.detail` | 🟢 | 已稳定 | v0.3.0 |
| HTML Renderer | `arch_viz.renderers.html` | 🟢 | 已稳定 | v0.3.0 |
| TransformerAnalyzer | `arch_viz.analyzers` | 🟢 | 已稳定 | v0.3.0 |
| QwenAnalyzer | `arch_viz.analyzers` | 🟢 | 已稳定 | v0.3.0 |
| DeepSeekAnalyzer | `arch_viz.analyzers` | 🟢 | 已稳定 | v0.3.0 |
| Qwen35Analyzer | `arch_viz.analyzers` | 🟢 | 已稳定 | v0.3.0 |
| Weight Inspector | `viz.weight_inspector` | 🟡 | 需要 safetensors 大文件测试 | v0.4.0 |
| Vocab Visualizer | `vocab_viz` | 🟢 | 已稳定 | v0.3.0 |

### NAS (Neural Architecture Search)

| 功能 | 模块 | 等级 | 稳定化条件 | 预计稳定版本 |
|------|------|------|-----------|-------------|
| Random Search | `nas.searcher` | 🟢 | 已稳定 | v0.3.0 |
| Evolutionary Search (GA) | `nas.searcher` | 🟢 | 已稳定 | v0.3.0 |
| Targeted Search | `nas.targeted_nas` | 🟡 | 需要更多约束组合测试 | v0.4.0 |
| RL Agent | `nas.rl_agent` | 🔴 | 需要收敛性验证和训练集成 | v0.5.0 |
| ArchitectureGene | `nas.search_space` | 🟢 | 已稳定 | v0.3.0 |
| LLMSearchSpace | `nas.search_space` | 🟢 | 已稳定 | v0.3.0 |
| HybridEvaluator | `nas.evaluator` | 🟡 | 需要更多零成本代理验证 | v0.4.0 |

### 架构进化 (Evolution)

| 功能 | 模块 | 等级 | 稳定化条件 | 预计稳定版本 |
|------|------|------|-----------|-------------|
| Evolution Tree | `evolution.tree_builder` | 🟢 | 已稳定 | v0.3.0 |
| Architecture Compare | `evolution.compare` | 🟢 | 已稳定 | v0.3.0 |
| Performance Simulator | `evolution.simulator` | 🟡 | 需要更多硬件 profile 校准 | v0.4.0 |
| Recommender | `evolution.recommender` | 🟡 | 需要更多用例覆盖 | v0.4.0 |
| Timeline | `evolution.timeline` | 🟢 | 已稳定 | v0.3.0 |
| Families | `evolution` | 🟢 | 已稳定 | v0.3.0 |

### KV Cache 压缩

| 功能 | 模块 | 等级 | 稳定化条件 | 预计稳定版本 |
|------|------|------|-----------|-------------|
| TurboQuant Runtime | `patches.kv_runtime_patches` | 🟢 | 已稳定 | v0.3.0 |
| KV Policy Presets | `kv.policy` | 🟢 | 已稳定 | v0.3.0 |
| KVCacheStore | `kv.cache_store` | 🟢 | 已稳定 | v0.3.0 |
| AdaptiveKVCodec | `kv.codec` | 🟡 | 需要更多注意力分布验证 | v0.4.0 |
| Sparse V | `patches.kv_runtime_patches` | 🟡 | 需要更多端到端精度测试 | v0.4.0 |
| Compute Skip | `patches.kv_runtime_patches` | 🟡 | 需要更多端到端精度测试 | v0.4.0 |
| Triton Kernels | `kv.triton_kernels` | 🟡 | 需要 GPU CI 环境 | v0.4.0 |
| **TurboQuantum** | `kv.turboquantum` | 🔴 | **研究假设，需要论文级验证** | v0.6.0+ |

### Benchmark & 评估

| 功能 | 模块 | 等级 | 稳定化条件 | 预计稳定版本 |
|------|------|------|-----------|-------------|
| KV Smoke | `bench.runner` | 🟢 | 已稳定 | v0.3.0 |
| KV Long | `bench.runner` | 🟢 | 已稳定 | v0.3.0 |
| KV Suite | `bench.runner` | 🟢 | 已稳定 | v0.3.0 |
| KV Report | `bench.runner` | 🟢 | 已稳定 | v0.3.0 |
| KV Plan | `bench.runner` | 🟢 | 已稳定 | v0.3.0 |
| KV Analyze | `bench.runner` | 🟢 | 已稳定 | v0.3.0 |
| PPL Evaluator | `bench.ppl_evaluator` | 🟢 | 已稳定 | v0.3.0 |
| AutoKV | `bench.autokv` | 🟡 | 需要更多模型覆盖 | v0.4.0 |
| TurboQuantum Benchmark | `bench.runner_turboquant` | 🔴 | 依赖 TurboQuantum 稳定化 | v0.6.0+ |

### 推理与部署 (Inference & Deployment)

| 功能 | 模块 | 等级 | 稳定化条件 | 预计稳定版本 |
|------|------|------|-----------|-------------|
| `infer` CLI | `cli.commands.infer` | 🔴 | 需要与 vLLM/TensorRT-LLM 集成验证 | v0.5.0 |
| `exobrain` CLI | `cli.commands.exobrain` | 🔴 | 需要知识蒸馏社区验证 | v0.5.0 |
| `trace` CLI | `cli.commands.trace` | 🔴 | 需要 trace 回放工具 | v0.5.0 |
| Web UI | `webui.app` | 🔴 | 需要 Gradio 稳定性测试和完整功能覆盖 | v0.4.0 |
| REST API | `api.server` | 🔴 | 需要 OpenAPI 验证和负载测试 | v0.4.0 |

### 安全与指纹

| 功能 | 模块 | 等级 | 稳定化条件 | 预计稳定版本 |
|------|------|------|-----------|-------------|
| FingerprintEngine | `utils.fingerprint` | 🟢 | 已稳定 | v0.3.0 |
| FingerprintRegistry | `utils.fingerprint` | 🟢 | 已稳定 | v0.3.0 |
| Security Context | `security/context` | 🟢 | 已稳定 | v0.3.0 |
| Config Security Audit | `security` | 🟢 | 已稳定 | v0.3.0 |

### CLI 命令

| 命令 | 等级 | 说明 |
|------|------|------|
| `check` | 🟢 | Golden path，已稳定 |
| `generate` | 🟢 | 已稳定 |
| `validate` | 🟢 | 已稳定 |
| `analyze` | 🟢 | 已稳定 |
| `batch` | 🟢 | 已稳定 |
| `cis` | 🟢 | 已稳定 |
| `export` | 🟢 | 已稳定 |
| `visualize` | 🟢 | 已稳定 |
| `viz` | 🟢 | 已稳定 |
| `arch-viz` | 🟢 | 已稳定 |
| `nas` | 🟡 | 接口稳定，但底层算法部分为实验性 |
| `evolve` | 🟢 | 已稳定 |
| `hash` | 🟢 | 已稳定 |
| `vocab-viz` | 🟢 | 已稳定 |
| `weight-viz` | 🟡 | 依赖 weight inspector 稳定化 |
| `bench` | 🟡 | 核心子命令稳定，TurboQuantum 子命令为实验性 |
| `infer` | 🔴 | 实验性 |
| `trace` | 🔴 | 实验性 |
| `webui` | 🔴 | 实验性 |
| `exobrain` | 🔴 | 实验性 |

---

## 稳定化路线图

### Phase 1: v0.4.0 — 核心硬化

目标：将所有 **Beta** 功能提升至 **Stable**（或明确标记为长期 Beta）。

- [ ] Learned / HybridLearned 策略训练兼容性测试 → Stable
- [ ] Weight Inspector 大文件测试 → Stable
- [ ] Targeted NAS 约束组合测试 → Stable
- [ ] AdaptiveKVCodec 注意力分布验证 → Stable
- [ ] Sparse V / Compute Skip 端到端精度测试 → Stable
- [ ] Web UI 功能覆盖测试 → Beta
- [ ] REST API OpenAPI 验证 → Beta
- [ ] Performance Simulator 硬件 profile 校准 → Beta

### Phase 2: v0.5.0 — 推理集成

目标：将推理相关实验性功能提升至 **Beta**。

- [ ] `infer` CLI 与 vLLM 集成验证 → Beta
- [ ] `exobrain` 知识蒸馏社区验证 → Beta
- [ ] RL NAS 收敛性验证 → Beta
- [ ] Triton Kernels GPU CI 环境 → Beta
- [ ] `trace` CLI trace 回放工具 → Beta

### Phase 3: v0.6.0+ — 前沿探索

目标：研究级功能保持 **Experimental**，但提供清晰的实验退出条件。

- [ ] TurboQuantum 论文级验证 → Beta（如果论文发表）或保持 Experimental
- [ ] Quantum 策略量子计算社区反馈 → Beta（如果硬件可行）或 Deprecated

---

## 实验性功能退出条件

| 功能 | 退出条件 (Experimental → Beta) | 退出条件 (Beta → Stable) |
|------|--------------------------------|------------------------|
| TurboQuantum | 1. 论文预印本发表；2. 至少 3 个独立复现结果；3. PPL 退化 < 5% | 1. 同行评审通过；2. 10+ 模型验证；3. 社区插件 ≥ 2 个 |
| RL NAS | 1. 5 个不同搜索空间收敛；2. 与 Evolutionary 对比 AUC 提升 > 10% | 1. 训练流水线集成完成；2. 100+ 次搜索零失败 |
| `infer` | 1. vLLM/TensorRT-LLM 集成 POC 完成；2. 3 个模型端到端验证 | 1. 生产部署案例 ≥ 1；2. 性能对比报告 |
| `exobrain` | 1. 知识蒸馏质量指标定义；2. 3 个模型蒸馏验证 | 1. 开源蒸馏模型 ≥ 1；2. 社区采用 ≥ 10 项目 |
| Web UI | 1. 所有 Stable 功能可访问；2. 无阻塞 bug | 1. 用户测试 ≥ 50 人；2. 性能测试通过 |
| REST API | 1. OpenAPI schema 完整；2. 负载测试通过 | 1. 第三方客户端 ≥ 1；2. 安全审计通过 |

---

## 贡献者指南

### 如何为实验性功能贡献代码

1. **实验性功能**（🔴）的 PR 可以突破常规架构，但需包含：
   - 清晰的研究假设说明
   - 可复现的实验脚本
   - 已知局限性文档

2. **Beta 功能**（🟡）的 PR 需要：
   - 向后兼容（如果修改公共接口）
   - 新增/更新的测试用例
   - 文档更新

3. **Stable 功能**（🟢）的 PR 需要：
   - 严格的向后兼容性
   - 100% 新增代码覆盖
   - 性能回归测试（如果适用）
   - 版本变更日志更新

### 废弃策略

- 功能标记为 **Deprecated**（⚫）后，保留 1 个 major 版本或 6 个月（以较长者为准）
- 废弃功能在文档中标记为 `@deprecated since vX.Y.Z`，并指向替代方案
- 废弃功能的 CLI 命令在运行时输出警告，但继续工作

---

> **维护承诺**: 本矩阵每 minor 版本更新一次。实验性功能的稳定性预期会在版本发布说明中明确说明。如有疑问，请在 Issue 中提出。
