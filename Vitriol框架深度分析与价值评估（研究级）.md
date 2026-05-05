# Vitriol 框架深度分析与价值评估（研究级）
> 面向：学术研究 + 工程落地 + 使用者上手 + 源码实现视角  
> 依据：仓库内 README/README_CN、已有静态审阅文档与设计稿，以及对 `src/vitriol/` 关键实现文件的抽样阅读  
> 注：你之前提醒“要分析的是 vitriol”，本报告已以 **Vitriol（⚗️）** 为对象重做分析

---

## 0. 摘要（Abstract）

Vitriol 是一个围绕 **“结构（Structure）—权重/数据（Weights/Data）解耦”** 的统一框架：它试图把对大模型（LLM/VLM）架构的研究、可视化、压缩/量化实验、KV Cache 推理优化，以及（轻量）架构搜索（NAS）等能力，组织到同一条可复用工具链中。

其最核心的工程创新在于：**只依赖 KB 级 `config.json` 即可在 Meta Device 上构造“零内存”模型骨架**，随后用多种“权重生成策略”（Random/Compact/Ultra/Quantized/LowRank/…）填充张量，以生成结构兼容的权重分片，从而在不下载真实 GB/TB 权重的前提下完成：

1. 架构分析与可视化（包括面向 MoE/GQA/MQA/MLA/多模态的专用分析器与 2D/3D viewer）
2. 压缩/量化研究（尤其是 KV Cache TurboQuant/TurboQuantum 与策略系统）
3. NAS/进化工具（约束优化、Pareto、多策略搜索、家族树/对比/模拟）
4. 复现与治理（实验报告导出、指纹哈希、以及对 `trust_remote_code` 风险的显式提示）

从学术角度，Vitriol 为“架构研究的可复现实验范式”提供了一个可操作平台：通过结构-权重解耦，可在较低成本下做架构对比、推理优化策略对比、以及基于可解释指标的压缩权衡研究。  
从工程角度，它提供了一个“面向模型生态演进的工具栈”：在 CI/CPU 环境中快速验证加载链路、生成可视化、做策略 A/B 报告，从而显著降低大模型工程试验的资源门槛。

---

## 1. 框架定位：Vitriol 解决的核心问题是什么？

### 1.1 资源门槛问题：架构研究与工程验证被权重体量“绑架”
传统流程中，想“看一眼架构/跑一次 load/做一次 KV 实验”往往要先：
- 下载（或挂载）GB~TB 级权重
- 准备 GPU/大磁盘/复杂环境

Vitriol 通过 **Skeleton + Algorithmic Weights** 的组合，把大量“前置门槛”转移到“结构层”和“策略层”，从而让：
- 架构检查/可视化/对比：只需配置
- pipeline 验证（加载、分片、tokenizer、最小推理 smoke）：可用生成权重替代
- 推理优化研究（KV Cache 策略、patching、benchmark）：可在更快迭代周期内完成筛选与实验记录沉淀

### 1.2 混淆变量问题：榜单提升来自架构还是训练数据？
Vitriol 在 README 中明确提出：模型性能是结构与数据的混淆变量。  
通过对同一结构生成不同“权重数据分布”（random/compact/ultra…），或者对不同结构统一使用相同“权重生成策略”，可以构造“受控对比”的实验环境，为以下研究问题提供工具支撑：
- 架构差异对某些 proxy 指标（如 FLOPs、KV 规模、理论 expressivity、路由复杂度）的影响
- KV 压缩策略对质量（PPL、token match、KL drift）与吞吐/显存的权衡
- 结构复杂度（MoE/MLA/GQA）与推理优化策略适配性的关系

---

## 2. 总体架构：从“配置”到“可运行实验”的分层

### 2.1 三阶段解耦流水线（配置 → 骨架 → 权重）

从 README 与 `core/generator.py` 的实现可以抽象为：

1. **Config → Structure**
   - 输入：HF model id 或本地路径（含 `config.json`）
   - 输出：`PretrainedConfig`（结构参数全集）
2. **Structure → Skeleton（Meta Device / init_empty_weights）**
   - 目标：得到 **参数三元组集合** `(name, shape, dtype)`，但不分配真实存储
3. **Skeleton → Weights（Strategy.generate_tensor）**
   - 对每个参数调用策略生成权重张量，并按 shard/index 结构写出

这条流水线的关键工程价值：**把“研究/工具链”的主要数据依赖，从“真实权重”下沉到“参数的形状与命名空间”**。

### 2.2 CLI 作为“统一操作面”：LazyGroup + 全局安全开关

在 `src/vitriol/cli/main.py` 中：
- 以 `LazyGroup` 惰性加载子命令，降低 CLI 启动开销
- 提供全局开关 `--trust-remote-code/--no-trust-remote-code`，并放入 `ctx.obj["trust_remote_code"]`

这体现了 Vitriol 的一个重要工程判断：
> 兼容性（trust_remote_code=True）与安全性（False）不可兼得，需要明确暴露为可治理的运行策略。

但从代码扫描结果看，部分路径仍存在 `trust_remote_code=True` 写死的情况（例如 evolve/webui/nas evaluator 等），这为“安全一致性闭环”留下了后续工程改进空间（见第 9 节）。

---

## 3. 核心模块深解（源码视角）

本节从“关键对象/关键数据结构/关键调用链”解释 Vitriol 的实现组织方式。

### 3.1 `GenerationConfig` 与 `SecurityOptions`：配置的三层合并

`src/vitriol/config/manager.py` 定义了：
- `GenerationConfig`：策略、dtype、分片大小、策略参数（n_bits/rank/sparsity）、以及 `security`
- `SecurityOptions`：`trust_remote_code / allow_network / local_files_only`
- `build_generation_config`：将环境变量、YAML、overrides 进行合并，并单独解析 security

这一结构使得“安全策略”可以作为 **横切关注点（cross-cutting concern）** 被传递到 generator/validator/analyzer/API job 中。

### 3.2 `MinimalWeightGenerator`：生成链路的关键工程点

`src/vitriol/core/generator.py` 中的 `MinimalWeightGenerator` 体现了 Vitriol 的“框架化”程度：

**(1) Strategy registry 统一入口**
- `self.strategy = get_strategy(...)` 将策略作为插件式组件注入

**(2) ultra 默认 shrink_config**
- `ultra` 策略默认启用 shrink_config（减少维度、最小化输出）
- `_shrink_config` 中针对 vision 子配置做了特殊最小维度，避免 VLM 维度校验失败（这反映出作者在“兼容多模态模型 from_config()”方面的工程经验）

**(3) “真实 storage bytes” 的 size 估算**
- `_estimate_tensor_nbytes` 优先取 `untyped_storage().nbytes()`
- 这是为 ultra 的 stride=0 张量做的关键修正：逻辑 `numel` 巨大，但底层 storage 只有极少元素

**(4) shard map 与 index 兼容**
- `_get_original_shard_map` 尝试从 HF index 获取 `weight_map`，并兼容本地路径
- 目的：输出的 shard/index 尽量与原模型一致，保证“结构兼容可加载”

### 3.3 权重策略系统：从“算法思想”到“可声明能力”

`src/vitriol/strategies/base.py` 设计了：
- `StrategyCapabilities`：是否支持 safetensors、是否支持训练、压缩率上界等
- `WeightGenerationStrategy`：统一接口 `generate_tensor` 与 `save_shard`

这在工程上很关键：  
它把“策略差异”从散乱实现，抽象为可声明的能力系统，为：
- CLI 选择策略/格式谈判（safetensors vs pytorch）
- 用户理解策略边界（是否可训练、是否适合 inference）
提供了结构化信息。

**Ultra 策略的代表性意义：**
- `UltraStrategy` 通过 `torch.as_strided(storage, shape, strides=[0..0])` 实现“1 个元素代表任意大张量”
- 明确声明：不支持 safetensors、不支持训练
- 该策略既是“极致压缩黑科技”，也天然引入研究讨论点：这种张量对下游工具、训练框架、序列化格式的兼容边界在哪里？

### 3.4 KV Cache 推理优化：运行时 patch + policy presets

#### 3.4.1 运行时 patch：对 SDPA 的 monkey-patch
`src/vitriol/patches/kv_runtime_patches.py`：
- 定义 `KVRuntimePatchConfig`（turbo quant / adaptive bits / sparse v / compute skip 等）
- `KVRuntimePatcher.apply()` 将 `torch.nn.functional.scaled_dot_product_attention` 替换为 patched 版本

关键实现点：
- decode-only gating（只在 decode 阶段 patch）
- 预处理缓存（OrderedDict LRU，限制容量）
- TurboQuant（按 block size、k/v bits/format）
- AdaptiveKVCodec（基于注意力熵/峰度的自适应位宽，支持 FWHT 旋转）
- compute-skip / sparse-v（跳过计算或跳过 V 块加载）

这套实现具有较强研究含义：它把 KV 压缩从“离线量化”推进到“在线运行时策略”，并提供可观测统计（cache hit rate、calls_patched 等）。

#### 3.4.2 Policy preset：面向场景的策略组合
`src/vitriol/kv/policy.py` 提供 `KVPolicyPreset`：
- `safe / balanced / fast-balanced / aggressive / ultra-long / ...`
- 以结构化 params 描述“哪些层量化、从哪个 token 开始量化、是否启用 sparse-v/compute-skip 等”

这使得 KV 优化具备“实验可复现单位”：同一 preset 在不同模型上可批量跑 benchmark，并输出统一报告（见第 4.3）。

### 3.5 NAS 与架构设计空间：约束优化 + Pareto

`src/vitriol/nas/targeted_nas.py` 展示了一个相对“轻量但可用”的 NAS 方向：
- `ConstraintOptimizer.optimize`：满足约束（params/vram 等）后做随机采样+目标打分
- `MultiObjectiveOptimizer`：维护 Pareto front（dominates 判定、非支配解更新）

它不像训练驱动 NAS 那样昂贵，更偏向“结构指标驱动/约束驱动”的架构探索工具，契合 Vitriol 的“低成本架构实验”定位。

### 3.6 模型指纹：可审计/可追溯的补充机制

`src/vitriol/utils/fingerprint.py` 中的 `FingerprintEngine`：
- 组合 `architecture_hash + weights_hash → content_hash`
- `signature = sha256(model_id:content_hash:secret_key)[:32]`
- 支持 compare_models（架构相似、权重相似）

学术/工程意义：
1. 工程：可用于版本追踪、数据集/模型市场的完整性验证（哪怕是 demo/生成权重）
2. 研究：可把实验对象（架构+策略+权重）纳入可复现标识体系，减少“实验对象漂移”

---

## 4. 使用者视角：Vitriol 能做什么？怎么用？

### 4.1 三类典型用户画像

1. **架构研究者**：看结构、做对比、做进化谱系、做 NAS 约束探索  
2. **推理优化研究者/工程师**：研究 KV cache 压缩、策略 A/B、吞吐与质量权衡  
3. **平台/CI 工程师**：在 CPU/低资源环境中验证加载链路、分片、tokenizer、可视化工件

### 4.2 最小权重生成闭环
- `vitriol generate <model_id> --strategy compact|ultra|random ...`
- `vitriol validate <output_dir>`

关键注意事项：
- ultra 的输出格式偏 `.bin`（stride=0 与 safetensors 不兼容）
- random 策略更接近“可训练性验证”，compact/ultra 更接近“加载/可视化/结构验证”

### 4.3 KV 量化与基准测试闭环
- `vitriol bench kv-smoke/kv-long/kv-suite/kv-report ...`
- 输出支持 `summary/json/markdown`，有利于把一次实验固化为“实验记录工件”
- 结合 policy preset，可以快速做：
  - `balanced` vs `ultra-long` 的速度/显存/质量对比
  - 特定层启用/禁用 residual sketch 的影响对比

### 4.4 架构可视化与演化工具
- `vitriol arch-viz`：静态 HTML/图
- `vitriol viz`：2D/3D 交互 viewer
- `vitriol evolve tree/compare/simulate/timeline/recommend`：谱系/对比/模拟/推荐

### 4.5 WebUI / REST API
Vitriol 提供 Gradio WebUI 与实验性 FastAPI。  
根据已有审阅文档，API 端目前存在：
- `/nas/search` 为模拟占位（非真实 NAS 执行）
- `/models` 为硬编码样例  
这不影响 CLI 主线，但会影响“对外服务化”闭环（见第 9 节改进建议）。

---

## 5. 工程落地价值：为什么它不是“玩具工程”？

### 5.1 资源成本的数量级降低（可量化）
Vitriol 的价值可以用一个简单公式概括：
> 把“实验成本”从 O(weight_size) 降到近似 O(config_size + strategy_cost)

这对以下场景极具吸引力：
- 模型生态快速迭代时的架构普查（大量模型/版本）
- CI 中的“加载链路回归”（无需每次下载权重）
- 推理策略筛选（先在小模型或生成权重上快速筛选，再投入真实权重做严谨评测）

### 5.2 把研究工具链工程化：preset、报告、指纹、静态站点
Vitriol 在“工具化产物”方面明显更系统：
- preset（policy）是可复用实验单位
- bench 输出可沉淀为 markdown 实验记录
- viewer/docs 可部署为静态站点（GitHub Pages）
- 指纹系统提供可追溯标识

这些都是把研究从“脚本堆”推向“可复现系统”的关键基础设施。

### 5.3 风险显式化：trust_remote_code 的治理入口
Vitriol 明确提示并暴露 `trust_remote_code` 开关，这一点对工程落地很重要：  
它承认 HF 生态的现实（大量模型需要 remote code 才能加载），同时给 CI/共享环境提供更安全的运行模式。

---

## 6. 学术意义：Vitriol 可能贡献哪些研究问题与方法范式？

### 6.1 “结构—权重解耦”作为可复现实验范式（Methodology）
在 LLM 研究中，很多对比分析无法摆脱训练数据、超参、权重初始化、以及授权模型不可得等因素。  
Vitriol 提供一种可操作的范式：
- 固定结构、改变“权重生成分布”（策略）
- 或固定策略、改变结构

它虽然不能回答“真实语义能力来自哪里”，但非常适合回答：
1. 架构差异如何影响**资源指标**（FLOPs/VRAM/KV footprint）
2. 推理优化策略如何在不同结构上呈现不同收益曲线
3. MoE/MLA/GQA 等结构特征与 KV 压缩策略的相互作用

### 6.2 KV Cache 压缩：从固定比特到策略系统与自适应编码
Vitriol 的 KV 路线不是单点算法，而是：
1) runtime patching 机制  
2) 自适应编码（entropy/kurtosis + FWHT）  
3) policy presets（按场景组合）  
4) bench suite（报告输出）  

研究上可形成一套系统性的对比框架，而非仅“提出一个算法、跑一个表”。

### 6.3 轻量 NAS：约束/多目标驱动的“架构筛选器”
Vitriol 的 targeted NAS 更像是“结构搜索空间 + 约束优化器”，这对工程与研究都现实可行：
- 不依赖训练
- 可以快速在可解释指标上做 Pareto 探索
- 更适合与架构可视化/模拟/推荐模块联动

### 6.4 可追溯与实验对象治理：指纹哈希的研究化表达
指纹系统（architecture/weights/content/signature）可以支撑：
- 实验对象版本化
- 模型 lineage/变体检测
- 实验数据集的可追溯治理

在可复现性研究（reproducibility）语境下，这是有价值的基础设施方向。

---

## 7. 可复现实验与评测设计（建议写进论文/技术报告的部分）

### 7.1 研究问题（RQs）
- RQ1：在相同模型与数据集上，KV policy presets 的吞吐/显存/质量（PPL、token match、KL drift）权衡曲线如何？
- RQ2：不同架构特征（GQA/MQA/MLA、MoE、层数/hidden size）对 KV 压缩收益与退化敏感性有何影响？
- RQ3：自适应编码（AdaptiveKVCodec/FWHT）相比固定 turbo bits 是否能在同等 bpv 下提升质量？
- RQ4：结构—权重解耦下，哪些“架构指标代理”（params、FLOPs、routing complexity）与真实推理性能/质量最相关？

### 7.2 基线（Baselines）
建议至少包含：
- no-quant（safe）
- 固定 turbo3（balanced/fast-balanced）
- aggressive/ultra-long（更强压缩）
- 自适应位宽（AdaptiveKVCodec）
-（若做 TurboQuantum）与其模式对比

### 7.3 指标（Metrics）
Vitriol 已覆盖/暗示的指标可以系统化为：
- 质量：PPL、token match rate、logit KL divergence
- 性能：tok/s、prefill/decode 时间、calls_patched/bypassed
- 资源：estimated KV MB、peak device MB、非 KV gap
- 策略解释性：逐层策略表（kv-plan）、逐层误差表（kv-analyze）

### 7.4 实验工件（Artifacts）
建议把以下作为论文/报告附录工件输出：
- `kv-report` 的 md/json
- policy preset 配置快照
- 模型指纹（signature）或 arch hash（避免对象漂移）
- 环境与依赖（torch/transformers/triton 版本）

---

## 8. 局限性与风险（Critical Review）

### 8.1 生成权重的外推风险
最小权重/替代权重并不用于真实语义推理，因此：
- 结构分析/资源分析是可靠方向
- 语义质量结论必须基于真实权重与真实推理链路

合理定位是：**先用低成本工具筛选策略与假设，再投入真实权重做严谨验证**。

### 8.2 monkey-patch 的兼容性与可维护性
对 `scaled_dot_product_attention` 的运行时 patch 具有侵入性：
- 对不同 torch/transformers 版本可能敏感
- 与其他优化框架（如 flash-attn、xformers、vLLM）可能存在冲突

工程上需要：
1) 明确 patch 的启用/回滚策略  
2) 在 CI 中对关键版本做回归矩阵  

### 8.3 `trust_remote_code` 一致性缺口（安全治理）
虽有全局开关，但部分路径仍写死 `trust_remote_code=True`。  
这会导致：
- CLI 的安全模式不完全可信
- WebUI/API 与 CLI 行为不一致

这是一个明确、可修复、且对外可信度影响较大的工程缺口。

### 8.4 API 的“真实性闭环”未完成
现有 API 的 `/nas/search` 为模拟占位；`/models` 为硬编码样例。  
如果把 Vitriol 作为服务对外开放，这会成为主要短板（但不影响本地 CLI/研究使用）。

---

## 9. 面向下一阶段的工程改进建议（优先级）

结合仓库内现有的 gap-closure 设计稿与代码扫描，建议按以下优先级推进：

### P0：安全一致性闭环（trust_remote_code 全链路贯彻）
目标：CLI/WebUI/API 的所有 HF 加载路径遵从同一“事实来源”。  
建议做法：
- 将 `trust_remote_code` 从 ctx/config 向下传递到 analyzer/evolve/webui/nas evaluator 等所有 Auto* 调用点
- 增加单测：在 `--no-trust-remote-code` 下断言调用参数

### P0：API NAS 真实性闭环
目标：`/nas/search` 调用真实 NAS（至少 targeted constraint optimizer），产出 artifacts 与可查询 job result。  
价值：
- 让“服务化”具备可验证性
- 也便于 WebUI 直接复用同一后端

### P1：API `/models` 动态化（信息可信）
目标：输出真实 families/adapters/capability matrix，而非硬编码示例。

### P1：工程发布卫生（仓库体积、缓存清理）
如果该仓库作为发布/开源主仓库，建议清理 `output/`、`__pycache__` 等大体量目录的追踪历史，降低 clone/CI 容量与审计成本。

---

## 10. 结论：项目价值与潜在学术/工程意义

### 10.1 工程价值（可落地）
Vitriol 的核心工程价值不是“又一个模型工具”，而是把大模型生态中的若干高频任务（生成/验证/可视化/对比/推理策略实验）统一进 **可脚本化、可报告化、可复用 preset** 的工作流，并显著降低资源门槛。

在团队/组织层面，它能支撑：
- 架构巡检与对比的自动化
- 推理策略的快速 A/B 筛选与实验记录沉淀
- CI 中的模型加载链路回归（无需真实权重）

### 10.2 学术意义（可发表）
Vitriol 提供了一个有潜力产出论文叙事的系统：
1) 结构—权重解耦的实验范式（methodology contribution）  
2) KV 压缩的策略系统（system + evaluation contribution）  
3) 轻量 NAS + 架构演化/模拟/推荐的整合（tooling contribution）  
4) 可追溯工件（bench md/json、fingerprint）提升复现性（reproducibility contribution）

如果以论文为目标，建议把贡献点收敛为：
- “KV Cache 策略系统 + 可复现评测框架”，并把 Vitriol 作为系统实现
或
- “结构-权重解耦驱动的架构研究平台”，并用多个案例展示其降低成本与提升可复现性的能力边界

---

## 附录 A：关键源码入口索引（便于读代码）

- CLI 入口：`src/vitriol/cli/main.py`
- 配置与安全：`src/vitriol/config/manager.py`
- 最小权重生成：`src/vitriol/core/generator.py`（`MinimalWeightGenerator`）
- 策略系统：`src/vitriol/strategies/*`（`WeightGenerationStrategy`）
- KV runtime patch：`src/vitriol/patches/kv_runtime_patches.py`
- KV policy preset：`src/vitriol/kv/policy.py`
- NAS targeted optimizer：`src/vitriol/nas/targeted_nas.py`
- 指纹引擎：`src/vitriol/utils/fingerprint.py`
- API server：`src/vitriol/api/server.py`

---

## 附录 B：最小可复现实验清单（MVP Repro Checklist）

> 目标：让读者在“最少步骤”下复现 Vitriol 的三条主线能力：  
> (1) 结构-权重解耦生成闭环；(2) KV 策略 A/B 基准闭环；(3) 可视化/对比闭环。

### B.1 环境与依赖记录（建议作为实验头）

至少记录：
- OS / Python 版本
- `torch`, `transformers`, `triton`（若用）版本
- GPU 型号与显存（若跑 long-context）

### B.2 结构-权重解耦闭环（CPU 即可）

```bash
# 1) 生成最小权重（compact）
vitriol generate gpt2 -o output/gpt2-compact --strategy compact

# 2) 验证可加载与 tokenizer
vitriol validate output/gpt2-compact

# 3) 生成极致压缩（ultra）并验证（注意：通常输出为 .bin）
vitriol generate gpt2 -o output/gpt2-ultra --strategy ultra
vitriol validate output/gpt2-ultra --no-inference
```

### B.3 KV Cache 策略 A/B（推荐先 smoke）

```bash
# 1) 快速 sanity：balanced vs fast-balanced
vitriol bench kv-smoke gpt2 --preset balanced --compare-preset fast-balanced --format markdown --output kv-smoke-ab.md

# 2) 导出逐层策略表（便于解释结果）
vitriol bench kv-plan gpt2 --preset balanced --format markdown --output kv-plan-balanced.md

# 3) 离线误差分析（不必等完整 decode）
vitriol bench kv-analyze gpt2 --preset balanced --compare-preset fast-balanced --format markdown --output kv-analyze-ab.md
```

### B.4 可视化闭环（架构 HTML + 交互式 viewer）

```bash
# 1) 静态 HTML（适合归档）
vitriol arch-viz gpt2 --html -o output/gpt2-arch.html

# 2) 交互式 viewer（2D/3D）
vitriol viz gpt2 --2d
vitriol viz gpt2 --3d
```

### B.5 安全模式回归（必做）

```bash
# 关键：验证 --no-trust-remote-code 是否在你关心的命令链路上真正生效
vitriol --no-trust-remote-code analyze gpt2
vitriol --no-trust-remote-code evolve families
```

如果在某些命令上仍出现 `trust_remote_code=True` 的内部硬编码行为，建议将其作为：
- 论文中的“系统局限性/威胁”说明
- 或工程上优先修复的 P0 闭环项

---

# 深度扩展章节（更深入：算法/数学 + 源码调用链 + 评测/论文叙事 + 工程闭环）

> 说明：以下内容对应你选择的 4 个加深方向：**源码级深挖、算法/数学细节、系统评测与论文叙事、工程化与安全闭环**。  
> 写法上我会尽量做到“每个算法点都能在源码里定位到某个函数/数据结构”，并给出“可发表的实验设计”。

---

## 11. TurboQuant（论文版）深挖：从数学到实现的逐步对照

Vitriol 的 TurboQuant 实现位于：`src/vitriol/patches/turboquant.py`。文件头部给出经典流水线：

> `x → Hadamard Rotation → Standardize → Lloyd-Max Quantization → QJL Residual → x̂`

### 11.1 旋转：Signed Hadamard（Rademacher ⊙ FWHT）

**目的（数学直觉）**：  
Hadamard 旋转是一种近似“随机正交变换”，能把能量在维度上更均匀地摊开；Rademacher（±1）符号相当于随机对角矩阵，增强“数据不可知”的随机性，从而降低量化误差对少数维度的集中影响。

**实现对应**：
- `fwht(a)`：Fast Walsh–Hadamard Transform（蝶形结构），最后除以 `sqrt(d)` 以保证近似正交。
- `_rademacher_signs(padded_dim, seed)`：生成 ±1 符号向量（带缓存）。
- `_signed_hadamard_rotate(tensor)`：pad 到 2 的幂后执行 `fwht(padded * signs)`。
- `_signed_hadamard_inverse(rotated)`：利用 Hadamard 自逆性质，复用 `fwht` 并再乘 signs，切回原始维度。

### 11.2 标准化（Standardize）：论文版 vs Vitriol 默认

在 `turbo_quantize(...)` 中有两条路线：

1) **Paper-exact（`use_blockwise=False`）**  
对每个向量做 z-score 标准化：  
`sigma = sqrt(mean(rotated^2))`，`normalized = clamp(rotated / sigma, grid_min, grid_max)`，再做 Lloyd-Max。

2) **Vitriol-enhanced（默认 `use_blockwise=True`）**  
代码注释称“per-block min-max scaling”，实际行为是：
- Step2 不做 z-score（`normalized = rotated`）
- Step3 进入 `_blockwise_quantize_dequantize(normalized, levels, block_size)`：在该函数内部做块内 min-max 缩放 + Lloyd-Max（见 11.3）

**研究提示**：这两条路线并非“实现细节差异”，而是改变了量化的统计假设（高斯 z-score vs 块内 min-max）。建议在论文中作为消融变量单列（见第 16 节）。

### 11.3 Lloyd-Max：Gaussian codebook + bucketize（以及 blockwise 的含义）

`_gaussian_lloyd_max_codebook(levels)` 会生成：
- `codebook`（重构值）
- `thresholds`（分桶阈值）

量化用 `torch.bucketize` 做高效分桶，然后用 `codebook[idx]` 得到量化值。

在 blockwise 路线 `_blockwise_quantize_dequantize` 中：
1. 按 `block_size` 分块
2. 对每块做 min-max 缩放到 `[0, levels-1]`
3. 对该“归一化后的值”做 Lloyd-Max bucketize
4. 反缩放回原尺度

这意味着：Lloyd-Max 并非直接作用于 N(0,1) 的输入，而是作用在“块内归一化空间”；严格来说这是启发式混合（同样适合论文对比）。

### 11.4 QJL Residual：1-bit sketch 回补残差

`_qjl_residual_sketch(residual)` 的核心是：
1. residual 做 signed-hadamard rotate
2. 取 sign（±1，0 替换为 +1）
3. 用 `scale = mean(|rotated|) * sqrt(pi/2)` 校准幅度
4. 构造 `sketch_rotated = sign * scale * strength`
5. inverse 旋转回去作为 correction

这相当于用极低比特保存“残差的方向+幅度统计”，常见于随机投影/1-bit 压缩思想。Vitriol 还在 `_TURBO_STATS` 中记录 residual 与 correction 的 L2/abs-mean，可作为论文中的诊断指标。

### 11.5 “存储压缩”与“计算近似”两条路径（必须区分）

Vitriol 同时存在两种语义不同的“压缩”：
1. **Runtime patch（近似计算）**：返回浮点张量，只改变数值，未必节省显存。  
2. **Packed storage（真实存储压缩）**：`kv/codec.py` 的 `PackedKVTensor`/bit-packing + `KVCacheStore` 路径才可能带来真实存储下降。

论文/工程报告里必须把这两条路径的收益与评测指标拆开。

---

## 12. AdaptiveKVCodec 深挖：熵驱动位宽分配 + 峰度触发旋转 + 向量化 QDQ

对应源码：`src/vitriol/kv/codec.py`（`AdaptiveKVCodec`）。

### 12.1 位宽分配（adaptive_kv_bits）：把注意力熵映射为 bits

实现要点：
- `w = softmax(QK^T / sqrt(d))`
- `H = -Σ w log w`，再除以 `log(seq_len)` 归一化到 `[0,1]`
- `importance = clamp(1 - H, 0, 1)`（越尖锐越重要）
- K bits：`min_bits + (max_bits-min_bits)*importance`
- V bits：用 `v_rms`（幅度）做重要性
- 最后整体缩放使平均 bits 接近 `target_avg_bits`，并用 `k_share` 控制 K/V 预算

这是一个可解释、可调参、可做消融的“注意力感知混合精度分配”方案。

### 12.2 峰度触发旋转：_maybe_rotate

`_maybe_rotate(x)` 用峰度（kurtosis）检测分布“重尾/尖峰”：
- 若 `kurt >= threshold`，做 `walsh_hadamard_rotate(x)`（近似高斯化）
- 量化后再对输出做一次同样的旋转以回到原空间（Hadamard 自逆）

### 12.3 向量化 QDQ：消灭 O(b·h) Python 循环

关键优化：`_vectorized_blockwise_qdq(x, per_batch_levels, block_size)`。

做法：
- 把 `[b,h,s,d]` reshape 成 `[N=b*h, s, d]`
- 把每个 entry 的 levels 扩展到每个 block
- 一次性计算所有 blocks 的 mins/maxs/scales，并 qdq

这让“每个 head 不同 levels”仍能高效运行，是系统实现层面的关键点。

---

## 13. ComputeSkip 与 Sparse-V：从上界推导到实现

### 13.1 ComputeSkip：贡献上界驱动的 block pruning

对应：`compute_skip_attention`（`kv/codec.py`）。

它对每个块计算：
- `attn_mass = Σ attention_weight`
- `v_norm = ‖V_block‖`
- `bound = attn_mass * v_norm`

再用 `keep = bound >= ε * total` 选择块。该 bound 可视作基于 Cauchy–Schwarz 的贡献上界，因此更“可论证”，适合论文叙事。

### 13.2 Sparse-V：按注意力阈值稀疏化

对应：`sparse_v_attention`（`patches/turboquant.py`）。  
把 attention 小于阈值的权重置零再归一化，属于简单 baseline，适合与 ComputeSkip 对照。

---

## 14. TurboQuantum 深挖：注意力熵驱动的自适应精度 + tunneling + entanglement residual

对应：`src/vitriol/kv/turboquantum.py`。

### 14.1 entropy → bits：核心可发表点应落在“注意力感知精度分配”

`compute_attention_entropy(query, key)` 明确实现：
`QK^T → softmax → entropy → /log(seq_len)` 并输出 collapsed/superposition heads 占比。  
`quantum_bit_allocator` 再把 entropy 与 `v_rms` 结合，生成 per-(batch,head) 的 `k_bits/v_bits`。

建议论文写法：把“量子类比”降级为直观解释，把贡献表达为 **attention-aware adaptive precision allocation** 更稳。

### 14.2 关键工程点：向量化量化、tunneling 保护、残差回补

`turboquantum_compress` 的流水线（可直接写进论文方法部分）：
1) per-head bits 分配  
2) K/V 分别做 signed-hadamard rotate  
3) z-score 标准化  
4) 向量化 quantize-dequantize（flatten b*h）  
5) inverse rotate  
6) tunneling：保护 top-k attention mass token（避免关键 token 精度损失）  
7) entanglement residual：对 residual 做 sketch 回补误差  

并直接输出 mse/cosine/effective_bpv 等报告字段，利于评测与复现。

---

## 15. KVStore 路径（真正“存储压缩”闭环的桥）：CacheHookPatcher + UniversalAttentionPatcher

对应：`src/vitriol/patches/cache_hooks.py`。

### 15.1 CacheHookPatcher：劫持 cache.update，把 KV 写入 backend

当 `_vitriol_kv_store_mode=True` 时，`update_wrapped` 会：
1. `backend.write_kv(handle, layer_idx, key_states, value_states, info)`
2. 维护 `_vitriol_seq_lens[layer_idx]`
3. decode（q_len==1）且非 passthrough 时，直接返回 contiguous 的 K/V，避免原 cache 存储路径

还会 patch `get_seq_length/get_mask_sizes`，保证 attention mask 与 past length 计算正确。

### 15.2 UniversalAttentionPatcher：在 attention forward 中从 backend 读 attention 输出

它 patch `transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS.get_interface`，在 decode 且 kv_store_mode 时尝试：
`backend.read_attention(...)` → 直接返回 attention output，失败再 fallback 到原实现。

这条路径是把“KV 压缩存储后端”真正接入推理的关键。

---

## 16. 评测与论文叙事：从逐层策略解释到端到端 PPL

### 16.1 runner 的逐层策略表：可解释性工件

`bench/runner.py::_collect_policy_insights` 能生成逐层表（layer_type + turbo_k/v + sparse_v + compute_skip）与汇总 counts。  
论文上可以把它作为“策略解释层”，与 `kv-analyze` 的逐层误差表对齐，定位 worst layers。

### 16.2 PPL evaluator：方向正确，但目前存在一致性/健壮性缺口

`bench/ppl_evaluator.py` 的目标是用 PPL/token match/KL drift 替代 proxy metric，非常“可发表”。  
但当前实现可见：
- `_load_model()` 里 `trust_remote_code=True` 仍硬编码（破坏安全开关一致性）
- `report()` 内部 `self.ppl_degeneration` 拼写疑似错误（应为 `ppl_degradation`），会影响报告输出可信度

建议：把这两点列为论文中的“系统实现修复点”或工程 PR 的 P0。

### 16.3 建议的论文级评测协议（最小但严谨）

1. 分离 prefill 与 decode（KV 策略主要影响 decode）  
2. 固化 `quantized_kv_start`，并报告该阈值（短上下文不量化，长上下文量化）  
3. 同时输出：速度、显存（peak 与 KV-only estimate）、PPL、token match、KL drift  
4. 逐层对齐：`kv-plan`（策略）× `kv-analyze`（误差）× `ppl_evaluator`（影响）  
5. 工件化：每次运行输出 md/json + 环境版本 + 模型指纹/arch hash

---

## 17. NAS evaluator 的“零成本 proxy”到底在算什么？（以及如何写成论文）

`nas/evaluator.py` 的 in-memory evaluation 路径体现了一个系统化设计：在 **一次 forward + 一次 backward** 中计算多种 proxy：
- Forward-only：NWOT、RankMe、表达性（dirichlet_energy × vocab_entropy）、attention diversity  
- Gradient-based：grad_norm、fisher、snip  
- 另跑 synflow（需要特定图处理）

可发表叙事建议：
> “我们提供一套面向 LLM 的 proxy 指标组合，并实证其与真实评测（PPL/throughput/long-context）之间的相关性；Vitriol 将其实现为可复用评测器并接入 NAS/推荐系统。”

同时要明确威胁：proxy 的跨家族可比性、tokenizer/vocab 依赖、以及 meta instantiation 与真实 runtime 的差异。

---

## 18. 工程化与安全闭环 2.0：从‘有开关’到‘全链路可证’

### 18.1 建立 HF 加载门面（facade），作为唯一事实来源

新增 `vitriol/utils/hf_loading.py`（或类似命名），统一封装：
- `load_config(model_id, security)`
- `load_tokenizer(model_id, security)`
- `load_model(model_id, security, dtype, device, ...)`

强制所有 Auto* 加载调用都走该门面，以统一 `trust_remote_code/allow_network/local_files_only` 行为。

### 18.2 单元测试闭环：用 monkeypatch 证明参数传播

最小测试策略：
1) monkeypatch `AutoConfig.from_pretrained` 记录 `trust_remote_code` 参数  
2) 执行 `vitriol --no-trust-remote-code analyze ... / evolve ... / bench kv-plan ...`  
3) 断言全链路传入 `False`

### 18.3 API NAS 真实性闭环：从占位 sleep → 复用 ConstraintOptimizer

把 API 的 `process_nas_job` 从模拟改为真实执行：
- 调用 `ConstraintOptimizer.optimize(...)`
- 写 artifacts（best_gene/best_score/best_metrics）
- 提供 progress（每 N 次迭代更新一次）

这样 WebUI/API 才能与 CLI 一致，形成“可验证服务化”叙事。

---

## 19. “更深”的一句话总括（对齐预期）

> **Vitriol 的关键贡献在于把结构驱动的低成本实验（生成/可视化/模拟/搜索）与策略驱动的 KV 压缩评测（runtime patch + store backend + presets + report）统一为一套可复现、可解释、可扩展的系统；其中向量化 QDQ、逐层策略工件与 cache/backend 接口化是其最硬的系统实现亮点。**

---

## 20. 深度优化路线图（可执行）：把“论文级系统”做成“可验证闭环”

你说“还需要进一步深度优化和分析”，我这里给出一个**不依赖主观判断、可以直接按 issue/PR 推进**的路线图：每一项都包含「目标→改动点→验收标准」。

### 20.1 P0：安全一致性闭环（trust_remote_code 彻底可控）

**目标**：任何入口（CLI / bench / webui / api / nas）都只从一个“事实来源”读取安全策略，并保证在 `--no-trust-remote-code` 下**不存在**内部写死 True 的加载点。

**改动点（建议）**
1. 新增统一门面：`src/vitriol/utils/hf_loading.py`
   - `load_config(model_id, *, trust_remote_code, local_files_only, allow_network)`
   - `load_tokenizer(...)`
   - `load_causallm(...)`
2. 替换所有 `AutoConfig/AutoTokenizer/AutoModel*from_pretrained(... trust_remote_code=True)` 的硬编码点：
   - `src/vitriol/bench/ppl_evaluator.py`（当前硬编码 True）
   - `src/vitriol/cli/commands/evolve.py`
   - `src/vitriol/webui/app.py`
   - `src/vitriol/nas/evaluator.py`
   - 以及其它 grep 命中点

**验收标准**
- `vitriol --no-trust-remote-code analyze gpt2` / `bench kv-plan gpt2` / `webui`（本地模型路径）能工作或明确失败，并且日志中能证明 trust_remote_code=False 被传递。
- 新增测试（不联网）：monkeypatch `AutoConfig.from_pretrained`，断言所有路径的参数一致。

### 20.2 P0：PPL evaluator 的健壮性与一致性

**目标**：PPL evaluator 作为“论文级指标”的核心组件，必须满足：可运行、可复现、输出无 bug，且遵守安全策略。

**已发现问题（来自源码）**
1. `_load_model()` 内 `trust_remote_code=True` 硬编码（破坏 20.1）
2. `report()` 内 `self.ppl_degeneration` 拼写疑似错误（字段名应为 `ppl_degradation`），会导致报告逻辑异常

**改动点（建议）**
- 修复 report 字段名与星级逻辑；为 `report()` 增加最小单测（构造 PPLResult 实例）。
- 将模型加载改为走 20.1 的门面；并将 trust_remote_code 暴露为 `PPLConfig` 字段或从 CLI ctx 传入。

**验收标准**
- `vitriol bench ppl ...`（如果有 CLI 入口）或直接跑 evaluator 的最小脚本能稳定产出 md/json。
- 输出报告能正确渲染 stars/表格，且结果字段齐全。

### 20.3 P0：API NAS 真实性闭环（从占位 sleep → 真正执行）

**目标**：`src/vitriol/api/server.py::process_nas_job` 当前为模拟逻辑，应改为复用 CLI/NAS 的真实实现，形成可验证产物。

**改动点（建议）**
- 在 API job 中调用 `ConstraintOptimizer.optimize(...)` 或 NASController 的统一入口（取决于现有 CLI 结构）。
- 产物落盘（`artifacts_path/result.json`），并在 job result 中返回：
  - `best_gene`（gene dict）
  - `best_metrics`
  - `best_score`
  - `artifacts_path`
- progress：每 N 次迭代更新一次（需要为 optimizer 增加 callback 或拆出循环）

**验收标准**
- 调 `/nas/search` 的结果不再是固定的 `{layers: 24, hidden_size: 1024}`。
- 可复现实验：相同 seed / search space 配置下，结果可重复或至少可解释（随机搜索需记录随机种子）。

### 20.4 P1：KVStore 真压缩闭环（“真的省显存/带宽”）

**目标**：把 KV 压缩从“浮点近似（runtime patch）”推进到“真实存储压缩（bit-pack/packed KV）”，并在 benchmark 中能观测到显存/带宽收益。

**改动点（建议）**
- 明确区分两条 benchmark：
  1) runtime patch path：测吞吐与质量（不承诺显存线性下降）
  2) kv store path：测显存/带宽（KV-only bytes、peak memory、cache hit/miss）
- 为 KVStore backend 定义统一统计接口（例如 `backend.stats()`），并在 `kv-report` 汇总输出。

**验收标准**
- report 中出现 “KV-only bytes（估算/实测）” 与 “peak device MB（实测）” 的一致对齐。
- 同模型同 prompt 下，store path 的显存下降与“理论 bpv”一致（允许常数项差异，但应解释）。

### 20.5 P2：仓库发布卫生与可维护性（面向开源与论文 artifact）

**目标**：降低 clone/CI/审计成本；让论文 artifact 与代码仓库分离（或可一键生成）。

**建议**
- `output/`、`__pycache__/`、大模型权重等不应进入主仓库历史（除非明确为 dataset/artifact repo）。
- 建议建立 `artifacts/` 标准目录：`bench/<date>/<model>/<preset>/report.{md,json}` + `env.txt` + `fingerprint.json`。

---

## 21. 算法严格化：统一符号系统 + 误差分解（用于论文“方法”章节）

本节目的：把 TurboQuant / AdaptiveKVCodec / TurboQuantum 三套路径，统一到一个可对比的数学框架里，避免“看似相似但不可比”的叙述。

### 21.1 统一符号

- Query/Key/Value：$Q, K, V \\,\\in \\mathbb{R}^{B\\times H\\times L\\times D}$
- 注意力权重：$W = \\mathrm{softmax}(QK^T/\\sqrt{D})$
- 量化算子（一般形式）：$\\hat{K} = \\mathcal{Q}_\\theta(K; Q, V)$，$\\hat{V} = \\mathcal{Q}_\\theta(V; Q, K)$  
  其中 $\\theta$ 表示策略参数（bits、block size、是否旋转、是否 residual sketch、tunneling 等）

### 21.2 TurboQuant 的误差项结构

把 TurboQuant 写成组合算子：
- 旋转：$R(\\cdot)$（signed Hadamard）
- 标准化：$S(\\cdot)$（per-vector z-score 或 blockwise min-max 对应的“尺度归一化”）
- 标量量化：$\\mathrm{LM}_m(\\cdot)$（m=levels 的 Lloyd-Max）
- 残差回补：$\\Delta(\\cdot)$（QJL sketch）

则可写为：
$$\\hat{x} = R^{-1}\\big(S^{-1}( \\mathrm{LM}_m(S(R(x))) + \\Delta )\\big)$$

误差可分解为：
1) 旋转/逆旋转的数值误差（通常小）  
2) 标量量化误差（主项）  
3) 残差 sketch 的近似误差（希望抵消 2) 的部分）  

在 Vitriol 实现里，残差统计（`_TURBO_STATS`）可用来量化 2) 与 3) 的相对规模。

### 21.3 AdaptiveKVCodec：bits 分配作为“控制变量”

AdaptiveKVCodec 的核心不是具体量化器，而是 **bit allocation**：
- $b_K(b,h)$ 由注意力熵 $H(W_{b,h})$ 决定
- $b_V(b,h)$ 由 $\\mathrm{RMS}(V_{b,h})$ 决定
- 再整体缩放命中 `target_avg_bits`

因此理论上更像一个“混合精度控制器”，量化器本身是 blockwise qdq（可替换）。

### 21.4 TurboQuantum：在统一框架下的增强项

TurboQuantum 可以被解释为：
1) 更强的 bits 分配（entropy-aware）  
2) 更强的保护机制（tunneling：关键 token 保留精度）  
3) 更强的残差回补（entanglement residual）  

建议论文写法：把“量子类比”作为解释，不作为核心定义；核心定义用上述三点即可。

---

## 22. 复杂度/带宽模型：用简单公式解释“为什么某些策略能提速”

### 22.1 KV 的主导成本：显存带宽而非 FLOPs

长上下文 decode 阶段，attention 读取 KV 的带宽成本往往主导。  
因此有效指标之一是：
$$\\text{KV bytes/value} \\times B \\times H \\times L \\times D$$

Vitriol 在 `kv/codec.py::kv_bytes_per_value` 中把 turbo2/3/4、adaptive_bits 等映射到 bytes/value，这可以直接用来做理论 KV-only bytes 估算。

### 22.2 为什么 runtime patch 不一定省显存

如果量化结果仍以浮点张量存储（runtime patch 路线），则：
- 带宽可能下降（如果 compute_skip/sparse-v 跳过了读取/计算）
- 但显存占用不一定按 bpv 下降（因为没有 bit-pack）

因此 benchmark 必须分别报告：
1) peak device memory（实测）
2) KV-only estimate（理论/编码后真实 nbytes）
3) non-KV gap（帮助解释常数项）

### 22.3 compute-skip 的“上界驱动”优势

ComputeSkip 用 $\\text{attn_mass}\\times\\|V\\|$ 的上界筛块：  
它更接近“证明式 pruning”，在长上下文下往往比固定阈值 sparse-v 更稳定（但依赖 epsilon 的选择）。

---

## 23. KVStore 真压缩：建议的“最小闭环评测方案”（比现在更深、更可证）

### 23.1 目标：证明“bit-pack 后 KV 真的变小”

建议新增/固化一个报告字段集合：
- `kv_store_encoded_bytes`：编码后 KV 的真实 bytes（例如 packed tensor 的 `storage_nbytes()` 之和）
- `kv_store_decode_bytes_read`：decode 阶段累计读取字节（需要 backend 计数器）
- `peak_device_mb`：实测峰值显存
- `toks_per_sec`：吞吐
- `ppl/token_match/kl`：质量

### 23.2 最小对比矩阵（论文表格可直接用）

对每个模型（至少 2 个家族）与两种上下文长度（短/长）：
1) safe（no quant）
2) turbo3 runtime patch
3) adaptive_bits runtime patch
4) turboquantum codec（如果走 store）
5) compute_skip（单独/组合）

### 23.3 统计显著性（最低要求）

至少对每个设置运行 N=5 次，报告：
- mean ± std
- speedup 的置信区间或 bootstrap

否则论文容易被质疑“偶然波动/缓存效应”。

---

## 24. 论文叙事升级版：把 Vitriol 写成“系统论文”而不是“工具介绍”

如果你的目标是学术发表，我建议把 narrative 收敛为四个“硬贡献点”，每个都有可验证实验支撑：

1) **统一框架**：结构-权重解耦 + 统一 CLI/WebUI/API + 工件化输出  
2) **KV 策略系统**：presets + 逐层策略解释 + runner/report  
3) **两条压缩路径**：runtime patch（近似计算）与 KVStore（真实存储）分离，并各自评测  
4) **可复现性与治理**：安全策略可控（trust_remote_code）、实验工件与指纹体系

这四点可以自然对应论文结构：
- Method：11–15 的算法+系统对照
- Implementation：向量化 QDQ、cache/backend patch、runner/report
- Evaluation：16 + 23 的协议与矩阵
- Discussion：20 的闭环路线图与威胁

---

## 25. 代码级优化落地进展（已实现，便于你们写“系统实现”章节）

本节记录“从报告建议 → 实际代码改动”的落地成果，便于你们直接引用到论文/技术报告的 Implementation 部分。

### 25.1 P0：trust_remote_code 安全治理（已落地）

- 新增测试 `tests/test_trust_remote_code_policy.py`：禁止在源码中出现硬编码 `trust_remote_code=True`（以避免绕过全局安全开关）。  
- 将多个模块中的硬编码点改为**从 ctx/config 透传**（包括 evolve / vocab_viz / analyzer / exporter / config_processor / generator README 模板 / webui / nas evaluator 等）。
- 同时对 `vitriol.bench` 与 `vitriol.nas` 做了 **lazy import**（避免 import 包就强制拉起 torch/transformers），提升模块化与可测试性。

### 25.2 P0：PPL evaluator 稳定性与安全一致性（已落地）

- 修复 `PPLResult.report()` 的字段拼写 bug（`ppl_degeneration` → `ppl_degradation`），并补充覆盖该分支的单元测试。  
- 为 `PPLConfig` 增加 `trust_remote_code: bool` 字段，并确保 `_load_model()` 对 tokenizer/model 的加载统一透传该参数。  
- 新增测试 `tests/test_ppl_evaluator.py`，在轻量环境下通过 stub 验证“报告渲染稳定 + 参数透传正确”。

### 25.3 P0：API NAS 从“占位”升级为“真实执行”（已落地）

- 将 `vitriol.api.server.process_nas_job` 从 sleep+假结果替换为：  
  - 基于 `LLMSearchSpace.sample()` 的可运行 search loop  
  - 逐迭代更新 job progress（支持 /jobs 轮询）  
  - 输出 `best_gene/best_metrics/best_score`，并可写入 `artifacts_dir/nas-result-<job_id>.json`  
- 为此补齐了 `vitriol.nas` 的 lazy import 与 `targeted_nas` 的去耦（移除未使用的 searcher/evaluator import），避免 API 仅因“导入副作用”而要求 transformers/torch。
- 新增测试 `tests/test_api_nas_job.py`：验证 NAS job 非占位、progress=100、产物字段齐全。

---

## 26. P1：统一 HuggingFace 加载门面（HF Loading Facade）（已落地）

### 26.1 目标与动机

Vitriol 的现实问题是：代码库存在大量 `AutoConfig/AutoTokenizer/AutoModel...from_pretrained` 的分散调用点。即便你已经提供了全局 `--trust-remote-code/--no-trust-remote-code`，也容易出现：

- 参数传播不一致（某些路径忘记传、或只传部分参数）
- 未来新增功能时回归（新代码再次写死或遗漏）
- 安全审计困难（无法快速确认“所有 HF 加载”是否遵守统一策略）

因此需要“唯一事实来源”：**统一加载门面（facade）**。

### 26.2 实现概览

新增模块：
- `src/vitriol/utils/hf_loading.py`

提供统一函数：
- `hf_kwargs(security, extra=...)`：把 `SecurityOptions` 转为 `from_pretrained` kwargs
- `load_config(...)`
- `load_tokenizer(...)`
- `load_causallm(...)`
- `load_model_from_config(...)`
- `load_causallm_from_config(...)`

关键规则（强约束）：
- `allow_network=False` ⇒ 强制 `local_files_only=True`（防止意外联网）
- `trust_remote_code` 只允许来自上层显式输入（CLI/config/API），不允许在调用点写死
- 全部采用**函数内延迟导入 transformers**，避免“import vitriol 就强制安装 transformers/torch”

### 26.3 已完成的接入点（示例）

目前已将以下模块迁移到门面（让它们成为“正确示范”）：
- `core/config_processor.py`：配置加载统一走 `hf_loading.load_config`（并统一解析 trust_remote_code/allow_network/local_files_only）
- `bench/ppl_evaluator.py`：model/tokenizer 加载统一走 `hf_loading.load_tokenizer/load_causallm`
- `evolution/tree_builder.py`：拉取 config 统一走 `hf_loading.load_config`
- `core/analyzer.py`：config + meta 模型构造统一走门面
- `core/exporter.py`：meta-config/config.json 的读取统一走门面

### 26.4 测试与可验证性

新增单元测试：
- `tests/test_hf_loading.py`：在不安装 transformers 的情况下，用 stub 验证：
  - trust_remote_code 透传正确
  - allow_network=False 会强制 local_files_only=True
  - from_config 路径同样遵守 trust_remote_code

并且与此前的“安全守门员测试”一起构成闭环：
- `tests/test_trust_remote_code_policy.py`：确保源码中不存在 `trust_remote_code=True` 的硬编码

这意味着：后续只要有人新增/回归硬编码或绕过门面，CI 会立刻红灯。

---

## 27. P1.5：把“门面”升级为“强约束系统不变量”（Enforced Invariants）（已落地）

如果只引入 `hf_loading`，但不强制团队使用，长期仍会退化为“部分路径走门面、部分路径走直调用”，最终又回到不可审计状态。  
因此本阶段把它升级为**系统不变量**（invariants），用测试把架构约束“钉死”。

### 27.1 不变量 1：禁止绕过门面直接调用 Auto*.from_pretrained/from_config

新增测试：
- `tests/test_hf_facade_enforced.py`

实现方式：
- 用 AST 解析源码（避免误伤 docstring/模板字符串）
- 扫描所有 `AutoConfig/AutoTokenizer/AutoModel/AutoModelForCausalLM` 的 `from_pretrained/from_config` 调用
- 除 `src/vitriol/utils/hf_loading.py` 外，其余文件出现即判定失败

**效果**：把 HF 加载入口收敛到单点，使得：
- 安全策略传播（trust_remote_code / allow_network / local_files_only）不会再因“忘记传参”而破功
- 审计与代码 review 成本显著下降（只需盯门面）
- 新人新增功能时自然走正确路径（否则 CI 红灯）

### 27.2 不变量 2：禁止硬编码 trust_remote_code=True（继续保留）

保留并继续使用：
- `tests/test_trust_remote_code_policy.py`

它与 27.1 的关系是：
- 27.1 防“入口分散”
- 27.2 防“入口里写死 True”

两者合起来才能形成真正的安全闭环。

### 27.3 迁移范围（覆盖率提升：从示范路径 → 全链路）

为满足 27.1，本阶段已把以下模块的 HF 加载迁移到门面（不再出现 Auto*.from_* 直调用）：
- `arch_viz/parser.py`
- `bench/runner.py`（大量加载点）
- `cli/commands/{evolve,infer,vocab_viz}.py`
- `core/{generator,validator}.py`
- `nas/evaluator.py`
- `strategies/learned.py`
- `tools/{minimax_pipeline,model_demo}.py`
- `vocab_viz/core.py`
- `webui/app.py`

### 27.4 门面能力补齐：新增 load_model（通用 AutoModel.from_pretrained）

新增接口：
- `hf_loading.load_model(...)`

并补单测：
- `tests/test_hf_loading.py` 扩展验证通用模型加载同样遵守安全参数（尤其是 allow_network=False ⇒ local_files_only=True）。

---

## 28. P2：安全语义的端到端一致性（Offline / Network / Remote Code）（已落地）

P1.5 解决了“入口收敛 + 禁止绕过”，但仍存在一个更深层的问题：  
**语义一致性**——即 *同一个安全意图*（例如“离线”）是否能跨 CLI / API / 库内部 codepath 一致生效，并且不可被局部代码误用绕过。

### 28.1 威胁模型（Threat Model）

在真实工程里，“不联网”失败通常不是因为开发者显式写了 `requests.get()`，而是：
- transformers/huggingface_hub 在某些分支会尝试请求（即便你传了 local_files_only）
- 其他调用点误传 `allow_network=True`，导致门面被“软绕过”
- 进程内不同组件对“离线”的定义不一致（CLI 以为离线，但 API/worker 仍允许联网）

因此 P2 要做的不是“再加一个 flag”，而是把“离线语义”提升为**不可被绕过的全局不变量**。

### 28.2 不变量定义（Invariants）

**Invariant A：allow_network=False ⇒ local_files_only=True（强制）**  
这个规则在门面层直接执行，避免调用方遗漏。

**Invariant B：OFFLINE 模式不可被绕过**  
一旦进程被标记为离线（环境变量），任何后续加载调用即便误传 `allow_network=True`，
也不能把 `local_files_only` 再变回 False。

### 28.3 实现要点（Implementation）

在 `vitriol.utils.hf_loading.hf_kwargs()` 中：
- `allow_network=False` 时：
  - 强制 `local_files_only=True`
  - 设置环境变量 `HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1`
- 若检测到环境变量已处于 OFFLINE：
  - 强制覆盖调用方输入（把 allow_network 视为 False）
  - 使离线语义变成“系统级锁定状态”

### 28.4 CLI / API 贯通（End-to-End Propagation）

**CLI**：
- 新增 `--offline`（等价于 `--no-allow-network --local-files-only`）
- 新增 `--allow-network/--no-allow-network` 与 `--local-files-only`
- 通过 `ctx.obj` 贯穿到命令实现，避免“命令内部硬编码 allow_network=True”

**API**：
- `GenerateRequest` / `NASRequest` 增加 `allow_network` / `local_files_only`
- `process_generation_job` 将其透传到 `build_generation_config(overrides=...)`

### 28.5 证明方式：端到端回归测试（Proof by Tests）

为避免 P2 变成“设计文档”，这里用测试把语义钉死：
- `tests/test_offline_semantics.py`：验证
  - allow_network=False 会设置 OFFLINE 环境变量
  - OFFLINE 环境变量存在时，调用方无法再绕回 allow_network=True
- `tests/test_cli_offline_propagation.py`：验证 `--offline` 能实际影响 HF 加载安全参数
- `tests/test_api_offline_propagation.py`：验证 API 请求的离线字段会进入生成配置 overrides
- 结合 `tests/test_hf_facade_enforced.py`：保证所有 HF 加载只能走门面，从而确保上述不变量覆盖全仓库

### 28.6 价值：把“安全意图”变成“可证明的系统属性”

P2 的最终产出不是“多了两个 flag”，而是：
- 任何新功能/新命令/新 worker 只要想加载模型，就会被迫走门面
- 只要进程处于 OFFLINE，就没人能在代码里“悄悄开网”
- 安全语义由测试证明，而不是依赖开发者自觉

---

## 29. P3：安全配置的单一事实来源（SecurityContext SSoT + Provenance）（已落地）

P2 做到了“离线不可绕过”，但在大型系统里仍会出现一个更深层的工程问题：  
**同一个安全字段在多处被解析/修正**（build_generation_config 一套、hf_loading 一套、CLI 又一套），久而久之会产生“语义漂移”：
- 某处修复了 offline，但另一处忘记跟进
- 某处新增字段（比如 local_files_only）但只有部分链路透传
- 发生线上问题时无法回答：“这个值最后为什么是 True/False？来自哪里？”

### 29.1 目标：Single Source of Truth + 可审计（Provenance）

P3 的目标是把安全语义解析集中到一个可复用组件：
- **SecurityContext**：最终生效的安全配置（trust_remote_code / allow_network / local_files_only）
- **provenance**：每个字段最终值来自哪里（base / explicit / env_offline）
- **apply_to_environ**：把“不可绕过”的全局语义（OFFLINE）落到进程环境变量

### 29.2 解析优先级（Precedence）

当前落地的优先级（从高到低）：
1) **env OFFLINE**（HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE）——最高优先级、不可绕过  
2) **explicit**（调用方显式覆写）  
3) **base**（默认值/配置文件/上层预设）

并在 resolver 内统一执行不变量：
- allow_network=False ⇒ local_files_only=True

### 29.3 代码落地（Implementation）

新增模块：
- `src/vitriol/security/context.py`

核心接口：
- `resolve_security_context(base=..., explicit=...) -> SecurityContext`

并将“门面层”完全委托给它：
- `vitriol.utils.hf_loading.hf_kwargs()` 不再自行处理 OFFLINE/allow_network/local_files_only，而是调用 `resolve_security_context` 获取最终裁决。

同时让“配置层”与其对齐：
- `build_generation_config` 在组装 `SecurityOptions` 后，会走 `resolve_security_context`，从而让 **env OFFLINE** 对配置层也生效（避免出现“配置认为可联网，但门面禁止联网”的语义分裂）。

### 29.4 证明：单测覆盖（Why this is deeper）

新增测试：
- `tests/test_security_context.py`：验证优先级与 provenance，并验证 build_generation_config 遵守 env OFFLINE

这样 P3 的效果是：
- 以后任何人想改安全语义，只需要改 SecurityContext resolver 一处
- 任何入口（CLI/API/config/hf_loading）最终行为一致
- 发生行为争议时可以通过 provenance 精确解释“为什么会这样”

---

## 30. P4：运行时审计（SecurityContext 的结构化传递 + Job/Report 可追溯）（已落地）

P3 解决了“语义解析一致”，但还缺少一个**运行时视角**：  
当一个 job 已经跑完（或线上事故发生），你需要能回答：
> “这次运行最终到底用了什么安全配置？每个字段来自哪里？是否被 OFFLINE 强制覆盖？”

因此 P4 的目标是：把 SecurityContext（含 provenance）作为**一等公民**，贯穿到配置产物与 job 结果里。

### 30.1 配置产物：GenerationConfig 暴露 security_context（含 provenance）

在 `build_generation_config()` 中：
- 在返回的 `GenerationConfig` 上新增 `security_context` 字段，结构为：
  - trust_remote_code / allow_network / local_files_only
  - provenance（每字段来源：base/explicit/env_offline/inferred_offline）

这让后续任何组件都不需要“重新猜测”安全语义，直接读取即可。

### 30.2 API Job 结果：返回 security_context

在 `process_generation_job` 完成后：
- 将 `config.security_context` 写入 `job["result"]["security_context"]`

这样前端/调用方无需看日志就能拿到“最终安全配置 + 来源解释”，从而把安全信息变成可观察指标（observable）。

### 30.3 证明：回归测试

新增测试：
- `tests/test_generation_config_security_audit.py`：验证 build_generation_config 一定产出 security_context 且含 provenance
- `tests/test_api_security_context_in_result.py`：验证 API generation job 结果必含 security_context（含 provenance）

> 注：后续若要更进一步，可把 security_context 写入生成模型的 vitriol-manifest.json，实现“产物级可追溯”。由于核心 generator 依赖 torch/transformers，本仓库的轻量测试环境暂未对 manifest 写入做集成测试，但实现路径已明确且与现有结构兼容。

---

## 31. P5：产物级可追溯（Manifest Schema v2 + SecurityContext 持久化）（已落地）

P4 让“运行结果”可追溯，但真正强的工程属性是：**离开服务端、只拿到产物目录也能复盘安全语义**。  
这对离线交付、审计归档、事故复盘都非常关键。

### 31.1 目标

把 `SecurityContext`（含 provenance）写入生成产物的 `vitriol-manifest.json`，形成“产物自描述”：
- 你不需要依赖 API/日志
- 你只需要打开产物目录里的 manifest，就能知道：
  - trust_remote_code / allow_network / local_files_only 最终值
  - 每个字段来源（base/explicit/env_offline/inferred_offline）

### 31.2 设计：把 manifest 构造从重依赖代码中抽离出来

generator 模块本身依赖 torch/transformers，导致“纯 manifest 行为”难以在轻量 CI 中测试。  
因此新增轻依赖模块：
- `src/vitriol/core/manifest.py`
  - `build_manifest(...)`：纯 Python 构造 manifest dict（无 torch/transformers 依赖）

这样：
- generator 在运行时负责收集信息（hash/尺寸/可加载性等）
- manifest builder 负责拼装结构与 schema（可单测）

### 31.3 实现：schema_version 升级 + 写入 security_context

在 `core/generator.py::_write_manifest()` 中：
- `schema_version` 从 1 升级为 2
- 在 manifest 顶层新增 `security_context` 字段（直接写入 `GenerationConfig.security_context`）

### 31.4 可测试性改进：core 包 lazy import

为了在无 torch 环境下能 import `vitriol.core.manifest`：
- `src/vitriol/core/__init__.py` 改为 lazy import（类似 nas/bench 的做法）

### 31.5 证明：回归测试

新增测试：
- `tests/test_manifest_builder.py`：验证 build_manifest 必含 security_context + provenance，并使用 schema v2
