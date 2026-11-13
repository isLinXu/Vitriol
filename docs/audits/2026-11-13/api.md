## API、配置、安全与 NAS 审查

审查范围：`src/vitriol/api/`、`config/`、`security/`、`nas/`，共 16 个 Python 文件，约 4700 行。逐文件阅读并对现有测试的契约反向核对，修复集中于 7 组可执行回归；没有联网下载模型或运行完整规模 NAS。

### 模块地图与调用关系

| 模块 | 公共接口与职责 | 关键边界 |
|---|---|---|
| api/server.py | GenerateRequest/Response、NASRequest、ModelInfo、SystemStatus、BatchGenerateRequest/JobStatus；root、health_check、get_status、generate_weights、get_job_status、list_jobs、start_nas_search、list_supported_models、list_model_families、list_model_adapters、list_strategies、stream_logs、start_batch_generation、get_batch_status、process_generation_job、process_batch_job、process_nas_job、main | HTTP 参数到后台作业；verify_api_key、RateLimiter；LogStreamQueue 处理日志缓存 |
| config/manager.py | SecurityOptions、GenerationConfig、build_generation_config、generation_config_schema、validate_generation_dict | YAML / 环境变量 / 显式参数的合并，数值范围与安全字段解析 |
| config/settings.py | ConfigEnvironment、GenerationDefaults、NASConfig、SystemConfig、SecurityConfig、VitriolConfig、ConfigManager、get_config、init_config | App 级配置与 generator 配置独立；ConfigManager 提供 load/get/set/save/watch/unwatch/validate 与环境判断 |
| config/constants.py | 默认参数、shrink 值、RoPE 默认、资源扩展名、回退配置链与内存阈值 | import 时读取环境变量，非法整数字符串会在 import 时抛错 |
| security/context.py | SecurityContext、resolve_security_context | base < explicit < OFFLINE 环境；生成 provenance；离线结果写入进程环境 |
| nas/search_space.py | ArchitectureGene、SearchSpace、LLMSearchSpace | 架构编码，from/to dict/config，sample/sample_random、mutate、validate_gene |
| nas/targeted_nas.py | ObjectiveType、ConstraintType、Constraint、OptimizationTarget、ParetoSolution、MetricsCalculator、ConstraintOptimizer、MultiObjectiveOptimizer、DirectedMutator、create_constraint_optimizer、quick_search | 参数/显存代理估计、约束筛选、目标评分、Pareto 支配关系 |
| nas/searcher.py | Searcher、RandomSearcher、EvolutionarySearcher | 搜索循环与历史、回调、种群选择及交叉 |
| nas/controller.py | NASController | run、算法选择、dataset 注入、JSON checkpoint 恢复、最终结果 |
| nas/evaluator.py | ZeroCostProxy、ParamCountProxy、GradNormProxy、SynflowProxy、FisherProxy、SNIPProxy、JacobianCovarianceProxy、RankMeProxy、VitriolExpressivityProxy、VitriolAttentionDiversityProxy、HybridEvaluator | 小模型实例化、load_dataset、evaluate 的前向/反向代理；并非正式训练后的模型精度 |
| nas/rl_agent.py | Experience、ArchitectureEncoder、PolicyNetwork、ValueNetwork、RLSearcher | 编码状态、动作选择、经验缓存与策略优化、save/load；实验性 API |
| 各 __init__.py | 公共导出、NAS 延迟导入 | NAS 轻量搜索空间可在不导入 torch 时使用 |

调用链：HTTP generate → request.model_dump → 后台协程 → `_run_generation`（worker + serialization lock）→ build_generation_config → resolve_security_context → MinimalWeightGenerator。NAS 约束优化器：sample → check_constraints → evaluator.evaluate → evaluate_objectives → 更新最优解；多目标版本把收益转成最小化目标，再按非支配关系保留 Pareto 前沿。

### 已修复的问题与原因

1. **嵌套安全覆盖失效（P1）与配置文档校正**。提取嵌套 `overrides['security']` 交给统一 resolver，使其不再被最终 `config.security = security` 丢弃。同一次调用同时提供平铺安全字段和嵌套安全字段时，平铺字段优先，以保留既有 CLI 调用方式；OFFLINE 环境仍是不可绕过的最高优先级。此处不能简单把 security 原样传给 dataclass：安全值还需要互相约束和来源审计，直接覆盖会丢失 offline 推导和 provenance。配置来源的既有契约是 **Explicit/CLI > YAML > Environment > defaults**，README.md:855、README_CN.md:855 和 test_generation_config_resolution.py 均确认；保持原 merge 顺序，只修正 manager.py 中与该契约冲突的过时头注释。**环境变量不应改为覆盖显式 YAML 配置，本项不把原优先级实现计为缺陷。**

2. **生成 API 缺省值（P1）**。Pydantic model_dump 包含 `allow_network=None` 与 `output_dir=None`，旧代码 `bool(request.get('allow_network', True))` 将缺省网络允许意外转换为 False，而且 `dict.get(..., fallback)` 对存在的 None 不会采用默认输出目录。修复只传入非 None 网络策略覆盖，trust_remote_code 继续默认显式 False，并用 `request.get('output_dir') or ...`，与请求模型所承诺的“省略则用配置默认值”一致。批量 trust_remote_code 原先 `bool('false')` 会开启信任，改为把值交给已有统一布尔解析，避免两条入口的解释不一致。

3. **HTTP 限流响应（P1）**。自定义 middleware 位于异常处理中间件之外，直接抛 HTTPException 不会由 endpoint handler 转成正常 429，而可能成为 500。修复在 middleware 直接返回 JSONResponse，保持原来的 detail 与 Retry-After header。回归从真正 TestClient 请求第二次 `/health` 来验证最终 HTTP 状态，而非只测 limiter.check 能抛异常。

4. **后台生成阻塞事件循环（P1）**。async process_* 协程内同步运行生成器构造及 generate，导致生成期间健康、日志和状态查询无法及时响应。修复将配置解析、构造、生成整体移动到 asyncio.to_thread，并用专用 threading.Lock 保留 API 生成任务串行语义，防止多任务同时修改生成库状态。作业 dict 的更新仍在事件循环线程，因此没有把共享 jobs 的修改也放入线程。同步锁只在 worker 中等待，绝不能在事件循环里等待该锁；锁也意味着吞吐量仍为串行，优化目标是可观测性和响应性，不是同时运行多个高内存模型。

5. **NAS 丢弃真实评估（P1）**。ConstraintOptimizer.optimize 和 MultiObjectiveOptimizer.optimize 都收 base_evaluator 参数，却使用 random.random 充当模型质量分数。因此用户提供的 evaluator 完全不调用，搜索结果可能与真实优劣相反。现在调用 evaluator.evaluate(gene)，从其 score 读取有限数值，写回结果 metrics.nas_score；NaN/Inf 显式拒绝，防止支配关系和 best_score 被污染。无 evaluator 的既有纯参数估计路径保留兼容并使用 0.0，而不制造随机质量分。约束失败的候选仍在评估前筛掉，避免昂贵且没有意义的模型执行。这里保留了最大化目标取负号供 Pareto 最小化比较的既有设计，没有改动算法的支配关系。

6. **API 作业状态账务（P1）**。删除只有入队、没有消费的 asyncio.Queue，状态接口直接从 jobs 的 status 统计 running 与 queued，避免同一个任务被两套独立结构记录造成累计虚假排队。batch 任务也进入统一状态统计，completed、failed、completed_with_errors 均不再计入活动和排队数量。保留历史查询行为，没有通过静默丢弃历史来掩盖统计错误。

7. **SSE 缓冲溢出后停流（P1）**。LogStreamQueue 为每条日志分配锁保护的单调 sequence，stream_logs 按 sequence 而不是 deque 长度获取增量。缓存仍为有界的 1000 条，满容量后新条目不再被误判为“没有变化”。回归用 maxlen=2 的实际 StreamingResponse body_iterator 先读取两条、溢出后再读取第三条，直接覆盖长度不变的失败条件。缓冲溢出可能丢失消费速度不足期间的旧日志，这是有界缓存的明确成本；修复保证新日志可继续流动，不声称无损持久订阅。

### 复杂度与实施权衡

配置合并的时间/空间复杂度为 O(K)，K 是配置键数，没有必要引入额外配置框架。普通约束搜索 N 个候选、C 条约束的检查成本为 O(NC)，外部评价 E 的总成本 O(NE)，所以只对可行候选调用 evaluator 很重要。当前增量 Pareto 方法最差 O(N²M)，M 是目标数，并没有完整 NSGA-II 的交叉、拥挤度或非支配排序世代；接口注释中的 inspired 不应解释为标准 NSGA-II 实现。API 生成保持单并发，因此模型内存峰值仍约为一个任务，代价是多个请求会等待锁；asyncio.to_thread 将同步 I/O 与耗时初始化从事件循环移开，但并不承诺 CPU Python 工作的并行加速。

### 回归验证设计

新增 `tests/test_api_config_nas_regressions.py`：配置文件/env/显式覆盖三级对照；nested security 与 flat precedence；真实 HTTP 429；与 Pydantic 真实 model_dump 相同的 None 输入；线程 Events 控制的 single/batch 阻塞复现；用两份确定 evaluator score 验证最优候选和 Pareto 解；无 evaluator 确认零质量分；NaN、正/负无穷显式拒绝。同步阻塞测试故意在生成器 constructor 等待 Event：旧实现会冻结事件循环直至超时，新实现下协程仍可释放 Event。这能捕获只把 generate 移到线程、却留下昂贵 constructor 的不完整修复。

修前源码快照位于 `/tmp/vitriol-api-baseline`，用于在依赖安装完成后执行同一组测试的修前失败证据，避免还原共享工作树影响其他模块修复。验证结果将在实际执行后追加。

### 仍需后续处理的限制（未宣称已修复）

- **API 历史生命周期**：已修复未消费队列及终态误计；历史任务结果仍保留在内存，进程重启会丢失，没有持久队列、取消协议或已定义的历史保留策略。当前修复没有假称完成生产级任务调度。
- **SSE 长连接**：已改用 sequence 修复缓存满后停流；仍是有界内存缓冲，消费者过慢时可能漏掉已淘汰的旧日志，没有持久回放协议。
- **NAS API 参数契约**：process_nas_job 实际仅用随机候选 + 最小参数量目标；algorithm/population_size/dataset 未驱动真实算法或数据评估。结果是结构估计，不是数据集表现或延迟实测。
- **NAS 资源估计**：MetricsCalculator 参数式假设 dense attention，忽略 GQA/MQA 的 KV 投影差异与 MoE 专家、MLA/Mamba 实际权重。calculate_flops 名称注释写 per token，却乘 seq_len；显存是参数量乘经验比例，不能当硬件约束的安全保证。MINIMIZE_LATENCY 没有 latency_ms 来源，得分恒零。不能凭粗糙估算确定真模型可部署。
- **架构实现和编码**：ArchitectureGene.to_config 固定 qwen2，部分 micro/advanced gene 标记可能没有改变实例化模型结构。hidden_size 可在构造时向下舍入，不保证仍属于搜索空间；n_heads=0 会除零。EvolutionarySearcher 小种群父本不足时 random.sample(parents,2) 会失败。
- **RL 实验能力**：列出 12 个 actions，但仅处理层数、hidden 与随机变异，其他动作没有实现；GAE 应保留轨迹顺序，当前却对 random.sample 后的经验算递推，且不使用 done；encoder 输出被 detach 存入缓存，更新无法回传 encoder；CPU tensor 默认设备在 CUDA encoder 下需验证；save 的 history 内有 ArchitectureGene，自定义对象会与 weights_only=True 恢复冲突。这些涉及算法语义及 checkpoint 迁移，没有用局部改动冒称 RL 正确。
- **安全与配置**：resolve_security_context 会把离线选择写入整个进程，之后网络请求被离线环境压制；这是已有测试明确要求的不可绕过行为，修复没有逆转它。ConfigManager 的 custom 字典加载分支用 hasattr(dict, key)，不能正常恢复普通 custom 键；validate 针对错误类型的输入可能抛 TypeError。API output_dir 是受信本地操作者指定的服务器路径，未提供租户隔离或目录沙箱，部署边界应与 localhost / API 鉴权的现有定位一致。

### 理解验证与适用范围

已可解释配置来源、None 缺省值、安全来源审计、ASGI 中间件异常边界、线程工作隔离与 Pareto 方向转换的原因，并能用正常/复杂/异常三类输入验证。相同思路可迁移到文件转码 API（把阻塞转码移入 worker、保持状态更新在线程安全边界）和超参数搜索（先筛预算约束，再调用真实 evaluator；无评估时不生成虚构质量分）。本次不是对 NAS 学术性能、GPU 数值一致性或公网多租户部署的认证。

### 实际验证结果

- 修前快照执行新增回归：**15 failed, 1 passed**，17.88 秒；15 项缺陷回归按预期失败，1 项原有 YAML > env 契约检查通过，日志 `/tmp/vitriol-api-baseline-tests.log`。
- 当前工作树执行新回归及现有 API/config/security/NAS 测试：**150 passed**，10.73 秒，仅 1 条既有 RL ExperimentalWarning，日志 `/tmp/vitriol-api-tests.log`。
- 验证命令：`OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -o addopts='' --timeout=60 tests/test_api_config_nas_regressions.py tests/test_api_server.py tests/test_api_models_dynamic.py tests/test_api_nas_job.py tests/test_config_system.py tests/test_security_context.py tests/test_generation_config_security_audit.py tests/test_generation_config_resolution.py tests/test_nas_targeted.py tests/test_nas_rl_compat.py tests/test_trust_remote_code_policy.py -q`。
- `ruff check src` 全源码通过；本任务改动的 3 个源码文件与 2 个测试文件也全部通过；compileall、git diff --check 通过。按要求另运行 `ruff check .`，报告 355 条未通过项（含 demos、既有 tests 等），`ruff check src tests` 报告 194 条测试目录问题；未进行与本任务无关的大批格式改动。完整日志 `/tmp/vitriol-api-full-ruff.log`、`/tmp/vitriol-api-src-tests-ruff.log`。
- 环境：Python 3.11、PyTorch 2.6.0 CPU、transformers 4.57.6、numpy 1.26。未下载模型，生成 API 回归使用可控 stub 验证真实异步调度和 HTTP 行为；不据此宣称真实 GPU 模型吞吐或多进程部署通过验证。

最终改动文件：`src/vitriol/api/server.py`、`src/vitriol/config/manager.py`、`src/vitriol/nas/targeted_nas.py`、`tests/test_api_server.py`、`tests/test_api_config_nas_regressions.py`。
