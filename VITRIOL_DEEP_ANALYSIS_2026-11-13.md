# Vitriol 深度分析与修复验证（2026-11-13）

基线提交：`d7311fd`。版本：`0.3.1`。原始工作区干净，使用独立 `.venv` 验证；没有下载模型权重。前八轮分析阶段的修改均保留在本地；随后按用户要求分批提交并同步至 GitHub `isLinXu/Vitriol` 的 `main` 分支，见下方同步记录。各轮文档中的“未提交、推送或发布”描述的是该轮验证结束时的状态。

本次工作以可执行证据为依据：保留原始源码快照，先确认缺陷，再验证修复；完整运行原有测试、新增回归及本地集成链路。这里的“通过”指列出的测试和执行路径，不表示所有实验功能、模型家族、GPU 后端或网页交互已经得到认证。

后续推进记录：[第二轮 KV 旋转、连续解码与后端选择验证](docs/audits/2026-11-13/kv-followup.md)。该记录包含第二轮的最新测试结果；下文第 7 节保留首轮验证数据，便于对照。

**最新第十三轮结果（2026-11-14）**：调度改为任务/容量变化通知，修复无空闲节点时的队列重排与一秒轮询延迟，保留直接队列入队兼容性。新增 **14 项**回归，专项 **108 passed**；5 项事件循环场景分别在实际 Python 3.9、3.12 下通过。本地完整非集成测试 **2989 passed、19 skipped**，语句覆盖率 **71.99%**。代码已推送为 `85a04c2`；[远程 Python 3.9–3.12 四版本 CI 及分发包构建全部通过](https://github.com/isLinXu/Vitriol/actions/runs/43566009980)。详见[第十三轮调度与队列审计](docs/audits/2026-11-14/distributed-scheduling-followup.md)。

**第十二轮结果（2026-11-14）**：心跳与任务执行监控改用单调计时，保留公开 Unix 时间戳；修复时钟跳变误判、短超时检查延迟和执行期限边界，校验监控参数，并验证重试/断开/重启后的计时重置与清理。新增 **23 项**回归，专项 **94 passed**；本地完整非集成测试 **2975 passed、19 skipped**，语句覆盖率 **71.94%**。代码已推送为 `511d717`；[远程 Python 3.9–3.12 四版本 CI 及分发包构建全部通过](https://github.com/isLinXu/Vitriol/actions/runs/43561393647)。详见[第十二轮监控计时审计](docs/audits/2026-11-14/distributed-watchdog-followup.md)。

**第十一轮结果（2026-11-14）**：修复任务等待的零/NaN 超时、短期限额外休眠及系统时钟跳变问题，隔离提交分片列表和状态查询的嵌套结果。新增 **20 项**回归，验证等待者取消不影响排队/执行任务，实际执行失败与重试计数正确。专项 **71 passed**，本地完整非集成测试 **2952 passed、19 skipped**，语句覆盖率 **71.92%**。代码已推送为 `8ed7308`；[远程 Python 3.9–3.12 四版本 CI 及分发包构建全部通过](https://github.com/isLinXu/Vitriol/actions/runs/43558540909)。详见[第十一轮任务接口审计](docs/audits/2026-11-14/distributed-task-api-followup.md)。

**第十轮结果（2026-11-14）**：修复分布式协调器及 worker 重复启动、停止后协程遗留、任务重启恢复，以及超时/断开后的旧执行污染。本地完整非集成测试 **2932 passed、19 skipped**，语句覆盖率 **71.92%**；专项 **51 passed**，新增 **12 项**回归。代码与测试已推送为 `fab1d81`；[远程 Python 3.9–3.12 四版本 CI 及分发包构建全部通过](https://github.com/isLinXu/Vitriol/actions/runs/43556899686)。模块仍为模拟执行，不代表真实多机生成已验证。详见[第十轮分布式生命周期审计](docs/audits/2026-11-14/distributed-lifecycle-followup.md)。

**第九轮结果**：**GitHub Python 3.9、3.10、3.11、3.12 四版本 CI 及构建任务全部通过**。修复 GitHub CI 的 Ubuntu 26.04 / Python 3.9 解释器缺失、隔离构建依赖、协调器事件循环绑定及浮点精确断言问题，并新增失败摘要和测试报告留存。本地完整非集成测试 **2920 passed、19 skipped**，语句覆盖率 **71.76%**；专项 **107 passed**，新增 4 项队列场景分别在实际 Python 3.9、3.12 中通过。远程通过的提交为 `b18a3ca`，详见[第九轮 CI 审计](docs/audits/2026-11-13/ci-followup.md)。本轮没有重跑模型集成套件。

**第八轮结果**：**2916 项非集成测试、39 项集成测试通过，19 项跳过；语句覆盖率 71.77%**。新增 12 项测试，修复 fork 子进程误解锁父进程、存活子进程继续占用已退出父进程的锁，以及继承上下文误关复用描述符的问题；显式解锁异常不再掩盖原生成异常。Linux 真实 Llama 验证了主进程强退、子进程仍存活时的恢复，已提交权重逐字节一致且模型 forward 正常。Windows、网络文件系统、GPU 和绕过 Python fork 回调的原生路径尚未实测。详见[第八轮 fork 锁生命周期审计](docs/audits/2026-11-13/fork-ownership-followup.md)。

**GitHub 分批同步（2026-11-13）**：按功能及依赖拆分为以下 7 批，各批分别提交、推送。此次同步没有修改已验证的源代码与测试内容，另更新本文的同步状态。该批次的测试数据来自上传前第八轮本地完整验证，当时尚未核验 GitHub Actions，也不表示每个中间提交都单独运行了完整套件；后续远程修复与通过记录见第九轮。

| 批次 | 内容 | 提交 |
|---|---|---|
| 1 | CI 依赖、离线测试与 bench 公共导出 | [bab1d4d](https://github.com/isLinXu/Vitriol/commit/bab1d4d) |
| 2 | 生成完整性、检查点恢复、目录互斥与 fork 生命周期 | [94f9c27](https://github.com/isLinXu/Vitriol/commit/94f9c27) |
| 3 | KV 数值、旋转、连续解码与运行时补丁 | [063f264](https://github.com/isLinXu/Vitriol/commit/063f264) |
| 4 | 缓存接入、原生历史回退与 hook 所有权 | [2df80c1](https://github.com/isLinXu/Vitriol/commit/2df80c1) |
| 5 | API 默认值、配置覆盖、任务处理与 NAS 评估 | [0851292](https://github.com/isLinXu/Vitriol/commit/0851292) |
| 6 | 因果 PPL、模型指纹与有界权重抽样 | [8e8ea3e](https://github.com/isLinXu/Vitriol/commit/8e8ea3e) |
| 7 | README、CHANGELOG、总报告与八轮审计记录 | 包含本文的文档提交 |

**第七轮结果**：**2905 项非集成测试、38 项集成测试通过，19 项跳过；语句覆盖率 71.75%**。新增 22 项测试，在标准生成入口增加输出目录非阻塞互斥，拒绝同目录并发与重入，并仅在取得锁后清理保留命名空间的遗留临时文件。Linux 真实 Llama 生成已验证独立竞争进程不改动任何产物、正常退出释放及 SIGKILL 后恢复。当时未覆盖的 fork 继承描述符问题已在第八轮补充处理。详见[第七轮输出目录互斥审计](docs/audits/2026-11-13/output-ownership-followup.md)。

**第六轮结果**：**2885 项非集成测试、36 项集成测试通过，19 项跳过；语句覆盖率 71.73%**。新增 60 项测试，修复最终分片重命名与元数据中断恢复、旧目标文件误用、索引非原子写入及 SparseSpec 模块重导入后的记账错误。random/quantum 的本地 Llama 通过重命名后进程强退与独立进程恢复验证。稀疏校验改为与物理分配无关的版本化分块摘要；1 GiB 稀疏文件实测读取量降至约 1 MiB，旧整文件 SHA-256 检查点仍可验证。详见[第六轮收尾恢复与稀疏校验审计](docs/audits/2026-11-13/finalization-followup.md)。

**第五轮结果**：**2829 项非集成测试、32 项集成测试通过，19 项跳过；语句覆盖率 71.67%**。新增 47 项测试，修复分片边界 RNG 恢复、检查点内容与计数校验、SparseSpec 记账，以及 Quantum 跨进程随机性和模型加载元数据。11 种内置策略通过恢复对比；random/quantum 的本地 Llama 通过独立进程恢复、权重精确对照及 forward。学习型策略因缺少网络状态而重新生成；GPU 尚未实测，摘要校验增加文件读取。详见[第五轮生成恢复审计](docs/audits/2026-11-13/generation-resume-followup.md)。

**第四轮结果**：**2784 项非集成测试、30 项集成测试通过，19 项跳过；语句覆盖率 71.36%**。新增 45 项测试，修复通用模型 cache hook 接入、请求关联、完整历史故障回退、缓存 mutation 同步及安装回滚。为保证回退正确，通用 hook 现在保留原生 KV 副本；store 字节估计不代表总显存节省。详见[第四轮模型缓存接入审计](docs/audits/2026-11-13/cache-hook-followup.md)。

第三轮结果：**2745 项非集成测试、24 项集成测试通过，19 项跳过；语句覆盖率 71.06%**。第三轮净新增 40 项测试，修复运行时缓存失效、补丁所有权与异常/并发退出、SDPA 参数传递、稀疏注意力 dropout/掩码、全屏蔽梯度和 FP64 范数溢出。详见[第三轮运行时审计](docs/audits/2026-11-13/runtime-followup.md)。

第二轮最终结果：**2708 项非集成测试通过、21 项集成测试通过，19 项跳过；语句覆盖率 70.90%**。新增 87 项单元回归及 1 项本地模型集成。修复通用非 2 次幂旋转、adaptive KV 连续追加/重建、GQA/半精度分配、滑窗容量，以及兼容初始化覆盖 SDPA 导致运行时补丁失效的问题。

## 理解验证状态

| 核心机制 | 源码理解 | 实际验证 | 范围 |
|---|---|---|---|
| 配置→空模型→策略→分片→索引 | 完成 | 分片、故障注入、恢复、真实 tiny Llama forward | CPU、本地文件 |
| 模型验证器 | 完成 | 尺寸错误、缺失张量、重用报告、正常模型 | HF 标准加载器 |
| API、安全配置与作业状态 | 完成 | HTTP 429、默认值、事件循环、串行任务、SSE overflow | TestClient 与线程事件 |
| KV 压缩与注意力 | 完成关键路径 | FP32/FP16/BF16、形状、掩码、padding、窗口、dropout | CPU 数值回归 |
| PPL/NAS 评估 | 完成关键路径 | 因果 NLL、真实 evaluator 调用、未测量指标 | 确定性小样本 |
| WebUI/GPU/真实语料质量 | 部分 | 见验证结果与限制 | 不作未运行的性能保证 |

## 项目完整地图

原始源码约 6.3 万行，主要 Python 模块包括 32 个 core 文件、28 个 kv 文件、33 个 arch_viz 文件、27 个 cli 文件、23 个 strategies 文件；原测试有 158 个 Python 文件。完整枚举见 [项目地图](docs/audits/2026-11-13/project-map.md)，它记录的是审查开始时的文件清单，不包含本次新增回归文件。

| 层次 | 入口/模块 | 职责与本次覆盖 |
|---|---|---|
| 用户入口 | cli/main.py、api/server.py、webui/app.py | CLI 懒加载、HTTP 作业、可选图形界面 |
| 主流程 | core/generator.py、pipeline、validator.py | 从配置生成结构一致的测试权重，输出并验证文件 |
| 生成策略 | strategies、adapters | 张量形状/dtype/存储格式与模型家族适配 |
| 推理与评价 | kv、patches、bench、nas | KV 算子、运行时补丁、基准与搜索 |
| 可视化 | arch_viz、viz、visualization、vocab_viz | 配置/权重解释与 HTML 输出 |
| 横切能力 | config、security、utils、telemetry、resilience | 参数来源、离线控制、指纹、检查点与审计信息 |

核心调用链：CLI/API → GenerationConfig/SecurityContext → MinimalWeightGenerator → meta 模型与策略 → 分片/索引/config/manifest → ModelValidator。`check` 在这条链之外编排分析、可视化、生成、验证和最终报告。

## 1. 快速概览

项目的主要价值是把模型结构与训练权重分离，用本地小产物验证架构、导出和加载链路。Ultra 等策略利用共享存储压缩测试权重，适合架构调试和 CI；它们没有恢复预训练模型知识，生成文件能被加载也不能证明任务精度。KV 量化是另一条链路，需要独立证明注意力近似和质量评价正确，不能用生成权重的压缩比代替推理效果。

本次发现既有回归与新增边界缺陷并存。原始完整套件在受控环境中为 **2480 passed、29 failed、22 skipped**，其中 27 项失败来自 bench 公共导出路径失效，1 项是测试仍引用重构前的私有常量，1 项是单元测试意外访问 Hub 导致超时。新增回归还暴露了原套件没有覆盖的分片丢失、校验误报、数值错误和 API 边界。

## 2. 背景与动机

优先修复的是“结果看起来成功但内容错误”的路径。文件能写出、模型能打开、接口返回 200、张量形状合法，各自都只是中间条件。索引必须指向实际存在的每个张量，验证器必须拒绝自动补出来的随机权重，注意力必须对正确的 key 维度归一化，指标必须来自实际观测。这些不变量决定用户能否信任最终报告。

修复保留了已确认的兼容约束：CLI/显式参数 > YAML > 环境变量 > 默认值；原始分片索引存在时保留原始分片边界；bench 继续懒加载；API 生成保持串行，重任务移出事件循环。配置优先级的旧头注释与 README/现有测试冲突，最终纠正文档而非改变用户行为。

不采用全面重构，是因为项目同时包含稳定主流程和大量实验能力。可证实的局部修复更便于通过历史测试对照；例如把伪造 KL 改成未测量，比继续返回一个平滑但没有物理意义的数字更符合评估语义。需要重新定义算法或接口的部分在限制中明确列出。

## 3. 核心概念网络

| 概念 | 为什么需要 | 与其他概念的关系 |
|---|---|---|
| 逻辑张量与物理存储 | stride=0 可让大量逻辑元素共享少量数据，逻辑 numel 不能直接代表文件大小 | 分片阈值、总字节数、抽样内存都必须区分两者 |
| 配置来源与安全来源 | None、False、未提供有不同含义，安全字段还需记录 provenance | API 参数需交给统一 builder/resolver，不能先随意 bool() |
| 检查点与落盘提交 | 生成到内存不等于文件已经可靠写入 | 只记录完整落盘分片；原子替换与签名共同保证恢复边界 |
| 加载成功与内容一致 | HF 可自动初始化缺失参数，加载成功未必意味着输入完整 | validator 必须检查 loading diagnostics |
| QJL 逐向量内积与两两内积 | 相同形状可能表示逐元素配对，也可能是 prefill 的 QKᵀ | attention 显式要求 all_pairs，避免猜测语义 |
| 自生成 PPL 与语料 PPL | 不同运行可能生成不同文本，其评分目标不同 | 报告注明 self-generation 范围，KL 需独立收集对齐 logits |
| 搜索目标与真实评估 | evaluator 的输出才有模型质量语义，随机数没有 | 约束先过滤，再调用 evaluator，再比较目标和 Pareto 支配 |

这些概念可以按两个方向自检：从文件回到模型时，检查结构、dtype、键与权重内容是否一致；从统计数字回到观测时，检查采样对象、归一化轴、目标 token 和实际测量来源。两者都避免只凭界面“成功”做判断。

## 4. 算法与理论

分片分组对张量条目做线性遍历，时间 O(P)，P 为参数和持久 buffer 的条目数；内存主要由一个目标分片的数据决定。原始分片不能在保持其索引边界的同时任意拆小，因此 `max_shard_size` 在这种场景不是强制上限；没有原始索引时按实际张量存储字节切分。单个超限张量仍必须单独保存。

非连续张量抽样改为把随机逻辑下标映射回多维坐标，样本数 K、张量维数 D 时额外索引开销 O(KD)，不先把 O(N) 个逻辑元素 reshape 成连续副本。这样才能在部分广播的巨大逻辑张量上维持抽样内存预算。全量统计仍只在元素总量不超过预算时执行。

compute-skip 的块对齐补齐值必须在 softmax 前以负无穷排除，否则零 key 仍有 exp(0) 的概率质量。全部 key 被 mask 时定义输出为零，并在半精度路径使用 float32 归一化，避免 1e-12 下溢后出现 0/0。这里修的是数学语义，不以更快的单次运行时间代替正确性证明。

约束 NAS 的主要成本是 N 次候选评估 O(NE)，E 为 evaluator 成本；先检查约束可减少不必要评估。现有 Pareto 前沿维护最坏 O(N²M)，M 为目标数，它不是完整 NSGA-II。PPL 使用 exp(平均负对数似然)，每个生成 token 必须由产生它的那一份 logits 评分，不能在消费该 token 后用下一份 logits 反向评分。

理论依据在本次直接落实为可运行对照：数值算子与 torch SDPA 比较，因果 NLL 与手算确定性模型比较，分片索引与 safetensors/PyTorch 实际键集合比较。没有引用未核验的性能数字。

## 5. 设计模式

策略模式把参数生成与持久化格式隔离：主流程必须尊重 strategy 的 file_extension、dtype 和实际存储尺寸，否则 Ultra 与普通 dense 权重容易混淆。原子提交把“写临时文件”和“替换有效文件”分开，中断时旧文件仍可读；这比直接写目标文件更适合检查点恢复。

HF facade 是安全策略的统一入口。普通 from_pretrained 能接收 local_files_only，但 from_config 会把参数继续交给模型构造器，因此可信自定义模型的离线路径需先本地解析动态类，再调用纯构造逻辑。不能简单丢掉离线标志，也不能盲传 Hub 参数给严格构造器。

API 采用 worker 线程执行阻塞生成，事件循环负责状态更新；专用锁保持生成串行，避免多个生成器同时更改进程级库状态。SSE 使用单调序号，而不是有界 deque 的长度作为游标。公共 bench API 继续使用模块级懒加载，把导出的符号路由到实际 owner，从而既保留调用方式，也不让轻量导入加载 torch。

## 6. 关键代码深度解析

### 核心片段清单（6A）

| 片段 | 位置 | 关键问题 | 详细记录 |
|---|---|---|---|
| #1 分片提交与恢复 | core/generator.py、incremental.py、validator.py | 覆盖、丢权重、误恢复、加载伪成功 | [核心审查](docs/audits/2026-11-13/core.md) |
| #2 QJL/compute-skip 与因果 PPL | kv/codec.py、cache_store.py、bench/ppl_evaluator.py | 维度、dtype、mask、评分时序 | [KV/PPL 审查](docs/audits/2026-11-13/kv.md) |
| #3 API 任务和配置入口 | api/server.py、config/manager.py、nas/targeted_nas.py | None 默认值、阻塞、限流、伪评分 | [API/NAS 审查](docs/audits/2026-11-13/api.md) |

### 片段 #1：完整分片才是恢复单位

#### 1.1 代码整体作用

原生成器达到阈值就保存当前 buffer，但后续又把同一目标分片当作新的空文件覆盖，最终 flush 还跳过已写分片中的残余张量。索引记录过的名字仍然存在，因此仅检查 index.json 或文件数量不会发现问题。现在以源索引对应的整个分片为组，只完整写一次；无索引才创建新的编号分片。

#### 1.2 核心逻辑分析

加载配置→构建 meta 模型→计算恢复签名→检查 checkpoint 文件/shape/dtype→恢复或安全重建→生成并原子落盘→记录 checkpoint→索引和元数据→清除 checkpoint。检查点里的名字只来自成功写出的 shard_map。异常路径不把未落盘 tensor 计为完成，恢复时也不会先删掉自己正准备复用的文件。

#### 1.3 实际代码

来源：[src/vitriol/core/generator.py](src/vitriol/core/generator.py)，第 665 行起。

```python
        def persist(data, filename):
            nonlocal shard_count
            self._save_shard(data, filename, shard_count)
            saved_dtypes.update({n: str(t.dtype) for n, t in data.items() if n in self.shard_map})
            shard_count += 1
            self.incremental.save_checkpoint({
                "signature": signature,
                "generated_param_names": list(self.shard_map),
                "shard_count": shard_count,
                "total_size": self.total_size,
                "shard_map": dict(self.shard_map),
                "tensor_dtypes": dict(saved_dtypes),
            })

        # Generate original shards together and write each exactly once. Flushing
        # fragments to the same filename overwrites previously saved tensors.
        # Original shard boundaries take precedence over max_shard_size.

        with tqdm(total=len(export_items), initial=len(generated_names),
                  desc="Generating tensors") as pbar:
            for target, items in groups.items():
                if items and all(name in generated_names for name, _ in items):
                    continue
                buf: Dict[str, Any] = {}
                buf_bytes = 0
                for name, param in items:
                    if name in generated_names:
                        continue
                    tensor = self.strategy.generate_tensor(param.shape, param.dtype, name)
                    nbytes = self._estimate_tensor_nbytes(
                        tensor, fallback_numel=int(param.numel()), fallback_dtype=param.dtype)
                    if not target and buf and buf_bytes + nbytes > self.max_shard_size:
                        persist(buf, f"{self.strategy.get_shard_prefix()}-{shard_count + 1:05d}.{self.strategy.file_extension}")
                        buf = {}
                        buf_bytes = 0
                    buf[name] = tensor
                    buf_bytes += nbytes
                    self.total_size += nbytes
                    pbar.update(1)
                if not buf and target and not items:
                    ph_key = f"__vitriol_pad__{target}"
                    buf[ph_key] = self.strategy.generate_tensor((1,), torch.float16, ph_key)
                if buf:
                    filename = target or f"{self.strategy.get_shard_prefix()}-{shard_count + 1:05d}.{self.strategy.file_extension}"
```

#### 1.4 关键设计点

签名绑定输入、配置和模型结构，防止复用旧任务进度；实际读取分片元信息防止只相信 JSON。原子替换保证单文件提交边界，但不等于多文件事务；元数据阶段中断可能安全重建而非精确续跑。源目录与输出目录相同时直接拒绝，分片文件名限定 basename，避免清理或写入越出目标位置。

#### 1.5 三组输入对照

正常：一个源分片包含五个参数，阈值很小，最终仍完整保存五个参数。复杂：无源索引时先保存完整分片，发生故障后恢复已落盘条目、只生成未完成参数。异常：checkpoint 指向缺失或 dtype 不符的分片，整体重建，禁止跳过这些参数。另用真实 tiny Llama 生成 8 分片、12 tensors，重新加载并执行 forward，确认不仅是 mock 层的成功。

#### 1.6 使用注意与改进

原始索引边界优先，超大单分片仍需要相应内存。首轮尚未保存 RNG；第五轮已增加分片边界随机状态与内容摘要校验，11 种内置策略的 CPU 恢复已做逐张量对照。学习型策略仍重新生成，GPU 与并发恢复未验证。后续可设计显式“重新分片”模式和多文件事务，但需要同步改变产物契约。validator 现在检查 HF missing/unexpected/mismatched/error_msgs；不遵守标准返回协议的自定义加载器仍无法被独立审计。

### 片段 #2：注意力和 PPL 的轴与时间必须一致

#### 2.1 代码整体作用

相同形状的 Q/K 在 QJL helper 中原本代表逐向量内积，但 prefill attention 需要所有 query 对所有 key 的两两内积。用形状隐式猜测造成输出维度错误；显式 all_pairs 参数保留旧 helper 行为，同时让 attention 表达自己的需求。PPL 则需要同样明确“当前 token 由哪个时刻的概率分布产生”。

#### 2.2 核心逻辑分析

QJL base 与残差在 float32 累加；窗口截短后禁用旧完整序列的 packed proxy；softmax 前排除 padding/all-mask；输出恢复 query dtype。PPL 先由当前 logits 选择 token、计算该 token NLL，之后才把 token 输入模型预测下一个 token。EOS 或达到长度上限时不再多做一次 forward，attention_mask 随序列增长。

#### 2.3 实际代码

来源：[src/vitriol/bench/ppl_evaluator.py](src/vitriol/bench/ppl_evaluator.py)，第 341 行起。

```python
        for step in range(max_new_tokens):
            current_input = next_token_logits.argmax(dim=-1)
            # Score the token with the distribution that selected it, before
            # feeding it back into the model to predict the following token.
            token_id = current_input.item()
            token_nll = F.cross_entropy(
                next_token_logits.float().reshape(-1, next_token_logits.size(-1)),
                current_input.reshape(-1),
            )
            total_loss += token_nll.item()
            total_tokens += 1
            per_token_nlls.append({
                "phase": f"decode_step_{step}",
                "nll_sum": token_nll.item(),
                "tokens": 1,
            })

            generated_ids.append(token_id)

            # Stop at EOS
            eos_ids = set()
            if self._tokenizer.eos_token_id is not None:
                eos_ids.add(int(self._tokenizer.eos_token_id))
            cfg_eos = getattr(getattr(self._model, 'config', None), 'eos_token_id', None)
            if isinstance(cfg_eos, int):
                eos_ids.add(cfg_eos)
            if token_id in eos_ids or step + 1 == max_new_tokens:
                break

            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    attention_mask.new_ones((attention_mask.shape[0], 1)),
                ], dim=-1)
            step_out = self._model(
                input_ids=current_input,
                attention_mask=attention_mask,
                past_key_values=past_kv,
                return_dict=True,
                use_cache=True,
```

#### 2.4 关键设计点

计算精度与存储精度可以不同。将 fp16/bf16 projection 以 float32 运算是为避免 matmul dtype 错误及归一化下溢，并不表示做了 GPU 性能优化。未收集对齐 logits 时 KL 返回 null、layer 列表为空，不能把 token mismatch 任意缩放后当分布距离。没有 KV 遥测时也不宣称 100% 或固定四倍内存节省。

#### 2.5 三组输入对照

正常：query (1,2,3,16) 的 prefill 输出仍为 (1,2,3,16)，旧版可错误输出 (1,2,2,16)。复杂：key 长度不是块长整数倍，padding 不占注意力概率；修前与参考结果最大绝对误差为 0.313884。异常：全部 key 被 mask，输出为有限零值。确定性生成 [2,3,4] 的 PPL 修前为 22.6607，按正确因果分布应为 1.12821，修后与手算一致。

#### 2.6 使用注意与改进

当前 PPL 是 prompt 与各次自生成 continuation 的评分，不是统一 held-out corpus 的质量评估；比较不同续写文本时目标不同。真实 logits KL、各层 activation、GPU 峰值显存和吞吐仍需要独立测量。未覆盖的非 2 次幂 Hadamard 截断路径在限制中保留，不能将本次数值回归推广成所有 codec 的无条件正确性。

### 片段 #3：API 状态更新与实际工作分离

#### 3.1 代码整体作用

Pydantic 序列化会保留 None，dict.get 的 fallback 不会替换一个已存在的 None；bool(None) 还会误把省略网络选项解释为禁网。API 必须先保留“未提供”的语义，再交给配置解析。与此同时，async 函数内直接调用同步生成器仍然阻塞事件循环，单纯把函数写成 async 不会让健康和日志请求自动响应。

#### 3.2 核心逻辑分析

请求→过滤非 None 网络覆盖→默认输出目录→to_thread→串行锁→配置/生成器构造/生成→事件循环更新状态。HTTP middleware 直接构造 429 响应并保留 Retry-After；任务数从实际状态表统计，不再维护一个只入不出的重复队列。日志条目分配递增 sequence，缓冲满后长度不变也能继续推送。

#### 3.3 实际代码

来源：[src/vitriol/api/server.py](src/vitriol/api/server.py)，第 856 行起。

```python
async def process_generation_job(job_id: str) -> None:
    """Process a generation job."""
    job = active_jobs.get(job_id)
    if not job:
        return

    job["status"] = "running"
    logger.info(f"Starting generation job: {job_id}")

    try:
        request = job["request"]

        overrides = {
            "strategy": request.get("strategy", "compact"),
            "dtype": request.get("dtype", "bfloat16"),
            "max_shard_size": request.get("max_shard_size", "5GB"),
            # Remote model code remains an explicit per-request opt-in.
            "trust_remote_code": bool(request.get("trust_remote_code", False)),
        }
        # Omitted optional fields are serialized as None; preserve config defaults.
        for key in ("allow_network", "local_files_only"):
            if request.get(key) is not None:
                overrides[key] = request[key]

        output_dir = request.get("output_dir") or f"./output/{request['model_id'].split('/')[-1]}"
        result_data = await asyncio.to_thread(
            _run_generation, request["model_id"], output_dir, overrides,
        )

        job["status"] = "completed"
        job["completed_at"] = datetime.now(timezone.utc).isoformat()
        job["output_dir"] = result_data.get("output_dir") or request.get("output_dir")
        job["result"] = result_data

        logger.info(f"Generation job completed: {job_id}")

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        logger.error(f"Generation job failed: {job_id}", exc_info=e)


async def process_nas_job(job_id: str) -> None:
    """Process a NAS job."""
    job = active_jobs.get(job_id)
    if not job:
```

#### 3.4 关键设计点

锁在 worker 中获取，不阻塞事件循环；单任务、批量任务使用同一路径，避免只修复一类入口。远程代码默认仍需显式授权；批量字符串 "false" 由统一布尔解析处理，不再经 bool("false") 变为真。NAS 同样尊重调用者意图：实际调用提供的 evaluator，不再将 random.random 当模型质量。

#### 3.5 三组输入对照

正常：仅提供 model_id 时默认输出 ./output/model，网络策略按配置解析。复杂：两任务同时到达，worker 保持串行而状态协程可继续执行；SSE 超过缓存容量仍输出新序号。异常：超过限流返回 429 而非 500；evaluator 返回 NaN/Inf 明确拒绝，避免污染最优解与 Pareto 判断。

#### 3.6 使用注意与改进

串行锁保证本进程内 API 生成顺序，不提供跨进程同目录互斥。作业历史仍保存在内存，重启丢失，也没有持久取消/重试协议；这些需要产品层明确契约后实现。NAS API 的 algorithm/dataset 等参数尚未完整驱动真实算法，现有结构估计不能冒充数据集精度和硬件延迟。

## 7. 测试用例分析

最终结果（最后一轮包含所有已安装可选依赖，按主 CI 的串行方式执行）：

| 检查 | 结果 | 说明 |
|---|---|---|
| 原始提交完整单元套件 | 2480 passed / 29 failed / 22 skipped | 同一 Python/torch/transformers 基础环境，含网络超时缺陷 |
| 修复后完整套件，4 workers | 2611 passed / 22 skipped | 44.05 秒；尚未安装 WebUI/jsonschema |
| 修复后完整套件，串行、全部 extras | **2621 passed / 19 skipped / 0 failed** | 70.51 秒；19 条警告为依赖弃用/实验功能等，不是失败 |
| 最终语句覆盖率 | **70.54%（17450 / 24737）** | 高于主 CI 的 60% 门槛，不等同分支覆盖率 |
| 本地集成套件 | **20 passed** | check / strategy compare / generator refactor，20.36 秒 |
| tiny Llama 实际生成→加载→forward | 通过 | 8 分片、12 tensors、7264 bytes；输出 [1,3,32] finite |
| 全源码 Ruff | 通过 | `ruff check src/`；原有测试目录仍有 194 条风格问题，基线为 196 条 |
| CI 范围 mypy | 通过 | api/server.py、cli/main.py、security/context.py，不冒称全库类型检查通过 |
| 构建与 evolution 定向检查 | 35 passed | 包含 wheel/sdist HTML 资源校验 |
| CLI version/help/bench/check | 通过 | bench 原入口已恢复 |
| 依赖一致性、编译与 diff 空白检查 | 通过 | `uv pip check`、compileall、git diff --check |

19 项跳过的具体边界：Hub smoke 未启用 12 项；需 Hub 下载的 adapter 1 项；HF config 加载不可用 1 项；空 tensor 评分不支持 1 项；TinyLLaMA 缓存缺失 1 项；Transformers 4.57.6 没有 Qwen3.5 类 1 项；可选 trace 模型 fixture 缺失 2 项。没有把这些计为通过。

原夜间命令也实际复现：20 个用例全部通过，但全库覆盖率仅 20.20%，继承 60% 门槛导致进程失败。修正后的 nightly 单独运行集成套件，完整主 CI 仍保留覆盖率门槛。

修前缺陷证据：API 新增回归 15 failed、1 兼容契约检查 passed；指纹/抽样新回归 12 failed；原 generator 的 5 个关键完整性用例全失败；KV/PPL 使用原 HEAD 源码做数值对照。原始源码通过快照或内存载入测试，没有回滚共享工作树。

测试特意覆盖失败后的状态：分片写到一半抛错、checkpoint JSON 序列化失败、可视化复制失败、验证器同一对象第二次调用、PPL 推理异常时恢复 patch，以及日志缓冲溢出。它们验证产物和后续行为，不只是断言某个内部函数被调用。

新增文件：test_core_generation_integrity.py、test_kv_numerical_regressions.py、test_ppl_numerical_regressions.py、test_api_config_nas_regressions.py、test_fingerprint_boundaries.py、test_weight_inspector_sampling.py、test_bench_public_exports.py。已有测试按真实语义调整：尺寸错误不应被随机重初始化掩盖；RoPE 常量从实际 owner 导入；evolution 单元测试通过本地 stub 避免网络。

指纹测试覆盖 BatchNorm affine 属性、float16/bfloat16、空参数和单维边界。抽样测试直接阻止大非连续 reshape，验证部分广播张量仍仅抽取预算内样本，同时确认 inspect_weights 的 max_per_layer_sample 参数被实际使用。新增 sample_size/sampled 明确统计来源；l2_norm 在抽样时仍是样本范数，不能解释为全量范数。

## 8. 应用迁移场景

场景一：从权重生成迁移到数据导出。共同原则是检查点只记录已提交的产物，不能记录尚在内存中的进度。需要把 tensor 名和 shape 校验替换成数据分区、schema 和分区内容校验；临时文件与原子替换仍适用。多文件事务和随机状态重放则要根据数据系统的新契约设计，不能从单文件原子写推导整体一致性。

场景二：从模型生成 API 迁移到文件转码服务。耗时构造和实际执行都应离开事件循环，状态更新保留在明确的并发边界；有界日志缓冲需用事件 ID 做游标。具体修改的是 worker 的任务函数、错误分类和结果 schema，限流响应、None 缺省值以及终态统计的验证方式可以直接迁移。

场景三：从架构搜索迁移到超参数优化。先筛掉违反预算的候选，再调用实际 evaluator，保留有限 score 和来源；没有评价时使用明确的“未评价/基准值”，不能制造随机质量。若新任务需要训练精度或真实延迟，应收集对应指标，而不是复用参数量经验式。

## 9. 依赖关系与复现

项目依赖 transformers 4.x、torch 2.x、numpy 1.x；独立环境采用 Python 3.11.18、torch 2.6.0+cpu、transformers 4.57.6、numpy 1.26.4；最终 WebUI 为 Gradio 6.17.3，huggingface-hub 0.36.2，所有 extras 联合求解后 `uv pip check` 通过。CPU 数值测试不要求 GPU，所有真实模型验证使用本地 tiny 配置。安装可选依赖必须和项目约束一起求解，单独安装新 WebUI 依赖可能升级 huggingface-hub 并破坏 Transformers 4.x 约束。

开发依赖补充 setuptools/wheel（构建测试）、httpx（TestClient）、jsonschema（配置 schema 校验），CI 安装 api extra 使 API 测试真正执行。夜间集成使用独立 addopts，避免继承完整套件的覆盖率门槛；这不降低主 CI 的 60% 要求。

```bash
# 首次安装：在合适 Python 的独立环境中联合安装，CPU torch 可预先单独安装。
uv pip install --python .venv/bin/python -e '.[dev,viz,api,webui]'

# 完整套件，含已有覆盖率要求。
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests \
  --ignore=tests/integration --timeout=60 -q --tb=short

# 本地集成链路，与修正后的 nightly 参数一致。
OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/integration -o addopts='' -v --tb=short

.venv/bin/ruff check src/
.venv/bin/mypy src/vitriol/api/server.py src/vitriol/cli/main.py \
  src/vitriol/security/context.py --ignore-missing-imports --follow-imports=skip
.venv/bin/python -m vitriol bench --help
```

## 10. 质量验证清单与限制

- 已核对正常、边界、异常恢复三类路径；已保留修前实证与修后完整测试结果。
- 全源码 Ruff、CI 关键模块 mypy、CLI help、构建产物检查与本地集成均按结果表记录；完整测试目录仍有既有风格告警，不将其描述为全部清零。
- 未进行真实 GPU/Triton、远端模型家族矩阵、长上下文语料质量、浏览器端交互或跨进程压力测试。
- 首轮发现的非 2 次幂 Hadamard 不可逆问题已在第二轮修复：采用正交、可自逆的分块变换并验证范数、内积、梯度及连续解码；旧版本的非 2 次幂旋转缓存需要重建。见后续报告中的算法依据与兼容范围。
- NAS 参数/显存估计对 GQA/MoE/MLA/Mamba 不完整；RL 存在动作、轨迹顺序与保存恢复等算法语义问题，详见模块报告。不能以本次套件通过宣称 RL 有效。
- 模型权重指纹仍含统计型/感知型摘要，不是精确文件内容的密码学证明。本次只修复 dtype、归一化层及空/退化形状的执行错误。
- 网页测试中一部分只检查模板标记，不能替代真实浏览器端功能回归。

## 覆盖率摘要

生成/持久化/验证、API/配置、安全加载、KV/PPL、NAS 评分、指纹及权重抽样完成定向源码检查与回归。其他 adapter、架构解析、进化、可视化、CLI、插件、遥测模块以完整既有测试加关键调用链检查覆盖。该范围是模块级覆盖声明，不是每行代码都获得形式化证明；详细模块清单与跳过理由分别见项目地图及模块审计。
