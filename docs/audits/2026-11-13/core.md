# Vitriol 核心生成 / 持久化 / 离线验证审计

## 范围与结构

入口 `MinimalWeightGenerator.generate` 通过 BootstrapStep/LegacyGenerateStep 调用核心生成流程；加载 HF 配置 → adapter 修补 → 可选 shrink → meta model 初始化 → 策略生成权重 → 分片/索引 → active/meta 配置、分词器、自定义代码、可视化与 manifest。策略决定真实存储格式（普通 safetensors、Ultra 的 PyTorch stride=0）；索引中的 weight_map 必须逐项对应落盘张量，total_size 表示真实存储字节而非 Ultra 的逻辑尺寸。ModelValidator 再从本地产物加载模型和可选推理。

## 已修复的问题

1. **分片静默丢权重**：原逻辑达到阈值后重写同一目标文件，后续 flush 覆盖此前内容；final flush 又跳过 flushed_shards 中仍有 residual 的分片。索引却仍保留旧张量。改为按原始目标分片分组，完整写一次，原始分片命名/数量保持一致；无源 index 时按实际字节新建分片，最终标准化 `-00001-of-00003` 命名。
2. **checkpoint 误恢复/未恢复**：原逻辑恢复前先删所有权重，然后依据 checkpoint 名字跳过生成，且正常路径未保存 checkpoint。现在只记录完整落盘分片，绑定源路径、配置、实际模型结构和原始 shard_map 的签名；校验文件、名字、shape、实际输出 dtype 及原始分片完整性；不兼容/损坏 checkpoint 整体安全重建。进度只更新一次，避免恢复时重复计数。
3. **中断破坏已有文件**：分片在输出目录同盘临时目录写完后 atomic replace，checkpoint 在同盘临时文件 JSON 序列化、flush/fsync 后 atomic replace。失败保留上一有效 artifact/checkpoint。策略未按声明格式写出文件时明确报错，不生成指向不存在文件的索引。
4. **源模型误删**：source/output 指向同一真实目录时立即拒绝，避免清理阶段删除输入权重。分片写路径限定 basename，避免路径逃逸。
5. **验证器伪成功与过期报告**：取消 size mismatch 时 ignore_mismatched_sizes 重试（会初始化随机替代权重），RuntimeError 不再降级 generic 模型掩盖；每次 validate 重置 report，修复后重试不会继承历史失败。HF 加载启用 output_loading_info，missing/unexpected/mismatched/error_msgs 均判失败，忽略项目保留的 __vitriol_pad__ unexpected 占位。真实 tiny Llama 缺少 model.norm.weight 的本地文件回归验证应失败。
6. **可视化异常损坏 active config**：在 rename 原配置后立即标记需恢复，后续 copy/move 失败也由 finally 还原。
7. **离线设置丢失**：generator._try_load_auto 之前硬编码可联网，已传真实 SecurityOptions；hf_loading 的 dynamic patch 解析添加 local_files_only。针对可信远程自定义模型 from_config，在离线模式先显式仅本地解析类，再调用不含 Hub 参数的 _from_config，避免 AutoModel 同时将 local_files_only 传给 Hub 与普通模型构造器的兼容问题。证据：本地缓存 Transformers 4.57.1 `models/auto/auto_factory.py:426` 的 from_config 实现（同一 kwargs 同时传 get_class_from_dynamic_module 和 model_class._from_config）。未变更全局环境、未下载任何模型。

## 取舍与残余界限

- 原始索引存在时以保留原始分片边界优先，max_shard_size 不强制拆散原始分片；峰值从多个原始分片同时缓冲改为一个原始分片。若单原始分片巨大仍需足够内存。无索引模型按阈值切分；单个张量大于阈值无法在 HF 单张量分片协议内再拆分。
- 首轮尚未保存 RNG；第五轮已修复内置无网络状态策略的分片边界随机恢复，详见 [第五轮审计](generation-resume-followup.md)。学习型策略仍需重新生成，GPU 与并发恢复未验证。
- 首轮遗留的最终重命名/元数据阶段恢复问题已在[第六轮审计](finalization-followup.md)修复；仍不提供整个输出目录的原子发布或断电持久性保证。
- validator 已检查 HF 标准 loading diagnostics；自定义/注入加载器若只返回 model 且不遵守 output_loading_info，则兼容返回，仅能证明该加载器成功，无法独立检查其参数修复行为。
- 尚未覆盖全部 adapter / 远程 trust_remote_code 仓库、巨型模型内存压力、GPU、多进程同时写同一输出目录、全部训练策略。生成 dummy 权重不验证语义推理质量。

## 验证

新增 `tests/test_core_generation_integrity.py` 覆盖分片阈值穿越、索引逐项落盘完整性、random/Ultra 双格式、无索引字节限制、故障注入与真实恢复、旧/坏 dtype checkpoint、原子落盘故障、源目录保护、路径逃逸、可视化失败还原、验证器误报/重复调用、离线策略传播。`tests/test_hf_loading.py` 新增 trusted custom class 本地解析与 dynamic patch 离线传播两项。

### 交叉验证（非项目支持依赖环境）

- genception Python3.12/Transformers5.x 环境，首轮新增核心与 HF 测试 23 passed。此环境超出 pyproject 上界，不作为正式兼容证明。
- 仅在内存载入 HEAD 原始 generator，再跑同一批关键回归：5 failed（random 和 Ultra 均漏权重、无索引未切片、中断无 checkpoint、stale checkpoint 引用不存在文件）。原工作区未回滚。日志 /tmp/vitriol-core-baseline.log。
- 本地 tiny Llama 随机权重生成→8 分片 / 12 tensors / 7264 bytes→实际 AutoModel reload→forward 输出 [1,3,32] 且全部 finite。没有 tokenizer 的纯配置 fixture，分词器保存仅发警告。脚本 /tmp/vitriol-tiny-e2e.py，日志 /tmp/vitriol-tiny-e2e.log。

### 最后一次核心集中交叉回归

命令：
```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/comp_robot/home/xu.lin/miniconda3/envs/genception/bin/python -m pytest \
 tests/test_core_generation_integrity.py tests/test_core_incremental_validator.py \
 tests/test_core_utils.py tests/test_core_modules.py tests/test_validator_security.py \
 tests/test_core_pipeline_bootstrap.py tests/test_core_pipeline_legacy_generate.py \
 tests/test_core_pipeline_steps.py tests/test_offline_semantics.py \
 tests/test_trust_remote_code_policy.py tests/test_hf_loading.py -q -o addopts=''
```
结果：**138 passed，2 条 pytest 插件配置警告，32.10s**。日志：`/tmp/vitriol-core-focused-final-cross.log`。请主代理在标准 `.venv` 准备好后用同命令（换解释器）复核；此交叉环境的 Transformers/NumPy 超出项目上界。

Ruff 对 8 个变更文件检查通过；`git diff --check` 通过；compileall 通过。

修改文件：
- src/vitriol/core/generator.py
- src/vitriol/core/generator_persistence.py
- src/vitriol/core/incremental.py
- src/vitriol/core/validator.py
- src/vitriol/utils/hf_loading.py（主代理授权扩展负责）
- tests/test_core_generation_integrity.py（新增）
- tests/test_core_incremental_validator.py（纠正原 size mismatch 误认证契约）
- tests/test_hf_loading.py

## 标准项目环境正式验证（最终）

标准 `.venv`：Python 3.11、torch 2.6 CPU、Transformers 4.57.6、NumPy 1.26；所有版本处于项目声明范围。

上述 11 个核心/HF 测试文件使用 `.venv/bin/python`、`OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`、`-q -o addopts='' --timeout=60` 执行：**138 passed in 12.64s，无失败、无 pytest warnings**。日志 `/tmp/vitriol-core-focused-standard.log`。

实际 tiny Llama 的纯本地配置→随机权重生成→8 分片/12 tensors/7264 bytes→AutoModel reload→forward 在标准环境再次成功，输出 `[1,3,32]` 且全部 finite。命令：
```bash
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python /tmp/vitriol-tiny-e2e.py
```
日志 `/tmp/vitriol-tiny-e2e-standard.log`。该纯配置 fixture 未提供 tokenizer，保存 tokenizer 预期发出缺少 tokenizer/protobuf 的提示；权重加载和 token-ID forward 不受影响，没有宣称文本 tokenization 验证成功。没有发现需要按标准依赖环境追加修复的问题。

### 全量回归中的旧测试导入修复

`tests/test_more_regressions.py` 原来从已重构的 `core.generator` 访问 `_ROPE_DEFAULTS`，改为导入实际 owner `core._generator_utils`；未恢复冗余生产别名。该文件测试辅助迭代器同步清除既有 Ruff UP028。标准环境整文件 **18 passed in 16.25s**，Ruff 通过；日志 `/tmp/vitriol-more-regressions-standard.log`。
