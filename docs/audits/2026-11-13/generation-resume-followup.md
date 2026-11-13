# 第五轮：生成恢复与产物完整性

本轮继续处理 Vitriol 的生成主链路，保留前四轮改动，没有提交、推送、发布或下载模型权重。环境仍为 Python 3.11.18、PyTorch 2.6.0+cpu、Transformers 4.57.6、NumPy 1.26.4。

## 已复现的问题与修复

1. **恢复后随机权重不同。** 检查点只保存已完成名称，没有保存随机状态；无源索引时，还会先生成下一个张量，再决定写出前一分片。因此仅在写出时读取 RNG 仍会错一张量。现在记录 Python、NumPy、Torch CPU 和已初始化 CUDA 默认生成器的状态；阈值触发的分片使用下一张量生成前的状态，最后一个分片使用生成后的状态。格式采用 JSON 和 Base64，不加载任意 pickle。先用独立生成器验证状态，再修改默认生成器；运行时版本不一致会拒绝恢复。

2. **同形状、同 dtype 的内容损坏被接受。** 每个已提交文件现在记录 SHA-256，恢复前验证摘要、键集合、形状、dtype、实际字节统计、分片数量及生成顺序前缀。包括空源分片的占位文件。检查点签名增加格式版本、策略 recipe、输出格式和实际分片阈值。旧版、损坏或不完整检查点整体重新生成。

3. **Quantum 跨进程结果不稳定，并影响其他随机操作。** 旧实现以 Python `hash(name)` 设种子且调用全局 `torch.manual_seed`。现在使用名称的 SHA-256 派生种子和策略内部的临时 `torch.Generator`；缩放计算不再重置默认随机流。

4. **Quantum 文件可读但模型不可加载。** 新增的真实模型测试发现，其 safetensors 自定义元数据缺少 `format: pt`，Transformers 因此拒绝加载。已补齐标准格式字段，保留量化元数据。

5. **SparseSpec 与检查点张量假设不兼容。** Sparse 策略返回描述对象，没有 `.dtype`/真实 storage；旧生成器使用错误的 fallback 字节数，写完后访问 `.dtype` 失败。现在按 SparseSpec.size 记账，并从文件元数据确认输出 dtype。Safetensors 检查使用 get_slice 元数据，不为读取 dtype 而物化整张权重；PyTorch 文件仍使用 weights_only 限制加载。

6. **学习型策略可能混合两个生成网络的产物。** Learned/HybridLearned 除 RNG 外还持有网络参数。当前检查点未保存该网络，所以恢复时明确警告并重新生成全部张量；未把“恢复随机状态”误称为已恢复整个策略。未知实现和非 CPU/CUDA 设备同样不进入可复现恢复路径。

## 可执行证据

修改前保存相关源码到 `/tmp/vitriol-phase5-before/`。先增加并运行 9 项回归：random、quantized、structured_sparse、lowrank 在有/无源索引下中断恢复，以及同形状内容改写；全部失败，日志 `/tmp/vitriol-phase5-before.log`。

最终本轮净新增 **47 项测试：45 项非集成、2 项集成**。

- 11 种内置策略 × 2 种分片方式，与同初始种子的完整生成逐张量精确比较。中断前故意额外消耗随机数，恢复前更换种子。
- RNG 的 JSON 往返、Python/NumPy 缓存高斯状态、非法 RNG 不改变现有随机流、CPU 不初始化 CUDA、Quantum 缩放不改变全局 RNG。
- 同形状权重改写、错误字节数/分片数/摘要/完成分组/RNG、空源分片不重复生成、学习型策略拒绝混合恢复。
- 真实本地两层 Llama：random、quantum 各启动三个独立 Python 进程，分别完整生成、中断、恢复；使用不同 PYTHONHASHSEED 和恢复种子。验证已提交张量没有重生成，最终索引和全部权重精确一致，重新加载后的 forward 输出 `[1, 3, 128]` 且全部有限。每次输出 21 个权重分片。

最终验证结果：

| 检查 | 结果 |
|---|---|
| 完整非集成测试 | **2829 passed，19 skipped，19 warnings**，57.19 秒 |
| 完整集成测试 | **32 passed**，48.25 秒 |
| 非集成语句覆盖率 | **71.67%**，17940 / 25033 |
| Ruff：全部 src 与本轮相关测试文件 | 通过 |
| CI 范围 mypy：api/server.py、cli/main.py、security/context.py | 通过；保留 untyped-body 提示 |
| git diff --check | 通过 |

19 项跳过原因与第四轮一致：Hub/模型下载禁用、缺少本地模型或 trace fixture、当前 Transformers 没有 Qwen3.5，以及空张量评分不支持。19 条 warning 未计为失败。

## 验证命令

```bash
PYTHONPATH=src GRADIO_ANALYTICS_ENABLED=False OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests --ignore=tests/integration --timeout=60 -q -ra --tb=short \
  --cov=src/vitriol --cov-report=term --cov-report=xml:/tmp/vitriol-phase5-coverage.xml \
  --junitxml=/tmp/vitriol-phase5-full.xml

PYTHONPATH=src GRADIO_ANALYTICS_ENABLED=False OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/integration -o addopts='' --timeout=90 -q -ra --tb=short \
  --junitxml=/tmp/vitriol-phase5-integration.xml
```

完整日志分别为 `/tmp/vitriol-phase5-full.log`、`/tmp/vitriol-phase5-integration.log`。源码和本轮测试文件运行 Ruff；CI 使用的 API/CLI/security 三文件 mypy 与 git diff --check 一并验证。

## 边界与代价

以下为第五轮完成时的限制。第六轮已优化稀疏空洞校验，并修复最终分片重命名后的恢复，详见[第六轮审计](finalization-followup.md)。

- 本轮实测 CPU。同版本、同生成设置、串行使用默认 RNG 是恢复等价的前提。CUDA 状态有保存/校验分支，但未实机验证 GPU；不承诺跨硬件或跨库版本逐位一致。
- 不保存自定义 Generator、策略网络参数或所有策略内部统计计数，也不保证时间戳、日志和内存统计与未中断执行完全一致。学习型策略当前重新生成，后续若要续跑，需要独立的策略状态协议。
- 摘要按 1 MiB 块读取，增加每个落盘文件的一次顺序读取和恢复时的一次读取；内存有界，但大文件及 sparse 文件的逻辑零区域仍有读取/哈希时间成本。摘要用于发现内容变化，不是可信签名。
- 无源索引的分片最终重命名后，如果元数据阶段中断，旧文件名可能不再匹配，仍会安全重新生成；尚未实现整个输出目录的多文件事务。
- 不支持多个进程同时写同一输出目录。原始分片边界优先于 max_shard_size 的约束保持不变。
