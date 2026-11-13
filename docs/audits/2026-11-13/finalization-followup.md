# 第六轮：生成收尾恢复与稀疏校验

本轮保留前五轮修改，继续处理分片生成后的恢复边界与稀疏文件校验成本。没有提交、推送、发布或下载模型权重；验证环境仍为 Python 3.11.18、PyTorch 2.6.0+cpu、Transformers 4.57.6。

## 问题与修复

### 1. 最终文件名与检查点脱节

无源索引时，分片先以 `model-00001.safetensors` 等名字落盘，最后改成含总数的规范文件名。旧实现既没有更新检查点，也没有更新生成器的 shard_map。此时配置、tokenizer 或 manifest 写入异常，会使下次恢复找不到原文件、重生成全部权重；生成器本身也保留了不存在的旧路径。

现在在移动文件前严格提交重命名计划。恢复时逐文件寻找原名或计划中的新名，并验证其摘要、键、形状及记账信息，允许同一次移动尚未完成的中间状态。索引发布成功后同步更新生成器和检查点中的文件名，后续元数据失败可以直接重试，无需再生成权重。

### 2. 已存在的目标文件可能被误用

旧代码遇到目标规范文件名已经存在就跳过移动，索引可能指向旧文件。现在以本轮已验证的分片原子替换目标。分片只移动一次；完成后生成器 shard_map 与索引一致。

### 3. 索引写入与检查点失败处理

索引改为同目录临时文件、flush/fsync、原子替换，序列化或写出异常不会留下半截 JSON。检查点增加可选 strict 模式：生成主链路的提交失败会传播异常；重命名计划无法保存时，不继续移动文件。普通 IncrementalGenerator 调用保留原有默认行为。

这些措施保证所测进程中断路径可恢复，不代表整个输出目录是一个原子事务，也不等价于断电持久性保证。

### 4. 稀疏文件摘要不再遍历全部空洞

上一轮的标准 SHA-256 会读取完整逻辑文件，即使大部分都是文件系统空洞。本轮使用带版本前缀的 `chunk-sha256-v1:` 内容摘要：纳入文件长度，以及每个非零 1 MiB 分块的位置、长度和 SHA-256；全零块隐含表示。支持 SEEK_DATA/SEEK_HOLE 的文件系统可以跳过空洞。

同样内容的稀疏文件和稠密文件得到相同摘要；不支持空洞查询时回退到顺序读取，结果不变。内容修改、零填充扩展、截断、移动数据位置均改变摘要。普通 I/O 错误不能被当成空洞。读取前后检查文件大小和修改时间，检测到并发修改时拒绝结果。旧版 64 字符整文件 SHA-256 检查点仍可验证恢复。

新摘要不是标准整文件 SHA-256 的十六进制值；格式已明确带版本标识，也不是可信数字签名。

### 5. 字节统计不依赖重新导入后的类型身份

完整套件发现 SparseSpec 在模块重新导入后可能不再通过旧对象的 isinstance 检查，导致回退到错误的输入 dtype 字节数。SparseSpec 现在提供 nbytes 属性；生成器首先读取真实 storage 字节数，再使用描述对象的 nbytes。这样既保留 Ultra 广播张量的紧凑存储统计，也避免 SparseSpec 类型身份变化造成误判。

## 验证证据

修改前备份到 `/tmp/vitriol-phase6-before/`，先运行新增的 18 项回归，全部失败：15 项是三种策略在五个元数据阶段中断后的恢复，另有旧目标文件、生成器路径失效及稀疏读取量测试。日志：`/tmp/vitriol-phase6-before.log`。

本轮净新增 **60 项测试：56 项非集成、4 项集成**。覆盖：

- random、Ultra、Sparse 的配置、自定义代码复制、tokenizer、README 元数据、manifest 阶段异常；恢复时禁止再次调用权重生成。
- random/Ultra 在提交重命名计划、移动第 1/2/3 个分片、发布索引、提交最终检查点时注入失败；恢复后的权重逐张量精确一致。
- 旧目标文件替换、索引序列化失败保留原文件、检查点提交失败停止生成、旧 SHA-256 兼容、不合法重命名路径。
- 稀疏/稠密摘要等价、空文件、空洞查询不可用、内容/位置/长度变更、短读取、并发改写/截断、I/O 错误及 storage/nbytes 记账顺序。
- 本地 Llama 的 random/quantum 跨进程验证扩为 6 种组合：张量生成中断、manifest 阶段异常、完成首个最终分片移动后 os._exit(73) 强退。独立进程恢复时检查已提交权重未重生成，最终全部权重与完整生成精确一致，模型加载与 forward 正常。

最终验证结果：

| 检查 | 结果 |
|---|---|
| 完整非集成测试 | **2885 passed，19 skipped，19 warnings**，56.98 秒 |
| 完整集成测试 | **36 passed**，131.69 秒 |
| 非集成语句覆盖率 | **71.73%**，18021 / 25125 |
| Ruff：全部 src 与本轮相关测试 | 通过 |
| CI 范围 mypy：API/CLI/security 三文件 | 通过，保留 untyped-body 提示 |
| git diff --check | 通过 |

19 项跳过与上一轮一致，涉及禁用的 Hub 下载、缺少的本地模型/trace fixture、当前 Transformers 不含 Qwen3.5，以及空张量评分不支持。

## 稀疏文件实测

使用真实 SparseStrategy 输出一个 536870912 元素的 BF16 张量；没有物化这个大张量。

| 指标 | 结果 |
|---|---:|
| 逻辑文件大小 | 1,073,741,950 字节 |
| 文件系统实际分配 | 8,192 字节 |
| 旧整文件摘要读取量 | 1,073,741,950 字节 |
| 新分块摘要读取量 | 1,048,702 字节 |
| 本机单次旧/新摘要耗时 | 1.236 秒 / 0.00462 秒 |

读取量减少约 99.90%。耗时仅为本机单次观测，不是跨硬件性能承诺。脚本 `/tmp/vitriol-phase6-sparse-benchmark.py`，原始结果 `/tmp/vitriol-phase6-sparse-benchmark.json`；临时大文件已自动删除。单元测试另用 64 MiB 稀疏文件，限制读取量不超过 2 MiB。

## 验证命令与边界

```bash
PYTHONPATH=src GRADIO_ANALYTICS_ENABLED=False OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests --ignore=tests/integration --timeout=60 -q -ra --tb=short \
  --cov=src/vitriol --cov-report=term --cov-report=xml:/tmp/vitriol-phase6-coverage.xml \
  --junitxml=/tmp/vitriol-phase6-full.xml

PYTHONPATH=src GRADIO_ANALYTICS_ENABLED=False OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/integration -o addopts='' --timeout=90 -q -ra --tb=short \
  --junitxml=/tmp/vitriol-phase6-integration.xml
```

日志分别为 `/tmp/vitriol-phase6-full.log`、`/tmp/vitriol-phase6-integration.log`。源代码和本轮测试运行 Ruff；CI 范围内的 API/CLI/security 三文件运行 mypy，另检查 git diff --check。

第七轮已增加标准生成入口的同目录互斥，并验证 Linux 独立进程竞争与强退释放，详见[第七轮审计](output-ownership-followup.md)。GPU、机器断电和网络文件系统持久性仍未验证。重命名阶段的目录中可能暂时混合旧名和新名；读取者应等生成成功后使用。学习型策略继续重新生成。完整事务发布和自定义策略状态协议仍未实现；文件系统不提供空洞信息时，校验仍需读取全部逻辑字节。
