# 第七轮：输出目录互斥与失败清理

本轮继续处理前六轮记录的同目录并发写入问题。保留此前改动，没有提交、推送、发布或下载模型权重。验证环境为 Linux、Python 3.11.18、PyTorch 2.6.0+cpu、Transformers 4.57.6。

## 问题与修复

原生成入口没有输出目录互斥。两个线程、两个生成器实例或两个独立进程使用同一目录时，后进入的任务可以读取未完成的检查点、清理对方的权重或覆盖产物。API 内部队列仅保护同一服务进程，不能覆盖 CLI 或其他进程。

新增 `core/output_lock.py`，在标准 `MinimalWeightGenerator.generate()` 入口取得输出目录的非阻塞操作系统锁，覆盖 bootstrap、模型初始化、分片、索引、元数据和结果构建。冲突在读取 HF 配置及修改产物前抛出 `OutputDirectoryBusyError`，其 recoverable 为 true；调用者可在原任务完成后重试或使用其他输出目录。同一生成器递归调用也拒绝进入。

目录按真实路径定位；相对路径和目录符号链接不会形成另一把锁。POSIX 使用 flock，Windows 分支使用 msvcrt 字节锁，未引入第三方依赖。普通 I/O 或不支持锁的错误直接传播，不退化成无法可靠释放的 PID/标记文件锁。

锁文件 `.vitriol-generation.lock` 保留原 inode，不在退出时删除或截断；否则等待者可能锁住旧 inode，而新进程锁住新建 inode，互斥就会失效。占用状态由 OS 锁决定，标记文件存在不代表目录仍忙。普通异常、KeyboardInterrupt、SystemExit 都释放锁，文件描述符在 finally 中关闭。锁路径是符号链接或非普通文件时拒绝使用。

## 失败后清理

进程强退可能绕过 TemporaryDirectory/NamedTemporaryFile 清理。现在只有成功取得目录锁的任务才清理生成器保留命名空间中符合 tempfile 命名规则的遗留项：

- `.vitriol-shard-<8 位临时后缀>`、`.vitriol-index-<8 位临时后缀>` 目录；
- `.vitriol-checkpoint-<8 位临时后缀>` 普通文件。

清理不会跟随这些路径的符号链接，也不删除正式检查点、已提交权重、索引、锁标记或其他命名的目录。竞争失败的任务不会执行清理，因而不能删掉正在写入的临时文件。清理失败记录警告，仍正常释放目录锁。

## 可执行证据

新增测试前保存源码至 `/tmp/vitriol-phase7-before/`。先运行 7 项竞争回归，全部失败：配置加载、模型初始化、分片写出、配置保存、manifest 五个阶段的竞争，以及线程与目录符号链接竞争。日志 `/tmp/vitriol-phase7-before.log`。

本轮净新增 **22 项测试：20 项非集成、2 项集成**。

非集成覆盖完整生成期间的冲突、线程快速失败、同对象重入、相对/符号链接路径、不同目录独立执行、三种异常退出、锁标记不截断、不跟随锁路径符号链接、锁 I/O 失败关闭描述符，以及遗留文件清理/保留/失败行为。

真实跨进程测试使用本地小型 Llama：原进程生成权重后暂停在 manifest 阶段，保留一份活动临时目录；第二个进程通过输出目录符号链接发起生成，必须收到可重试的忙错误，目录内全部文件字节保持一致。随后分别验证正常退出和 SIGKILL 强制终止；强退后由新进程恢复，禁止重新生成已提交权重，清理遗留临时目录并完成 manifest。前六轮的进程强退、逐张量对照及模型加载/forward 测试同时保留。

最终验证结果：

| 检查 | 结果 |
|---|---|
| 完整非集成测试 | **2905 passed，19 skipped，19 warnings**，60.42 秒 |
| 完整集成测试 | **38 passed**，155.84 秒 |
| 非集成语句覆盖率 | **71.75%**，18078 / 25197 |
| Ruff：全部 src 与本轮测试 | 通过 |
| CI 范围 mypy：API/CLI/security 三文件 | 通过，保留 untyped-body 提示 |
| git diff --check | 通过 |

19 项跳过原因与上一轮一致：禁用 Hub 下载、缺少本地模型/trace fixture、当前 Transformers 不含 Qwen3.5，以及空张量评分不支持。

## 验证命令

```bash
PYTHONPATH=src GRADIO_ANALYTICS_ENABLED=False OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests --ignore=tests/integration --timeout=60 -q -ra --tb=short \
  --cov=src/vitriol --cov-report=term --cov-report=xml:/tmp/vitriol-phase7-coverage.xml \
  --junitxml=/tmp/vitriol-phase7-full.xml

PYTHONPATH=src GRADIO_ANALYTICS_ENABLED=False OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/integration -o addopts='' --timeout=120 -q -ra --tb=short \
  --junitxml=/tmp/vitriol-phase7-integration.xml
```

日志为 `/tmp/vitriol-phase7-full.log` 和 `/tmp/vitriol-phase7-integration.log`。全部 src 与本轮测试文件运行 Ruff；CI 范围 API/CLI/security 三文件运行 mypy，另检查 git diff --check。

## 适用边界

- 已实测 Linux 本地文件系统。Windows 分支、网络文件系统锁语义、GPU 与断电持久性没有实机验证。
- 这是配合式互斥，保护使用标准 generate 入口的写入者；不阻止旧版程序、外部工具、私有写文件方法或可视化代码绕过协议。运行期间不可由外部删除/替换锁文件。
- 不提供整个输出目录的原子发布，也不阻止读者观察生成中的中间状态。
- 本轮跨进程验证使用独立启动的 Python 进程，当时未验证存活 fork 子进程对锁生命周期的影响；该问题已在[第八轮](fork-ownership-followup.md)复现、修复并补充验证。
- 不同输出目录允许并行；同进程多个任务仍共享默认随机流，若要求随机结果可复现，应继续使用串行调用或独立进程。
- 遗留清理只针对保留的临时文件命名空间；未知命名的旧文件不会主动删除。
