# 第八轮：fork 子进程的目录锁生命周期

本轮继续验证第七轮留下的 fork 继承边界。保留此前修改，没有提交、推送、发布或下载模型权重。环境为 Linux、Python 3.11.18、PyTorch 2.6.0+cpu、Transformers 4.57.6。

## 问题与修复

POSIX flock 关联打开的文件描述，fork 后父子进程共享该对象。原实现存在两个可复现问题：父进程异常退出但子进程仍存活时，继承的描述符继续占锁，阻止新任务恢复；子进程退出继承的锁上下文时，显式 LOCK_UN 会解除父进程仍在使用的锁，允许竞争写入。

`core/output_lock.py` 现在登记所有活动锁描述符，并通过 `os.register_at_fork` 在子进程中关闭继承副本。子进程只 close，不执行 LOCK_UN，因此父进程仍持有互斥锁；父进程退出后也不会由存活子进程继续占用。登记表的线程锁覆盖打开/登记及关闭/注销，在 fork 前取得该锁，避免另一个线程刚打开描述符、尚未登记的窗口。显式模块 reload 保留原登记表，既有描述符仍由原回调管理。

锁上下文记录取得锁时的 PID，仅由该进程执行释放。子进程退出继承上下文时不再操作旧描述符编号，避免关闭已复用于其他文件的编号。子进程仍可正常取得自己的新目录锁并继续 fork。

同时处理释放阶段的双重异常：生成本身已经失败时，显式解锁的 OSError 记录警告，保留原生成异常，并在 finally 中关闭描述符；生成成功但解锁报错时，仍向调用者报告解锁异常。

## 可执行证据

修复前保存源码至 `/tmp/vitriol-phase8-before/output_lock.py`。新增的首批 4 项测试全部失败：单个/多个锁上下文被子进程解除、另一个线程持有的描述符被继承，以及存活子进程继续占用已退出父进程的锁。日志 `/tmp/vitriol-phase8-before.log`。

本轮净新增 **12 项测试：11 项非集成、1 项集成**。fork 用例在独立 Python 解释器内执行，避免直接 fork pytest 的 ML 运行时；Linux 孤儿进程测试使用独立控制进程的 child subreaper 回收子进程。

- 9 项 fork 回归覆盖单目录/多目录、模块 reload、多线程持锁、打开与登记的竞争窗口、父进程退出、子进程描述符复用和再次 fork。
- 2 项异常释放回归确认原异常保留、正常路径的解锁异常传播，以及失败后目录可重新取得锁。
- 真实本地 Llama 生成在 manifest 阶段 fork 工作进程，主生成进程随后以 `os._exit(73)` 强退。工作进程始终存活，新进程恢复成功，禁止重新生成已完成张量，索引和所有权重文件逐字节一致，manifest 完成、检查点清除，模型能重新加载且 forward 输出有限。

专项测试：**31 passed**（含此前目录锁回归），10.64 秒；新增真实模型测试 **1 passed**，20.43 秒。

## 完整验证

| 检查 | 结果 |
|---|---|
| 完整非集成测试 | **2916 passed，19 skipped，19 warnings**，65.00 秒 |
| 完整集成测试 | **39 passed**，165.62 秒 |
| 非集成语句覆盖率 | **71.77%**，18113 / 25237 |
| Ruff：全部 src 与本轮测试 | 通过 |
| CI 范围 mypy：API/CLI/security 三文件 | 通过，保留 untyped-body 提示 |
| git diff --check | 通过 |

19 项跳过原因与上一轮一致：禁用 Hub 下载、缺少本地模型/trace fixture、当前 Transformers 不含 Qwen3.5，以及空张量评分不支持。

```bash
PYTHONPATH=src GRADIO_ANALYTICS_ENABLED=False OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests --ignore=tests/integration --timeout=60 -q -ra --tb=short \
  --cov=src/vitriol --cov-report=term --cov-report=xml:/tmp/vitriol-phase8-coverage.xml \
  --junitxml=/tmp/vitriol-phase8-full.xml

PYTHONPATH=src GRADIO_ANALYTICS_ENABLED=False OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/integration -o addopts='' --timeout=120 -q -ra --tb=short \
  --junitxml=/tmp/vitriol-phase8-integration.xml
```

日志：`/tmp/vitriol-phase8-full.log`、`/tmp/vitriol-phase8-integration.log`。语句覆盖率只统计非集成套件。Ruff 检查全部 src 与本轮三个测试文件；mypy 检查 CI 范围 API/CLI/security 三文件。

## 适用边界

- 实测为 Linux 本地文件系统和 Python `os.fork` 路径；Windows、网络文件系统、GPU、断电持久性及绕过 Python fork 回调的原生扩展路径未实测。
- 子进程关闭继承句柄并不授权其继续执行父进程已进入的生成体。子进程如需写入，应通过新的标准 generate 调用重新取得目录锁；任意复制执行中的生成流程并不在保证范围内。
- 第七轮的配合式互斥边界仍适用：外部绕过协议的写入、替换锁文件、读取中间状态以及同进程不同任务共享随机流的问题不由目录锁解决。
