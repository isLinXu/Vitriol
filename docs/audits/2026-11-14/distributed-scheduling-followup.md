# 第十三轮：调度唤醒与排队顺序

本轮从 `f0e6a77` 继续，检查无节点、节点忙碌、节点恢复及执行结束时的调度行为。

## 复现的问题

首批 9 项回归在修改源码前得到 **6 failed、3 passed**，日志 `/tmp/vitriol-phase13-before.log`。

原调度器先取出队首任务，再选择空闲节点。如果没有节点，会将该任务放到队尾并休眠 1 秒。其结果是没有容量时也会改变排队顺序，并且节点刚注册、执行刚完成或失败后释放容量，都要等旧轮询休眠结束才能继续派发。

失败用例分别覆盖无节点时的队列顺序、忙碌节点后的派发顺序、注册/成功/失败三种容量变化的延迟，以及无节点时持续轮询。3 个通过项验证原有提交、`Queue.put` 与 `put_nowait` 的入队唤醒行为，作为兼容性约束保留。

## 修复

调度器改为等待事件通知。新任务入队、节点注册和执行成功/失败释放容量都会通知调度器。每次先清除通知，再同步检查容量、取出任务并分配；这些检查与分配之间不让出事件循环，避免在检查与等待之间遗漏同一循环上的通知。

没有空闲节点时不取出任务，因此保留现有 FIFO 队列顺序。重试、断开后的重新分配以及停止后恢复仍按既有规则放入队尾，已有待执行任务可先获得容量；不承诺重试任务仍保留最初提交位置，也不承诺多节点完成顺序与派发顺序一致。

待执行队列使用一个内部 `asyncio.Queue` 子类，在 `put_nowait` 成功后发出通知；继承的异步 `put` 同样经过此入口。因此直接使用现有队列的两种入队方式也能唤醒调度器。事件和队列都在实际执行循环中延迟创建，保留 Python 3.9 同步构造兼容性。

停止仍取消并等待后台协程，不需要哨兵任务或定时唤醒。调度异常保留日志并退避 1 秒，避免异常循环占满事件循环；正常等待任务或容量不再定时轮询。

## 验证

新增 **14 项**回归：FIFO 排队与派发、三种容量变化、三个入队入口、空闲时不轮询、任务/容量按不同顺序到达、重试/断开后的队尾重分配，以及关闭旧循环后同步初始化再异步启动。

- 分布式专项 **108 passed**，3.94 秒；日志 `/tmp/vitriol-phase13-focused.log`。
- 5 项独立事件循环脚本分别在实际 Python 3.9 和 Python 3.12 执行，全部通过；这些标准库级检查不冒充两套完整 ML 环境测试。
- 全部 `src/vitriol`、新增调度测试和事件循环测试的 Ruff 通过；CI 范围 API/CLI/security 三文件的 mypy 通过。
- 本地完整非集成测试 **2989 passed、19 skipped、18 warnings**，75.64 秒；语句覆盖率 **71.99%**（18252 / 25355）。日志 `/tmp/vitriol-phase13-full.log`，JUnit `/tmp/vitriol-phase13-full.xml`，覆盖率 `/tmp/vitriol-phase13-coverage.xml`。跳过项仍为 Hub/模型缓存、可选 fixture、Transformers 支持范围及既有空张量边界。
- 代码与测试已提交并推送：[85a04c2](https://github.com/isLinXu/Vitriol/commit/85a04c2)。

对应 `85a04c2` 的 [GitHub Python CI 43566009980](https://github.com/isLinXu/Vitriol/actions/runs/43566009980) **全部通过**。四个 Python 版本均完成测试、覆盖率门槛与 CLI 检查，随后分发包构建成功。既有非阻断 lint/type 检查策略未改变。

| 远程任务 | 结果 |
|---|---|
| [Python 3.9](https://github.com/isLinXu/Vitriol/actions/runs/43566009980/job/132359904057) | success |
| [Python 3.10](https://github.com/isLinXu/Vitriol/actions/runs/43566009980/job/132359903859) | success |
| [Python 3.11](https://github.com/isLinXu/Vitriol/actions/runs/43566009980/job/132359904038) | success |
| [Python 3.12](https://github.com/isLinXu/Vitriol/actions/runs/43566009980/job/132359904092) | success |
| [Distribution build](https://github.com/isLinXu/Vitriol/actions/runs/43566009980/job/132360818558) | success |

远程结果快照：`/tmp/vitriol-phase13-ci-run.json`、`/tmp/vitriol-phase13-ci-jobs.json`。第二批只同步本报告、总报告和 CHANGELOG，不改变上述已验证代码与测试。

## 范围

本轮没有做吞吐量或 CPU 百分比基准，不将取消轮询转换为未经测量的性能数字。单个协调器仍要求在同一事件循环使用，不支持跨线程直接操作队列；仍为模拟执行，没有新增真实网络传输。

原生命周期、任务等待、重试与监控回归全部纳入专项。未重复运行此前的 39 项本地模型集成测试。
