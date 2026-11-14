# 第十轮：分布式协调器生命周期与任务重试

本轮从 `a921bdd` 继续，检查 `DistributedCoordinator` 和 `WorkerNode` 的后台协程、停止与重启、超时重试及节点断开。模块目前仍模拟执行，未接入真实网络传输；本轮验证限于单个事件循环中的调度和状态管理。

## 已确认的问题

- `start()` 每次都创建后台循环，重复调用会产生多个调度器和监控器。
- `stop()` 只修改 `running`，没有取消或等待后台循环和执行协程。旧协程可能在停止后完成任务，快速重启也会让旧循环继续运行。
- 超时重试和节点断开只重排任务，没有撤销旧执行。旧结果可能覆盖重新分配后的状态，重复增加完成计数，或提前释放正在处理新任务的 worker。
- 超时监控通过时间戳的真值判断是否开始，漏掉 `started_at=0` 的边界。

第一批 6 项新增回归在修改源码前得到 **5 failed、1 passed**；日志 `/tmp/vitriol-phase10-before.log`。其中重复队列用例在原实现中恰好通过，因此不将该项描述为已经复现的旧缺陷。该次命令还触发了全项目覆盖率门槛；上述失败数来自用例断言，后续专项使用 `--no-cov`，覆盖率由完整套件检查。

## 修复行为

协调器与 worker 保存后台任务句柄，重复启动直接返回。停止由共享清理协程取消并等待所有自有协程；并发停止复用清理过程，取消某个等待停止的调用者也不会取消清理。重启等待清理结束，再创建新一组后台循环。

协调器停止时将运行中的任务重新入队并释放 worker，不增加失败或重试计数。队列消费者直接等待 `Queue.get()`，通过停止取消解除等待，避免额外 `wait_for` 消费协程在取消交界处取走任务。调度器过滤非 pending 条目，防止陈旧条目再次派发。

每个任务只保留一个有效执行句柄。断开、超时和停止会先撤销执行归属，再取消旧执行；旧执行即使在取消后返回结果或异常，也不能更新新执行的状态、计数或 worker。已撤销但尚未结束的执行仍被跟踪，停止会等待它们退出。正常重试清理旧结果与错误；最终失败记录完成时间。保留既有 `max_retries` 判断语义。

原有直接调用私有执行与失败处理函数的测试补齐运行状态及分配关系，使测试条件与实际调度一致。同步构造和事件循环启动前的任务重分配继续受上一轮 4 项回归保护。

## 验证范围

新增 **12 项**回归覆盖重复启动、并发停止、运行中停止后恢复、断开/超时后迟到的成功或失败、重复队列条目、停止等待者被取消、清理期间重启、消费者唤醒与停止竞态，以及超时达到重试上限。

- 协调器、事件循环和新生命周期专项：**51 passed**。
- Python 3.9.25、3.12.3 独立标准库检查：重复启动、停止回收、重启与任务恢复均通过；另外注入实际执行异常，验证重试成功及耗尽上限时的状态与计数。脚本 `/tmp/vitriol-phase10-loop-smoke.py`。这不是两个解释器上的完整 ML 套件。
- 全部 `src/vitriol`、新回归及上一轮事件循环回归的 Ruff 通过；CI 范围 API/CLI/security 三文件的 mypy 通过。
- 本地完整非集成套件：**2932 passed、19 skipped、18 warnings**，79.91 秒；语句覆盖率 **71.92%**（18204 / 25313）。日志 `/tmp/vitriol-phase10-full.log`，JUnit `/tmp/vitriol-phase10-full.xml`，覆盖率 `/tmp/vitriol-phase10-coverage.xml`。跳过项仍是 Hub/模型缓存、可选 fixture、Transformers 支持范围及原有空张量边界。
- 源码与测试已提交并推送：[fab1d81](https://github.com/isLinXu/Vitriol/commit/fab1d81)。

对应 `fab1d81` 的 [GitHub Python CI 43556899686](https://github.com/isLinXu/Vitriol/actions/runs/43556899686) **全部通过**。四个 Python 版本均完成测试、覆盖率门槛与 CLI 检查；随后分发包构建成功。既有非阻断 lint/type 检查策略未改变。

| 远程任务 | 结果 |
|---|---|
| [Python 3.9](https://github.com/isLinXu/Vitriol/actions/runs/43556899686/job/132331708847) | success |
| [Python 3.10](https://github.com/isLinXu/Vitriol/actions/runs/43556899686/job/132331709057) | success |
| [Python 3.11](https://github.com/isLinXu/Vitriol/actions/runs/43556899686/job/132331709033) | success |
| [Python 3.12](https://github.com/isLinXu/Vitriol/actions/runs/43556899686/job/132331709053) | success |
| [Distribution build](https://github.com/isLinXu/Vitriol/actions/runs/43556899686/job/132332940038) | success |

远程结果快照：`/tmp/vitriol-phase10-ci-run.json`、`/tmp/vitriol-phase10-ci-jobs.json`。第二批仅提交本报告、总报告及 CHANGELOG，不改变上述已验证代码与测试。

## 适用边界

停止依赖协程响应取消；若将来接入不能取消的外部操作，需要传输层自己的中断与幂等协议。本轮不保证远程计算已终止，也不构成真实多机生成验证。同一个协调器/worker 的异步操作及重启须使用同一事件循环，不承诺跨线程或跨循环复用。

未修改模型生成、KV 算法或发布流程，未重复运行此前第八轮的 39 项模型集成测试。

## 后续更正（第十七轮）

本轮曾选择保留旧的 `max_retries` 判断方式。后续审计发现它把该参数当作总尝试次数，`max_retries=1` 实际不进行任何重试，与参数名及重试计数契约不符。第十七轮已改为将其解释为初次执行之后的重试次数，并让 `retry_count` 只统计已安排的重试；详见[第十七轮审计](distributed-retry-budget-followup.md)。
