# 第十五轮：调度派发失败回滚

本轮从 `a97fd30` 继续，检查第十四轮提到的调度器内部派发异常捕获，重点验证异常发生在任务出队和标记运行之后时，任务与 worker 是否能恢复。

## 修前复现

新增两项故障注入后运行原实现：**2 failed**，日志 `/tmp/vitriol-phase15-before.log`。

第一项在创建执行任务时注入异常。此时队列头任务已被取出并标记 `running`，worker 已标记 `BUSY`；原 `except` 只记录短错误文本并等待一秒，因此 task 永远不会完成，也不会重新出现在待执行队列。第二项令 `_select_worker()` 持续抛错；内部 `except` 以固定一秒间隔重试，屏蔽了第十四轮加入的后台监督和指数退避。

## 修复

把调度循环的异常交给共享监督器，获得统一的 traceback 日志和 1、2、4 秒递增、最高 30 秒的退避。异常发生在 dispatch 准备事务时，快照保留 task 的状态、worker 的可用状态和任务元数据；回滚会撤销可能建立的执行映射、释放单调启动时间、恢复双方快照，并把尚未启动的任务恢复到 FIFO 队首。该失败不作为一次模型任务尝试，因此不增加 `retry_count` 或任务失败统计。

`_PendingTaskQueue.put_front_nowait()` 仅恢复刚刚出队且尚未完成的项目，因此不会重复增加 Queue 的 unfinished-task 计数；它会唤醒可能的 getter 和调度器。执行任务创建失败时关闭尚未调度的 coroutine，任务登记失败时取消已创建的 task，避免遗留无主执行。

## 验证

- 修前：2 项新增故障注入 **2 failed**；第一个断言观察到 `running` 而不是 `pending`，第二项在 1.5 秒内只触发两次固定间隔重试。
- 修后新增测试：**2 passed**；验证 worker/task 状态完整回滚、FIFO 队首恢复、业务重试次数不增加、持久 scheduler 异常按指数退避且 stop 可中断等待。
- 分布式定向套件：**116 passed**，9.54 秒，包含生命周期、event-loop、API、watchdog、调度、监督相关测试。
- 全量非集成套件：**2997 passed、19 skipped、18 warnings**，82.51 秒；覆盖率 **71.99%**（25,374 条语句），通过 CI 的 60% 门槛。
- 全 `src/` 与调度回归 Ruff 通过；CI 指定 API/CLI/security 三个入口文件 MyPy 通过；`git diff --check` 通过。修前失败由本轮故障注入运行直接复现，修后全量日志 `/tmp/vitriol-phase15-full.log`，JUnit `/tmp/vitriol-phase15-full.xml`，Coverage XML `/tmp/vitriol-phase15-coverage.xml`。
- 代码与测试提交：[95d964c](https://github.com/isLinXu/Vitriol/commit/95d964c)。[Python CI run 43595133361](https://github.com/isLinXu/Vitriol/actions/runs/43595133361) 全部成功：[Python 3.9](https://github.com/isLinXu/Vitriol/actions/runs/43595133361/job/132447911870)、[Python 3.10](https://github.com/isLinXu/Vitriol/actions/runs/43595133361/job/132447912015)、[Python 3.11](https://github.com/isLinXu/Vitriol/actions/runs/43595133361/job/132447912081)、[Python 3.12](https://github.com/isLinXu/Vitriol/actions/runs/43595133361/job/132447912286)、[分发包构建](https://github.com/isLinXu/Vitriol/actions/runs/43595133361/job/132449013889)。

19 个跳过项仍来自 Hub/model 下载或缓存缺失、HF 配置不可用、空 tensor 评分边界、当前 Transformers 版本缺少 Qwen3.5 类、可选 trace fixture 缺失。18 条 warning 未造成失败。MyPy 检查范围仅限 CI 指定的三个入口文件。

## 范围与限制

队首恢复依赖 asyncio.Queue 当前使用的内部 deque 和 getter 唤醒方法；该实现由项目 Python 3.9–3.12 CI 矩阵验证。分布式任务执行仍为模拟，不涵盖真实网络节点、进程故障恢复或长时间压力。
