# 第十六轮：待执行队列完成计数

本轮从 `374a125` 继续检查分布式队列的公开 `asyncio.Queue.join()` 语义。调度器会取走队列条目，但原实现没有在任何路径调用 `task_done()`，因此只要提交过任务，`join()` 就会一直等待。

## 修前复现

新增四项回归测试并在修复前运行：**4 failed**，日志 `/tmp/vitriol-phase16-before.log`。四项分别覆盖正常执行、执行失败后重试、stop 后重启重入队，以及重复的过期队列条目；每项都因 unfinished-task 计数未归零而在 `Queue.join()` 超时。

## 修复

执行条目派发后继续保留其 unfinished-task 计数，直到对应执行 task 真正结束，再由完成回调调用 `task_done()`。失败重试和 stop 重入队会先为新队列条目增加计数，旧执行完成时只结清旧条目；因此 `join()` 在运行中或重试仍待执行时不会提前返回。若调度器取到的条目已不再是 pending，则立即结清这条过期的队列记录。派发事务回滚仍把同一条目放回队首，不重复改动计数。

新增 `tests/test_distributed_queue_accounting.py`，以事件同步执行状态，验证 `join()` 在执行期间仍等待，并在成功、重试、stop/restart 与重复条目场景中最终完成。

## 验证

- 修前：新增队列回归 **4 failed**；修后新增队列回归 **4 passed**。
- 分布式定向套件：**120 passed**，9.59 秒。
- 全量非集成套件：**3001 passed、19 skipped、18 warnings**，74.88 秒；覆盖率 **72.00%**（25,408 条语句），超过 CI 的 60% 门槛。跳过项仍因 Hub/model 缓存或下载、HF 配置、空 tensor 评分、当前 Transformers 不含 Qwen3.5、可选 trace fixture 等环境边界；warning 未造成失败。
- 全 `src/` 与新增回归文件 Ruff、CI 指定 API/CLI/security 三个入口文件 MyPy 及 `git diff --check` 通过。完整测试日志 `/tmp/vitriol-phase16-full.log`，JUnit `/tmp/vitriol-phase16-full.xml`，Coverage XML `/tmp/vitriol-phase16-coverage.xml`。
- 代码与回归先提交为 [`c94dde0`](https://github.com/isLinXu/Vitriol/commit/c94dde0)。[Python CI run 43612034226](https://github.com/isLinXu/Vitriol/actions/runs/43612034226) 全部成功： [Python 3.9](https://github.com/isLinXu/Vitriol/actions/runs/43612034226/job/132498950401)、[Python 3.10](https://github.com/isLinXu/Vitriol/actions/runs/43612034226/job/132498950165)、[Python 3.11](https://github.com/isLinXu/Vitriol/actions/runs/43612034226/job/132498950497)、[Python 3.12](https://github.com/isLinXu/Vitriol/actions/runs/43612034226/job/132498950460)、[分发包构建](https://github.com/isLinXu/Vitriol/actions/runs/43612034226/job/132500141672)。

## 范围与限制

`pending_tasks.join()` 只跟踪队列条目及其对应执行尝试的完成，不替代 `wait_for_task()` 的业务终态查询。执行仍是本地模拟；本轮没有验证真实网络 worker、跨进程恢复或长时压力。
