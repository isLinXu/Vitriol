# 第十七轮：分布式重试预算语义

本轮从 `e4d2e6c` 继续审计任务失败路径，重点核对公开参数 `max_retries` 与 `GenerationTask.retry_count` 的契约。

## 修前复现

新增/调整的 6 项回归在原实现上运行，结果为 **5 failed、1 passed**，日志 `/tmp/vitriol-phase17-before.log`。`max_retries=1` 第一次执行失败就进入终态，完全没有重试；`max_retries=0` 却把失败次数记为一次 retry；负数、浮点数和布尔值也都被构造器接受。原行为实际上把 `max_retries` 当作“总尝试数”，并让 `retry_count` 统计失败次数。

## 修复

现在 `max_retries` 表示初始执行之外允许安排的重试次数。失败后只有在预算内实际重排任务时才增加 `retry_count`；终态失败不增加该字段。`max_retries=0` 表示只执行一次，`max_retries=1` 最多执行两次。worker 的 `failed_tasks` 仍逐次记录失败的执行尝试。

构造器拒绝负数、非整数和布尔值，避免参数在比较时产生隐式或误导的尝试上限。这是有意的 API 行为修正：若调用方此前把 `max_retries` 当作最大总尝试数传入，应将其值减一，或按允许的重试次数重新配置。

## 验证

- 修前：预算与配置回归 **5 failed、1 passed**；修后对应回归 **6 passed**。
- 分布式定向套件：**125 passed**，9.88 秒；覆盖零预算、一次/多次预算、成功恢复、预算耗尽、超时终止与无效配置。
- 全量非集成套件：**3006 passed、19 skipped、18 warnings**，80.33 秒；覆盖率 **72.00%**（25,410 条语句），通过 60% 门槛。跳过项仍由 Hub/model 下载或缓存、HF 配置、空 tensor、当前 Transformers 不含 Qwen3.5 类及可选 trace fixture 等环境边界造成。
- 全 `src/` 与本轮修改测试 Ruff、CI 指定 API/CLI/security 三文件 MyPy、`git diff --check` 均通过。完整日志 `/tmp/vitriol-phase17-full.log`，JUnit `/tmp/vitriol-phase17-full.xml`，Coverage XML `/tmp/vitriol-phase17-coverage.xml`。
- 代码与测试已先提交为 [`b364f01`](https://github.com/isLinXu/Vitriol/commit/b364f01)。[Python CI run 43617805682](https://github.com/isLinXu/Vitriol/actions/runs/43617805682) 全部成功：[Python 3.9](https://github.com/isLinXu/Vitriol/actions/runs/43617805682/job/132516591273)、[Python 3.10](https://github.com/isLinXu/Vitriol/actions/runs/43617805682/job/132516591315)、[Python 3.11](https://github.com/isLinXu/Vitriol/actions/runs/43617805682/job/132516590980)、[Python 3.12](https://github.com/isLinXu/Vitriol/actions/runs/43617805682/job/132516591226)、[分发包构建](https://github.com/isLinXu/Vitriol/actions/runs/43617805682/job/132517817449)。

## 范围与限制

worker 断连后的重分配与主动 stop 的恢复不计入执行失败重试预算；超时仍走任务失败路径并消耗预算。执行仍是本地模拟，本轮没有验证真实网络 worker 或跨进程恢复。
