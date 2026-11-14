# 第十一轮：任务等待期限与状态快照

本轮从 `47ab6fe` 继续检查分布式任务的公共接口，重点是等待超时、等待者取消、无节点时的排队，以及提交/查询对象与内部任务的关系。

## 复现与修复

第一批 16 项回归在修改源码前得到 **10 failed、6 passed**，记录于 `/tmp/vitriol-phase11-before.log`。确认以下问题：

- `timeout=0` 被真值判断忽略，实际成为无限等待；NaN 也无法使期限判断成立。
- 每次固定休眠 100 毫秒，即使只剩 25 毫秒预算也继续睡满 100 毫秒。
- 使用系统墙上时钟计算等待耗时，时钟向前跳会提前返回，向后跳会延长等待。
- 提交的分片列表被直接保存在任务中，调用方之后修改原列表会改变队列内任务。
- 状态查询返回内部列表与结果字典的引用，修改查询结果会改变内部任务；内部后续修改也会改变此前取得的查询结果。

`wait_for_task` 改用 `time.monotonic()` 计算已用时间，并将每次休眠限制为轮询周期和剩余预算中的较小值。非正超时立即检查当前状态；NaN 抛出 `ValueError`；`None` 和正无穷保留无限等待行为。已完成/失败任务仍直接返回，未知任务仍返回 `None`。事件循环调度本身可能延迟唤醒，因此这不是硬实时期限保证。

提交时复制分片列表，查询状态时复制分片列表并深拷贝结果字典，形成独立快照。`wait_for_task` 继续返回原有的终态 `GenerationTask` 对象，不改变其身份语义；需要独立快照应使用 `get_task_status`。直接访问内部 `tasks` 字典仍可修改任务，不将本轮修复描述为全面不可变模型。

## 验证

新增 **20 项**回归，包括确定性时钟前跳/后跳、短期限、零/负/NaN/无限超时、终态与未知任务、提交列表隔离、嵌套查询结果双向隔离，以及等待者取消/超时与任务执行相互独立。

无节点时取消等待者，任务保持排队；之后注册节点可以完成任务。运行中取消一个等待者或让它超时，另一等待者仍能取得成功结果。实际执行抛错的场景覆盖一次失败即终止和失败后重试成功，核对尝试次数、错误清理、完成时间及任务/worker 计数；这些是既有行为的保护性验证，不列为新修复缺陷。

- 协调器、事件循环、生命周期和公共接口专项：**71 passed**，3.92 秒；日志 `/tmp/vitriol-phase11-focused.log`。
- 全部 `src/vitriol` 与新增测试文件的 Ruff 通过；CI 范围 API/CLI/security 三文件的 mypy 通过。
- 本地完整非集成套件：**2952 passed、19 skipped、18 warnings**，73.64 秒；语句覆盖率 **71.92%**（18211 / 25321）。日志 `/tmp/vitriol-phase11-full.log`，JUnit `/tmp/vitriol-phase11-full.xml`，覆盖率 `/tmp/vitriol-phase11-coverage.xml`。跳过项仍为 Hub/模型缓存、可选 fixture、Transformers 支持范围及原有空张量边界。
- 源码与 20 项回归已提交并推送：[8ed7308](https://github.com/isLinXu/Vitriol/commit/8ed7308)。

对应 `8ed7308` 的 [GitHub Python CI 43558540909](https://github.com/isLinXu/Vitriol/actions/runs/43558540909) **全部通过**。四个 Python 版本均完成测试、覆盖率门槛与 CLI 检查，随后分发包构建成功。既有非阻断 lint/type 检查策略未改变。

| 远程任务 | 结果 |
|---|---|
| [Python 3.9](https://github.com/isLinXu/Vitriol/actions/runs/43558540909/job/132336808512) | success |
| [Python 3.10](https://github.com/isLinXu/Vitriol/actions/runs/43558540909/job/132336808532) | success |
| [Python 3.11](https://github.com/isLinXu/Vitriol/actions/runs/43558540909/job/132336808347) | success |
| [Python 3.12](https://github.com/isLinXu/Vitriol/actions/runs/43558540909/job/132336808503) | success |
| [Distribution build](https://github.com/isLinXu/Vitriol/actions/runs/43558540909/job/132337936361) | success |

远程结果快照：`/tmp/vitriol-phase11-ci-run.json`、`/tmp/vitriol-phase11-ci-jobs.json`。第二批只同步本报告、总报告和 CHANGELOG，不改变上述已验证代码与测试。

## 范围

本轮不改变执行重试上限、调度轮询周期、网络传输或模型生成逻辑。分布式模块仍为模拟执行，异步操作仍要求单个事件循环。未重跑第八轮的 39 项本地模型集成测试，不将此前结果作为本轮新增验证。
