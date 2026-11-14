# 第十二轮：心跳与执行超时的计时边界

本轮从 `12e5535` 继续，验证上一轮等待接口之外的两类内部监控：worker 心跳失联判定，以及单次任务执行超时。

## 复现的问题

首批 18 项回归在修改源码前得到 **16 failed、2 passed**，日志 `/tmp/vitriol-phase12-before.log`。其中包括时钟跳变、短超时检查间隔及非法参数行为；通过项是正常失联和刷新后仍在期限内的控制场景。

- 心跳和执行耗时都使用 `time.time()` 相减。系统时钟前跳会提前判定失联/超时；后跳会延迟回收。重新注册节点也不能避免后续时钟跳变引发误判。
- 执行监控固定每 10 秒检查一次，25 毫秒等短超时配置仍需等待长轮询周期。
- 恰好达到执行超时阈值时，原先的严格大于判断不触发失败。
- 零、负数、NaN 和无穷监控参数没有在构造时拒绝，可能导致无效监控或过快轮询。

## 修复

协调器维护两个进程内单调时钟记录表：注册/刷新心跳时记录最近心跳时刻，调度新执行时记录本次开始时刻。监控通过 `time.monotonic()` 计算耗时；对外 `last_heartbeat`、`started_at`、`completed_at` 等字段继续使用 Unix 时间，不改变查询数据格式。

注销节点清理对应心跳记录，重新注册使用新记录。重试、断开后重新分配和停止重启都重新计算执行预算。执行成功、失败、撤销及停止清理旧执行时钟；迟到的旧执行仍受上一轮的归属检查保护，不能删除新执行的记录。停止保留仍注册节点的心跳记录，只有新心跳或重新注册才刷新节点存活时间。

执行监控周期改为 `min(10, task_timeout)`，达到执行期限即触发失败处理。构造参数 `heartbeat_interval`、`task_timeout` 要求有限正数，接受小数秒。此次参数校验是明确的行为变化；无限等待仍可通过上一轮 `wait_for_task(timeout=None)` 或正无穷表达，但不能用无穷配置内部监控周期。

原有通过直接改写对外时间戳模拟过期的测试，改为调整内部单调时钟记录。没有从公开时间戳倒推内部耗时，否则会再次引入时钟跳变问题。

## 验证

新增 **23 项**回归：心跳/执行计时的前跳与后跳、起点为零、心跳刷新、同 ID 节点重新注册、短监控周期、非法配置、重试/断开/重启后的完整新预算，以及终态/停止后的计时清理。

- 分布式专项共 **94 passed**，4.22 秒；日志 `/tmp/vitriol-phase12-focused.log`。
- 全部 `src/vitriol`、新增监控回归和既有生命周期回归的 Ruff 通过；CI 范围 API/CLI/security 三文件的 mypy 通过。
- 本地完整非集成测试：**2975 passed、19 skipped、18 warnings**，78.03 秒；语句覆盖率 **71.94%**（18226 / 25335）。日志 `/tmp/vitriol-phase12-full.log`，JUnit `/tmp/vitriol-phase12-full.xml`，覆盖率 `/tmp/vitriol-phase12-coverage.xml`。跳过项仍为 Hub/模型缓存、可选 fixture、Transformers 支持范围及既有空张量边界。
- 代码与测试已提交并推送：[511d717](https://github.com/isLinXu/Vitriol/commit/511d717)。

对应 `511d717` 的 [GitHub Python CI 43561393647](https://github.com/isLinXu/Vitriol/actions/runs/43561393647) **全部通过**。四个 Python 版本均完成测试、覆盖率门槛与 CLI 检查，随后分发包构建成功。既有非阻断 lint/type 检查策略未改变。

| 远程任务 | 结果 |
|---|---|
| [Python 3.9](https://github.com/isLinXu/Vitriol/actions/runs/43561393647/job/132345644275) | success |
| [Python 3.10](https://github.com/isLinXu/Vitriol/actions/runs/43561393647/job/132345644438) | success |
| [Python 3.11](https://github.com/isLinXu/Vitriol/actions/runs/43561393647/job/132345644413) | success |
| [Python 3.12](https://github.com/isLinXu/Vitriol/actions/runs/43561393647/job/132345644610) | success |
| [Distribution build](https://github.com/isLinXu/Vitriol/actions/runs/43561393647/job/132346568116) | success |

远程结果快照：`/tmp/vitriol-phase12-ci-run.json`、`/tmp/vitriol-phase12-ci-jobs.json`。第二批只同步本报告、总报告和 CHANGELOG，不改变上述已验证代码与测试。

## 边界

监控仍按周期扫描，实际过期处理可能落后期限最多约一个检查周期，且受事件循环调度影响，不是硬实时保证。内部单调时钟值仅用于当前进程，不用于持久化或跨节点比较。异步操作仍限单个事件循环；直接改写内部状态字典不是受支持的注册/调度入口。

分布式执行仍为模拟实现，本轮没有接入或验证真实网络传输，也未重复运行此前的 39 项本地模型集成测试。
