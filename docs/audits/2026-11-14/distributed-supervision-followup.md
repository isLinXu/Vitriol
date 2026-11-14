# 第十四轮：分布式后台循环监督

本轮从 `db3b426` 继续，检查协调器和 worker 的后台任务遇到运行时异常后是否还能工作。

## 修前复现

先新增 6 项失败注入测试，再运行原实现：**5 failed、1 passed**，日志 `/tmp/vitriol-phase14-before.log`。

协调器心跳监控、任务超时监控及 worker 心跳/任务处理循环一旦抛出异常就结束，后台任务虽然已经死亡，拥有者仍保持 `running=True`。worker 心跳循环内部捕获异常后立即回到下一轮，没有退避。后台循环若意外正常返回也没有重新创建。唯一修前通过项验证了 stop 在退避期仍应迅速完成。

## 修复

新增共享后台循环监督器，应用于协调器的心跳监控、调度器和超时监控，以及 worker 的心跳和任务处理循环。异常会附带 traceback 记录；运行期间意外正常返回会记录 warning。两种情况都按 1、2、4 秒递增，最高 30 秒后重启；单次运行稳定至少 60 秒后退避恢复至 1 秒。stop 仍取消并等待监督任务，取消异常不会被误认为普通故障。worker 心跳循环现在向监督器传播错误，避免内部无延迟重试。

新增测试覆盖协调器两类监控、worker 两类循环失败后重启，监督器退避期间的停止行为，以及运行中循环意外返回后的重启。所有用例都用事件同步恢复状态并在 finally 清理后台任务。

## 验证

- 修前：**5 failed、1 passed**；修后新增监督回归 **6 passed**，约 5.08 秒。
- 分布式定向套件：**114 passed**，9.01 秒，包含协调器、事件循环、生命周期、任务 API、watchdog、调度和监督回归。
- 全量非集成套件：**2995 passed、19 skipped、18 warnings**，80.92 秒；覆盖率 **71.98%**（25,374 条语句），超过 CI 的 60% 门槛。
- 全 `src/` 与新增测试 Ruff 通过；CI 同款三个入口文件 MyPy 通过；`git diff --check` 通过。
- 代码与测试提交：[3ba3d9d](https://github.com/isLinXu/Vitriol/commit/3ba3d9d391ca85da88fc755328c1199a761bfcfc)。[Python CI run 43592539958](https://github.com/isLinXu/Vitriol/actions/runs/43592539958) 全部成功： [Python 3.9](https://github.com/isLinXu/Vitriol/actions/runs/43592539958/job/132440146555)、[Python 3.10](https://github.com/isLinXu/Vitriol/actions/runs/43592539958/job/132440146652)、[Python 3.11](https://github.com/isLinXu/Vitriol/actions/runs/43592539958/job/132440146427)、[Python 3.12](https://github.com/isLinXu/Vitriol/actions/runs/43592539958/job/132440146708)、[分发包构建](https://github.com/isLinXu/Vitriol/actions/runs/43592539958/job/132441077252)。

本轮跳过 19 项的类别：Hub smoke 12 项、adapter Hub 模型 1 项、HF 配置加载 1 项、空 tensor 评分 1 项、TinyLlama 缓存缺失 1 项、当前 Transformers 版本缺少 Qwen3.5 模型类 1 项、可选 trace fixture 2 项。18 条测试 warning 均未造成失败。Mypy 只覆盖 CI 指定三个入口文件。

## 范围与限制

分布式生成仍是模拟实现，没有真实网络传输。调度器原有内部派发异常捕获会等待固定 1 秒后继续；共享监督退避作用于逸出循环的异常及意外返回，不替代这一局部恢复逻辑。本轮未覆盖远端节点、跨进程恢复或长时间运行压力。
