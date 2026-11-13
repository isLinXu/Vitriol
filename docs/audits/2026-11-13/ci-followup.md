# 第九轮：GitHub CI 跨版本验证与构建

本轮从前八轮已上传的 `3176a54` 继续，检查 GitHub Actions 的实际执行结果，并按用户要求分批提交、推送后续修复。

## 已确认的问题

上传后的 [Python CI 43439502691](https://github.com/isLinXu/Vitriol/actions/runs/43439502691) 中，Python 3.10 和 3.11 的测试与 CLI 检查通过，3.9 在安装解释器时失败，3.12 在测试阶段失败。文档提交仅触发 Pages 工作流，因此不能把 Pages 成功视作 Python CI 通过。

Python 3.9 的 GitHub 注释明确指出：`ubuntu-latest` 已迁移到 Ubuntu 26.04，setup-python 没有为该系统提供 3.9 x64 构建。官方 python-versions 清单列出了 Python 3.9.25 在 Ubuntu 22.04 和 24.04 的构建。因此将测试矩阵固定到 `ubuntu-24.04`，保留 3.9、3.10、3.11、3.12 四个版本；没有通过删除旧版本检查回避兼容性问题。

在只安装 pip/build 的独立 Python 3.11 环境复现了另一项构建隐患：`python -m build --no-isolation` 因缺少 setuptools 后端失败。改用 `python -m build`，由隔离构建安装 pyproject 声明的依赖，源码包和 wheel 构建成功。检查了新增锁与检查点模块、可视化资源及源码包内回归测试的存在。

## CI 诊断改进

公开 GitHub API 可以读取工作流结果、步骤和检查注释，但此次会话没有下载完整作业日志的 API 凭据。为让后续失败直接可定位，测试命令新增 JUnit XML；后续步骤将失败和收集错误写入 GitHub 注释，并保留 JUnit/coverage 报告七天。覆盖率门槛和测试范围保持不变。

对报告步骤进行了实际执行验证，覆盖失败用例、收集错误、百分号/换行转义和报告缺失情况。

## 跨版本失败与修复

新增诊断步骤在 [CI 43476256705](https://github.com/isLinXu/Vitriol/actions/runs/43476256705) 揭示了具体失败：

- Python 3.12：`TestEvolutionTreeComputeSimilarity.test_identical` 将余弦相似度与 1.0 精确比较，实际结果为 `0.9999999999999999`。算法已将结果限制在 [0, 1]，因此只将该断言改为绝对误差 `1e-15` 的比较，不改变相似度算法或其他测试门槛。
- Python 3.9：`DistributedCoordinator.__init__` 创建 `asyncio.Queue` 时会立即查找事件循环；之前执行过 `asyncio.run` 后没有当前循环，多个同步初始化用例报错。在未运行任务前创建的队列，也可能绑定到与后续执行不同的循环。

协调器现在在首次异步访问时创建队列，同步初始化、worker 登记和统计不创建或安装事件循环。同步注销 worker 时直接保留待重分配任务；队列创建后按顺序转入，已存在的无界队列使用 `put_nowait`，不再创建可能未执行的后台 put 协程。直接访问 `pending_tasks` 初始化队列须处于协调器的执行循环；统计信息可同步读取。

新增 4 项回归覆盖：关闭循环后的同步构造、不改变全局循环、等待中的消费者绑定执行循环、异步执行前的任务重分配，以及已初始化队列上的立即重分配。修复前在 Python 3.11 为 **3 failed、1 passed**，并在独立 Python 3.9 中实际复现构造异常。修复后专项套件 **107 passed**；将这 4 项独立子进程场景分别交给真实 Python 3.9、3.12 解释器执行，全部通过。后者是标准库级生命周期验证，不冒充两个版本的完整 ML 套件运行。

## 提交记录

| 提交 | 内容 |
|---|---|
| [9475651](https://github.com/isLinXu/Vitriol/commit/9475651) | 在 Ubuntu 24.04 保留 Python 3.9 CI |
| [00fd8e7](https://github.com/isLinXu/Vitriol/commit/00fd8e7) | 使用隔离构建安装声明的后端依赖 |
| [f75b81e](https://github.com/isLinXu/Vitriol/commit/f75b81e) | 发布失败摘要并留存测试报告 |
| [ebcd41b](https://github.com/isLinXu/Vitriol/commit/ebcd41b) | 延迟创建任务队列并保留同步重分配任务 |
| [b18a3ca](https://github.com/isLinXu/Vitriol/commit/b18a3ca) | 浮点相似度断言允许 1e-15 的舍入误差 |

## 验证记录

- 构建修复前日志：`/tmp/vitriol-ci-build-before.log`，缺少 setuptools 后端。
- 构建修复后日志：`/tmp/vitriol-ci-build-after.log`，源码包与 wheel 成功。
- 构建产物：`/tmp/vitriol-ci-build-after/`。
- 回归修复前日志：`/tmp/vitriol-phase9-before.log`。
- 专项测试日志：`/tmp/vitriol-phase9-focused.log`。
- 本地完整非集成测试：**2920 passed、19 skipped、18 warnings**，73.41 秒；语句覆盖率 **71.76%**，18120 / 25250。日志 `/tmp/vitriol-phase9-full.log`，JUnit `/tmp/vitriol-phase9-full.xml`，覆盖率 `/tmp/vitriol-phase9-coverage.xml`。
- 全部 src 与新增回归文件的 Ruff、CI 范围 API/CLI/security 三文件的 mypy、git diff --check 通过。
- [最终 GitHub CI 43477542304](https://github.com/isLinXu/Vitriol/actions/runs/43477542304)，对应 `b18a3ca`：**四版本测试任务及构建任务全部成功**，详见下表。任务包含既有阻断性 lint/type 检查、带覆盖率的非集成测试与 CLI 检查；历史非阻断检查仍保持原有策略。
- 本轮没有重跑本地模型集成套件；第八轮的 39 项集成结果仍作为此前生成与 KV 修复的验证记录。

| 远程任务 | 结果 |
|---|---|
| [Python 3.9 / Ubuntu 24.04](https://github.com/isLinXu/Vitriol/actions/runs/43477542304/job/132065202453) | success |
| [Python 3.10 / Ubuntu 24.04](https://github.com/isLinXu/Vitriol/actions/runs/43477542304/job/132065201990) | success |
| [Python 3.11 / Ubuntu 24.04](https://github.com/isLinXu/Vitriol/actions/runs/43477542304/job/132065202451) | success |
| [Python 3.12 / Ubuntu 24.04](https://github.com/isLinXu/Vitriol/actions/runs/43477542304/job/132065202343) | success |
| [Distribution build](https://github.com/isLinXu/Vitriol/actions/runs/43477542304/job/132067904618) | success |

本地保留远程结果快照 `/tmp/vitriol-phase9-ci-run.json` 和 `/tmp/vitriol-phase9-ci-jobs.json`。后续文档提交不改变上述已验证的源代码、测试或 CI 配置。

## 适用边界

本轮队列测试验证同步构造和单个执行循环中的使用，不承诺同一个协调器跨事件循环或跨线程重用。分布式模块仍包含模拟任务执行，本轮不构成真实多机生成的验证。未改变 Python 支持范围、覆盖率门槛或数值算法。
