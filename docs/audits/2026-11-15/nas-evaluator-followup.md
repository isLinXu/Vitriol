# 第十八轮：NAS 评分边界与本地数据集加载

本轮审计覆盖 `src/vitriol/nas/evaluator.py` 的 RankMe、expressivity 指标和本地数据源分支。该模块在上一轮覆盖率报告中只有 11% 语句覆盖，现有 NAS 测试主要覆盖搜索空间与搜索器，没有验证这些评分边界和本地文件加载。

## 修前复现

增加 6 个回归场景后，在原实现上 6 项全部失败：

- RankMe 对中心化后全零的隐藏表示计算 `S / S.sum()`，得到 NaN。
- expressivity 对长度为 1 的序列计算相邻 token 能量；差分张量为空，`mean()` 得到 NaN。
- 这两种结果也会进入 `HybridEvaluator.evaluate()` 的综合分数。搜索器直接按分数选择最佳候选，NaN 会让随机搜索无法选出首个最佳 gene，并使进化搜索排序不可靠。
- 本地加载器只对目录做 `glob`，把单个文件路径当目录处理；目录扫描遗漏 `.jsonl`。此外传入的 `split` 参数没有生效，调用始终加载 `train`。

## 修复

将有效秩和相邻 token 能量计算提取为共享函数，独立代理和融合评估路径共用。有效秩遇到空表示或奇异值总量为零/非有限时返回 `0.0`；长度不足两个 token 时，Dirichlet 能量返回 `0.0`。因此退化表示会返回明确的低分，不会把 NaN 带入综合评分。

本地数据源现在接受单个 `.parquet`、`.json`、`.jsonl` 文件及目录递归发现。目录含 parquet 时仍优先使用 parquet；否则按稳定顺序合并 JSON 与 JSONL 文件。加载器按请求的 split 构造 `data_files` 映射并返回该 split。

## 验证

| 检查 | 结果 |
|---|---|
| NAS 定向测试及新增回归 | **32 passed**，11.49 秒 |
| 完整非集成套件 | **3012 passed、19 skipped、18 warnings**，75.52 秒 |
| 语句覆盖率 | **72.26%**（18,369 / 25,420），通过 60% 门槛 |
| 修改文件 Ruff | 通过 |
| CI 指定 MyPy 文件 | 通过：API server、CLI main、security context |
| CLI version/help 冒烟 | 通过 |
| sdist 与 wheel 构建 | 通过 |
| 工作树空白检查 | `git diff --check` 通过 |

完整测试日志为 `/tmp/vitriol-phase18-full.log`，JUnit 为 `/tmp/vitriol-phase18-full.xml`，Coverage XML 为 `/tmp/vitriol-phase18-coverage.xml`。六项修前失败在修改实现前直接复现；对应回归已保留在 `tests/test_nas_evaluator.py`。

代码与测试已提交为 [`cb91e98`](https://github.com/isLinXu/Vitriol/commit/cb91e98798dc2e64ccecd9c37828f585c0e49cb5)。[Python CI run 43676636779](https://github.com/isLinXu/Vitriol/actions/runs/43676636779) 全部成功：[Python 3.9](https://github.com/isLinXu/Vitriol/actions/runs/43676636779/job/132698114198)、[Python 3.10](https://github.com/isLinXu/Vitriol/actions/runs/43676636779/job/132698113986)、[Python 3.11](https://github.com/isLinXu/Vitriol/actions/runs/43676636779/job/132698114139)、[Python 3.12](https://github.com/isLinXu/Vitriol/actions/runs/43676636779/job/132698114160)、[分发包构建](https://github.com/isLinXu/Vitriol/actions/runs/43676636779/job/132698984230)。

## 范围与限制

本地环境没有安装可选 `datasets` 包，因此没有调用真实 Hugging Face 数据集读取；单文件、目录、格式和 split 参数由加载器桩验证。也没有下载真实模型运行完整 NAS 搜索或做 GPU 压力测试。直接对整个 evaluator 执行 MyPy 时，检查被已安装 Pydantic 依赖中的 Python 3.10 语法与项目 Python 3.9 检查目标不兼容所阻断；仓库 CI 指定的 MyPy 检查通过。发行包构建成功，但 setuptools 对现有 license 表及 license classifier 给出弃用警告，本轮未更改项目级发布元数据。
