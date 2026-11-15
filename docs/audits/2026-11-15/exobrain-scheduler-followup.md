# 第二十轮：ExoBrain 调度与 KV 压缩边界

本轮审计低覆盖的 `src/vitriol/kv/exobrain/scheduler.py`，覆盖自适应注入、PPL 计算、教师 KV 压缩和 KVPrefetcher。此前覆盖率报告中该模块语句覆盖率约为 18%，且没有独立 scheduler 测试文件。

## 修前复现

新增用例在原实现上复现了 22 个失败场景，之后另一个 SVD 回归也独立失败：

- `mean_pool` 截去不能整除窗口的尾部 token；Top-K 按向量幅值返回位置，打乱原始序列顺序。
- 错误的 KV 形状会落到 reshape/gather 的底层异常；压缩器参数和 scheduler 参数缺少边界校验。
- PPL 函数将越界 token ID 静默夹到词表边缘，形状不合法或空序列会返回错误结果；`eps` 参数未生效。
- SVD 将序列长度变量覆盖为奇异值张量，reshape 异常被宽泛捕获后静默返回原 KV，因此 `svd_lowrank` 实际没有执行近似。
- `quantize_8bit` 返回原浮点 dtype，`svd_lowrank` 也返回稠密原尺寸张量；二者不会减少实际存储，但原说明会让调用方误以为节省内存。

## 修复

均值池化把末尾不完整窗口汇总为一个尾部位置；Top-K 先筛选重要位置，再按源序列索引排序，并分别支持不同的 key/value 特征维度。压缩器检查方法参数、张量秩、批头序列尺寸、设备与浮点 dtype；scheduler 检查阈值、窗口、预热步数、注入率及 PPL/熵观测值。无步骤时的 skip rate 现在为 0。

PPL 计算校验 logits/target 形状、词表范围和 epsilon，拒绝空输入与非法 token ID，并实际使用 `eps` 限制极端损失。SVD 修复变量覆盖，以请求的 rank 进行低秩重建；数值运行错误会记录警告并安全回退，不再吞掉编程类异常。文档明确量化路径为 fake quantization、SVD 返回稠密张量，因此两者都不声称实际降低内存占用。

## 验证

| 检查 | 结果 |
|---|---|
| ExoBrain scheduler、KV 与 inference 定向测试 | **113 passed**，3.49 秒 |
| 完整非集成套件 | **3062 passed、19 skipped、18 warnings**，80.96 秒 |
| 项目语句覆盖率 | **73.21%**（18,676 / 25,511），超过 60% 门槛 |
| scheduler.py 语句覆盖率 | 从 **17.8%** 提升至 **92.0%**（241 / 262） |
| 修改文件 Ruff / scheduler MyPy | 通过 |
| sdist 与 wheel 构建 | 通过 |
| GitHub Actions Python 3.9–3.12 矩阵 | **4/4 成功**（含覆盖率与 CLI 检查）([运行记录](https://github.com/isLinXu/Vitriol/actions/runs/43720801292)) |

完整测试日志为 `/tmp/vitriol-phase21-full.log`，JUnit 为 `/tmp/vitriol-phase21-full.xml`，Coverage XML 为 `/tmp/vitriol-phase21-coverage.xml`。构建成功；setuptools 继续报告项目既有 license 表与 license classifier 弃用警告。

## 范围与限制

KV 压缩器目前是独立导出的组件，没有接入 `ExoBrainInferencePipeline` 的默认推理路径。fake quantization 和稠密 SVD 仅改变数值，不节省真实张量存储；要实现物理压缩，需要新增带 scale/低秩因子的存储格式，并在注意力读取端配套解码。该格式变更超出本轮范围。
