# 第十九轮：ExoBrain 推理缓存融合边界

本轮审计 `src/vitriol/kv/exobrain/pipeline.py` 的教师 KV 注入与缓存重建路径。该文件在此前覆盖率报告中只有约 8% 语句覆盖，现有 ExoBrain 测试没有覆盖短教师缓存、未初始化层、DynamicCache 配置保留和 GQA KV 头数差异。

## 修前复现

新增的缓存回归用例在原实现上复现了以下问题：

- 教师序列短于 shell cache 时，replace 会把 shell 前缀重复拼到输出尾部；residual 会截断 shell 缓存；gated 会因序列长度不匹配报错。
- 含 `None` 未初始化层的 legacy cache 在解包时失败。
- 重建 DynamicCache 时未传入模型配置，滑窗注意力层会变成普通缓存层，并丢失 offload 设置。
- 教师和 shell 的 KV head 数不同时，即使 head_dim 相同也无法进行维度投影，注入会失败。
- 手动 gated 融合忽略 `gate_temperature`；无效 `fusion_mode` 会被误当作 gated 模式。两项都增加了定向失败用例，并在修复前分别确认失败。
- 项目支持 Transformers 4.40 起的版本，而 4.40 的 `DynamicCache` 只接受无参数构造；若直接传入新版本的配置/offload 参数，旧版本推理会在缓存重建时失败。

首批 7 个缓存回归场景在修复前全部失败；修复后逐步加入配置回归，定向测试最终通过 74 项。

## 修复

replace、residual 和 gated 仅融合教师与 shell 序列重叠区，并保留 shell cache 的尾部，使返回缓存长度与输入一致。replace 在重叠区采用教师 KV；residual/gated 在重叠区混合两侧 KV。遇到批量、KV 头数、键值长度或特征维度不兼容的教师 KV 时，会记录警告并保留 shell 层，避免一次不兼容的教师缓存中断推理。

DynamicCache 重建按安装版本的构造器签名筛选配置与 offload 参数，在支持这些能力的版本保留模型配置、滑窗层类型和 offload 设置；在旧版本只使用其支持的无参构造，避免破坏项目声明的兼容范围。legacy tuple cache 保留 `None` 层及原有额外 tuple 元素。gated 注入使用配置的温度，构造器拒绝非正数、非有限温度及不支持的融合模式。

上游源码确认 Transformers 4.40 的 `DynamicCache.__init__` 无参数，而 4.49 仅接收 `num_hidden_layers`；本轮的版本兼容测试模拟旧式无参构造。[Transformers 4.40 cache_utils](https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/cache_utils.py#L69-L82)、[Transformers 4.49 cache_utils](https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/cache_utils.py#L313-L339)。

## 验证

| 检查 | 结果 |
|---|---|
| ExoBrain 推理缓存回归及现有 KV 测试 | **75 passed**，2.61 秒 |
| 完整非集成套件 | **3024 passed、19 skipped、18 warnings**，116.44 秒 |
| 语句覆盖率 | **72.59%**（18,472 / 25,446），超过 60% 门槛 |
| 修改文件 Ruff 与 `git diff --check` | 通过 |
| CI 指定 MyPy 文件 | 通过：API server、CLI main、security context |
| CLI version/help 冒烟 | 通过 |
| sdist 与 wheel 构建 | 通过 |

完整测试日志为 `/tmp/vitriol-phase20-full.log`，JUnit 为 `/tmp/vitriol-phase20-full.xml`，Coverage XML 为 `/tmp/vitriol-phase20-coverage.xml`。构建继续报告项目既有 license 表与 license classifier 弃用警告；本轮未更改发布元数据。

代码修复分两次提交：[`8b7bbaa`](https://github.com/isLinXu/Vitriol/commit/8b7bbaa)（缓存几何和融合边界）与 [`cd2b1f7`](https://github.com/isLinXu/Vitriol/commit/cd2b1f7)（旧版 DynamicCache 构造器兼容）。[Python CI run 43714458233](https://github.com/isLinXu/Vitriol/actions/runs/43714458233) 在 Python 3.9、3.10、3.11、3.12 及发行包构建上全部成功。

## 范围与限制

验证使用合成 KV 张量和现有 ExoBrain/KV 测试，没有下载真实教师模型、运行 GPU 推理或评估生成质量。KV 头数不同的教师层目前安全回退到 shell KV，不执行跨头映射；后续若要求融合 GQA/MQA 头，需要定义并验证明确的头映射策略。
