# Vitriol 第四轮：模型缓存接入与故障回退

日期：2026-11-13。在前三轮未提交改动上继续修复，未提交、推送或发布。环境仍为 Python 3.11.18、PyTorch 2.6.0+cpu、Transformers 4.57.6；没有新增依赖或下载权重。

## 1. 修前证据

首批 22 项新增回归在修复前为 **18 failed / 4 passed**。当前 Transformers 的 `ALL_ATTENTION_FUNCTIONS` 为 `AttentionInterface` 映射，没有旧 `get_interface` 方法；原通用 attention patch 因而只打印 unsupported 警告并返回，没有实际接入。与此同时 cache hook 仍可能启用 short-circuit decode，返回仅包含当前 token 的 K/V。

即使旧 registry 可用，原实现也通过 `_thread_local.current_cache` 关联 attention 与 cache。两个请求在同一线程交错 update 后，先发请求可能读到后发请求的 store；该强引用还会延长完成请求的缓存生命周期。原生更新被短路后，后端读失败的 catch 分支只能把不完整的 K/V 交给原 attention，异常被吞掉但结果已经错误。

新增故障用例还发现：原生 mutation 在修改内部数据后抛错时，侧路 store 没有失效。单独补充回归修前为 **1 failed / 30 passed**，现已在 finally 中使侧路状态失效。

## 2. 修复行为

| 路径 | 修复结果 |
|---|---|
| registry 接入 | 同时支持当前 MutableMapping 和旧 get_interface；无支持接口时明确失败，避免静默称已安装 |
| attention 与请求关联 | 在实际返回的 key 上挂一次性 binding，核对 key/value 对象、handle 弱引用、backend、layer、epoch 和安装状态；不再从线程级“当前缓存”路由 |
| 完整历史与故障回退 | 通用 cache hook 始终执行原生 update；后端写/读失败后返回原生完整 K/V 的 attention，并标记当前 handle 的失败层，避免反复调用已失败层 |
| 原生缓存状态 | get_seq_length 使用原生行为；get_mask_sizes 保留原生长度/offset，仅兼容 int/scalar position 输入 |
| warm cache | 初次接入或侧路长度不符时，从原生返回的完整 K/V 重建，不遗漏接入前历史 |
| reset/crop/reorder/批量选择或扩展 | 调用原生方法后使 store 和 binding epoch 失效；即使原生方法修改数据后抛异常，也不会继续信任旧 store |
| 安装与撤销 | 不同 active owner 明确冲突；只撤销自己安装的函数；保留外部 wrapper；继承来的方法恢复为继承关系；组合安装失败回滚已安装部分 |
| 显式清理与读取 | clear_vitriol_kv 同时使 binding 失效并释放新增元数据；读取未写入层明确失败，不顺手创建空 store |
| 可观测性 | backend/benchmark 输出增加 native_cache_retained、fallback_layers 和估计字节范围说明 |

binding 只弱引用 handle 和输入张量，避免保持请求存活。每次 update 和缓存 mutation 推进 epoch，旧 binding 即使尚未消费也不能路由到新的缓存状态。一次 attention 消费之后，相同张量再次用于其他调用时回到原 attention。

## 3. 明确的兼容与内存取舍

通用 hook 无法保证所有模型、所有 attention 后端都消费压缩 store，因此本轮优先保证可正确回退：**保留完整原生 KV 副本**。`CacheHookConfig.passthrough_update` 默认改为 True；False 作为兼容参数仍接受，但记录 warning 并执行完整原生更新，不再短路 decode。

这意味着通用 hook 路径可能同时持有原生 KV 和侧路 store，不能据此宣称显存下降。`estimated_kv_bytes_scope="vitriol_store_payload_excludes_native_cache"` 明确说明现有 store 字节估计不包括原生副本；GPU peak 指标仍独立存在，本轮没有 GPU 实测。原生和侧路可能共享部分 tensor storage，也没有把二者简单相加冒充精确进程内存。

这项取舍限于通用 CacheHookPatcher；独立 KVCacheStore 和 Qwen 专用 hook 不由本轮统一改变。若后续需要真正消除原生副本，应先建立针对目标模型的完整路由和故障恢复契约，再用显存实测验证。

旧 `_thread_local` 名称暂为内部导入兼容保留，但不再发布 current_cache。依赖此私有通道的实验 ExoBrain 旧专用 patch 需要迁移；它在当前 4.57.6 本就缺少可用 registry 接口，本轮不宣称修复该专用路径。ExoBrainBackend 等后端算法测试通过也不能替代接入验证。

## 4. 验证方法

新增 `tests/test_cache_hook_integrity.py` 的 **39 项**回归，同时覆盖映射和旧式 registry：真实 backend 命中、请求交错、一次性消费、错误 layer/替换 value、不完整历史故障、写失败降级、warm cache、缓存 mutation/异常、请求释放、owner 冲突、外部 wrapper、组合安装回滚、不支持 registry、未知层读取以及原生窗口 offset。

已有 mask-size 测试的 DummyCache 改为提供有意义的原生长度/offset 行为；不再依靠人为 `_vitriol_seq_lens` 覆盖原生缓存来通过测试。

新增 `tests/integration/test_native_cache_hooks.py` 的 **6 项**实际模型测试：正常/读失败/写失败 × passthrough True/False。使用本地随机两层 Llama、head_dim=80、2 个 query heads 和 1 个 KV head，执行两个不同请求、每次 prefill 后 3 步 decode，并逐步与完整前缀重算 logits 比较。正常路径每项确认 **12 次 backend attention 命中**；故障路径每项确认 **4 次层级降级**，后续不再反复调用失败层。撤销两个 hook 后继续使用同一 DynamicCache 解码，仍与完整前缀重算一致。

模型集成关闭量化以隔离缓存接入与故障恢复；量化的数值性质沿用前三轮和现有 KV 回归，不能把这个模型测试解释为量化质量评估。

## 5. 尚未覆盖的边界

- 当前真实模型验证限于 Transformers 4.57.6、CPU、Llama SDPA；旧 registry 使用测试替身验证协议，未安装完整版本矩阵。
- eager attention 在模型内部可能直接调用函数而不经过 registry；此时仍保留完整原生历史，但不能声称压缩 attention 被使用，应检查 read_attention_hit。
- 新注册的 registry 函数、模型私有 registry、专用 Qwen/ExoBrain patch、beam search 全流程、GPU/FlashAttention/Triton 尚未做真实模型矩阵验证。
- 原生缓存的 mutation 由原生实现负责；本轮保证侧路失效和重建，不承诺回滚原生方法已经执行的部分修改。
- hook 仍修改进程级接口；不同请求共享同一安装策略，不构成多租户策略隔离。同一个原生 cache 被多线程同时写入不在保证内。

## 6. 最终验证结果

| 检查 | 结果 |
|---|---|
| 完整非集成套件 | **2784 passed / 19 skipped / 0 failed**，77.40 秒 |
| 全部集成测试 | **30 passed / 0 failed**，22.48 秒 |
| 语句覆盖率 | **71.36%（17797 / 24938）**；第三轮为 71.06% |
| 相对第三轮新增测试 | **45 项**：39 项回归、6 项本地模型集成 |
| Ruff | 全 `src/` 及本轮两个新测试文件通过 |
| CI 关键模块 mypy | api/server.py、cli/main.py、security/context.py 通过；无类型函数体按既有配置处理 |
| diff whitespace | `git diff --check` 通过 |

19 项跳过与前几轮一致，原因包括远端模型访问未启用、缺失本地模型资产、当前 Transformers 不含 Qwen3.5、空张量评分尚不支持等。19 条 warnings 为既有依赖弃用/实验能力提示，没有降低 60% 覆盖率要求。

源码修改：`patches/cache_hooks.py`、`kv/backend.py`、`kv/utils.py`、`bench/runner.py`。新增两个测试文件，调整 `tests/test_cache_hooks.py` 的原生 mask fixture，并同步 CHANGELOG 与本轮报告。

复现命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/test_cache_hook_integrity.py tests/integration/test_native_cache_hooks.py \
  -o addopts='' --timeout=60 -q --tb=short

PYTHONPATH=src GRADIO_ANALYTICS_ENABLED=False OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests --ignore=tests/integration --timeout=60 -q -ra --tb=short

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/integration -o addopts='' --timeout=60 -q --tb=short
```

临时证据日志：`/tmp/vitriol-phase4-before.log`、`/tmp/vitriol-phase4-expanded.log`、`/tmp/vitriol-phase4-full.log`、`/tmp/vitriol-phase4-coverage.xml`、`/tmp/vitriol-phase4-integration.log`。关键证据和测试均保留在仓库，交付不依赖临时文件。
