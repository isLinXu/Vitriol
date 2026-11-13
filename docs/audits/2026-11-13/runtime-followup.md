# Vitriol 第三轮：运行时缓存与补丁生命周期

日期：2026-11-13。延续第二轮工作区，未提交、推送或发布。仍使用 Python 3.11.18、PyTorch 2.6.0+cpu、Transformers 4.57.6，没有新增依赖或下载模型权重。

后续记录：[第四轮模型缓存接入与故障回退](cache-hook-followup.md)已继续处理本报告保留的通用 cache hook 问题。本文保留第三轮历史结果。

## 1. 修前问题与修复

最初 25 项新增回归在修复前为 **23 failed / 2 passed**。对照源码保存在 `/tmp/vitriol-phase3-before/`，没有回滚已有工作区修改。

| 问题 | 修前行为 | 修后行为 |
|---|---|---|
| 原地修改 Q/K/V | 缓存键只有对象 ID，复用过期量化结果；adaptive 分配也依赖 Q | 核对活对象、tensor version、shape/stride/offset、存储地址、dtype/device；输入变更重新处理 |
| 对象 ID 重用 | 原对象销毁后，另一组张量可能命中旧结果 | entry 保存输入弱引用并验证 `ref() is tensor`；使用确定性 ID 碰撞测试覆盖 |
| 无 version 的 inference tensor | 无法识别其内容变化，却仍缓存 | 跳过复用，继续正常量化和注意力计算 |
| autograd 与 autocast | 复用已反向传播的计算图；缓存结果可能跨精度上下文复用 | 需要梯度的工作和 CPU/CUDA autocast 绕过预处理缓存 |
| 补丁所有权 | 未 apply 的实例也可 restore 另一个实例；第二个实例 apply 静默无效 | 只撤销自己的安装；同一实例重复 apply 幂等，不同 active 实例显式报错 |
| 外部 wrapper 与重新安装 | 外层封装仍会调用已要求撤销的量化逻辑 | 每次安装独立的 active 状态；撤销后旧闭包直接调用原函数；保留外部 wrapper，兼容 `functools.wraps` |
| 生命周期与并发退出 | restore 不清空缓存，正在执行的量化可能在 restore 后重新填入缓存 | restore 清空缓存；安装/撤销与 LRU 修改受锁保护；失活安装禁止回填 |
| 注意力参数 | compute-skip/sparse 忽略位置形式的 mask/dropout/causal、scale 和 GQA | 显式接收 SDPA 参数，自定义分支传递 scaling 并正确扩展 GQA heads |
| sparse V 数值语义 | 二维 bias 无法接收 batch/head mask；dropout 后归一化抵消缩放 | 在 logits 上广播 mask；剪枝归一化后再 dropout；全屏蔽行输出零 |
| 梯度与精度 | 全屏蔽 softmax 的 NaN 会污染反向传播；半精度中间值容易溢出；FP64 value 范数被降为 FP32 后可能溢出并错误屏蔽全部分块 | softmax 前后处理全屏蔽行；FP16/BF16 使用 FP32 中间计算，FP64 包括范数在内保持 FP64 |
| 默认 adaptive codec | 多个配置共享同一个可变 codec 对象 | 使用 default_factory，每个配置独立实例 |

未根据假设提高缓存容量或添加吞吐承诺。缓存仍以 entry 数量为上限，不是显存字节预算。

## 2. 关键设计与兼容变化

缓存命中需要三类条件同时成立：输入对象仍是同一个活对象；可追踪的版本和布局未变化；量化配置相同。输入使用弱引用，避免仅为身份校验延长原始 Q/K/V 生命周期；缓存保存的是量化结果。不能验证的情况直接重新计算，不使用内容采样哈希猜测是否变化。

`torch.inference_mode()` 创建的张量没有版本计数器。本轮选择跳过这些输入的预处理复用，保留量化功能；没有把 ID 相同当成不可变的保证。默认 `no_grad()` 场景下有版本计数器的张量仍可命中缓存，正向命中与原地变更后的失效都有测试。

通过 `with KVRuntimePatcher(config): ...` 使用补丁，可以在异常时恢复；同一实例的嵌套上下文直到最外层退出才恢复。手动 apply/restore 继续可用。不同实例不能同时安装在同一个 SDPA 全局入口上，第二次安装现在抛出明确的 RuntimeError，这是替代旧静默失效行为的兼容变化。

撤销期间，已经开始的注意力调用允许用原策略完成，但不能重新填充已清空的缓存；后续调用使用恢复的入口。事件同步测试在量化中途暂停工作线程，主线程 restore 后再放行，确认输出有限、入口恢复、缓存保持为空。该测试验证单个进程内的退出竞争，不是多租户并发策略隔离证明。

Dropout 的逐元素 seeded 对照使用与 block size 对齐的 key 长度，确保双方抽取同样数量的随机掩码；非对齐 padding 会改变随机数消耗，不能要求与另一种布局逐位相同。

## 3. 验证范围

`tests/test_kv_runtime_lifecycle.py` 包含 **37 项**回归，覆盖缓存命中/失效、ID 碰撞、inference tensors、重复 backward、autocast、原输入释放、配置隔离、所有权冲突、外部 wrapper、嵌套/异常退出、处理中撤销、位置参数、custom scale、GQA、不同 K/V 维度、批量掩码、全屏蔽梯度和大值 FP64 范数。数值结果与缓存关闭的同策略或 PyTorch SDPA 对照，而不是仅断言内部方法被调用。

`tests/integration/test_adaptive_kv_decode.py` 从 1 项扩展为 **4 项**：缓存容量 0/2 × no_grad/inference_mode。每项使用本地两层 Llama、head_dim=80、GQA、固定 8 bit 分配，连续处理两个不同 prompt；每次执行 prefill 与 3 步解码，并与完整前缀重算 logits 比较。每项确认 **28 次实际 KV patch 调用**；缓存开启的普通推理路径确认 28 次 miss 和实际 LRU 淘汰，inference_mode 确认不缓存；restore 后缓存必须清空。

模型测试的 Q/K/V 每次重新创建，因此并不声称在真实模型中获得命中加速；命中行为由重复使用同一输入的单元回归验证。固定 bit 测试隔离运行时和缓存正确性，可变 bit 分配仍由前两轮及本轮单元测试覆盖。

FP64 边界用例把 V 放大到 `1e20`。独立载入修前 codec 后，compute-skip 返回 64 个零，相对最大误差为 1.0；修后 64 个输出均非零，相对最大误差为约 `1.0004e-12`，符合 PyTorch SDPA 参照。该用例修前测试为 1 failed，修后纳入完整套件。

## 4. 仍有边界

- 本补丁修改的是进程级 `torch.nn.functional.scaled_dot_product_attention`。不同请求不能据此假设拥有互相隔离的并发量化策略。
- version 校验覆盖 PyTorch 跟踪的原地修改；通过 `.data`、外部 NumPy 共享内存、原生扩展等绕过版本计数的写入不在此保证内。存在此类写入的调用方应关闭预处理缓存。
- 处理中的输入被其他线程同时修改不保证一致性；本轮保护的是补丁状态和缓存容器，不是任意 tensor 并发读写事务。
- CUDA/GPU、Triton、其他设备 autocast、真实预训练模型语料 PPL 和吞吐/显存收益未验证。
- 其他 cache hook、Qwen 专用补丁和 NAS/RL 遗留问题仍按原审计边界处理；本轮所有权机制专用于 KVRuntimePatcher。

## 5. 最终验证

| 检查 | 结果 |
|---|---|
| 最终完整非集成套件 | **2745 passed / 19 skipped / 0 failed**，90.22 秒 |
| 最终语句覆盖率 | **71.06%（17660 / 24853）**；第二轮为 70.90% |
| 全部集成测试 | **24 passed / 0 failed**，10.55 秒 |
| 相对第二轮净新增测试 | **40 项**：37 项新单元回归，模型集成从 1 项扩展到 4 项 |
| Ruff | 全部 `src/`、新生命周期回归和模型集成测试通过 |
| CI 关键模块 mypy | api/server.py、cli/main.py、security/context.py 通过；无类型函数体仍按既有配置处理 |
| diff whitespace | `git diff --check` 通过 |

19 项跳过与前两轮一致，涉及未开放的远端模型访问、缺失本地模型资产、当前依赖不包含 Qwen3.5、空张量评分不支持等。19 条 warnings 为既有依赖弃用或实验能力提示。原 60% 覆盖率要求未降低。

本轮修改的源码为 `src/vitriol/patches/kv_runtime_patches.py`、`src/vitriol/patches/turboquant.py`、`src/vitriol/kv/codec.py`；新增 `tests/test_kv_runtime_lifecycle.py`，扩展 `tests/integration/test_adaptive_kv_decode.py`，并同步 CHANGELOG 和审计文档。

复现：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/test_kv_runtime_lifecycle.py tests/integration/test_adaptive_kv_decode.py \
  -o addopts='' --timeout=60 -q --tb=short

PYTHONPATH=src GRADIO_ANALYTICS_ENABLED=False OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests --ignore=tests/integration --timeout=60 -q -ra --tb=short

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/integration -o addopts='' --timeout=60 -q --tb=short
```

本地日志：`/tmp/vitriol-phase3-before.log`、`/tmp/vitriol-phase3-double-norm-before.log`、`/tmp/vitriol-phase3-final-full.log`、`/tmp/vitriol-phase3-final-coverage.xml`、`/tmp/vitriol-phase3-integration.log`。临时日志不作为仓库交付依赖，回归测试和关键结果均已保存到项目中。
