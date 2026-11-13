# Vitriol KV / PPL 数值审查

## 已修复

1. QJL packed attention 在 prefill / 单 token cache query/key shape 相同时，helper 默认为逐向量内积，attention 却要求 QK^T。这可导致跨 head 广播、形状错误、错误分数。保留 helper 旧逐向量语义，attention 所有调用显式 `all_pairs=True`。
2. QJL CUDA projection 选择 fp16，unpack residual 为 fp32，原 matmul 类型不匹配；bf16 query 对 fp16/fp32 projection 同样失败。累加显式 fp32，attention 输出恢复 query dtype；普通 packed decode 进入 SDPA 前统一 Q/K/V dtype。CPU 回归模拟 CUDA 存储类型布局，不代表 GPU 实测。
3. Sliding window 截短 K/V 后仅清空 k_proxy，没有清除 residual_proxy_eligible，compute skip/sparse/temporal 分支仍使用未裁剪的 packed K。同步禁用旧 proxy 后使用裁剪的 dense K/V。
4. compute_skip 未 mask 对齐填充 token，导致 padding 分到真实 attention mass；全 mask 行 softmax 为 NaN；dropout 在归一化前执行，缩放被抵消。dense / residual proxy 路径均修复。Temporal pooling 全 mask 清零并用 fp32 softmax/归一化。
5. PPL generation 先消费当前 token，再用预测下一 token 的 logits 反向评分当前 token，NLL 时序错位。改为先评分后消费，传播增长的 attention_mask、排除 padding loss，EOS / 最后一步不做多余 forward。
6. PPL token mismatch 任意乘 0.1 冒充 KL 和 MSE、虚构 layer 0 压缩率 0.25。生成 token 无法推出分布距离，改为 unknown (`null`, 空 layers)，report / JSON 明确 PPL 为各自 greedy continuation、内存为估计、吞吐含 prefill。
7. PPL 未获 cache stats 时原先返回 0 bytes / 100% savings，异常时又假定 4x savings。改为缺乏遥测时 baseline 保守估计，报告标明 0% fallback。
8. Adaptive bit allocation 在 s=1 除 log(1) 为 NaN，s=0 无明确定义；改为 singleton 零熵和 empty key ValueError。数值回归另复现 vectorized quantizer 的 torch.clamp(Tensor, int, Tensor) TypeError；改为 clamp_min + minimum，确保各 head quantization levels 的上下界。
9. Tuned PPL inference 异常会泄漏共享 Transformers patch；finally 按逆序还原两个 patcher，单个 restore 异常也会尝试另一个。

## 改动
- src/vitriol/kv/codec.py
- src/vitriol/kv/cache_store.py
- src/vitriol/kv/temporal_pooling.py
- src/vitriol/bench/ppl_evaluator.py
- tests/test_kv_numerical_regressions.py
- tests/test_ppl_numerical_regressions.py

## 边界与剩余发现
- 尚未运行真实 GPU、Triton、远端 checkpoint、长上下文质量/峰值 VRAM/吞吐。dtype 回归仅 CPU 张量。
- PPL 不捕获对齐 logits / layer activation，不再假报 KL。比较不同自生成文本仍不能等同 held-out corpus PPL；num_prompts 目前没有在 evaluate 中循环。
- 内存统计仅 encoded KV payload，不计 raw cache / decode cache / projection 全驻留开销；PPL 没有实际 handle 遥测时没有可证实节省。
- `walsh_hadamard_rotate` 非 2 次幂维度 pad 后截断问题在首轮保留，现已由[第二轮修复](kv-followup.md)处理；新方案保持维度、范数和内积，旧版旋转缓存需重建。
- 未逐项证明所有 experimental KV codecs / 所有 Transformers model family patch 的端到端正确性。

## 验证
- Python compileall 通过。
- git diff --check 通过。
- 初步 CPU：genception Python3.12 / torch2.6+cu124 环境（CPU tensors）：149 passed；此环境 transformers5 / numpy2 不符合项目声明，待 .venv 正式复跑。

## 修前 HEAD 源码隔离复现（CPU）

使用 `/tmp/vitriol-original-numerics.py` 从 `git show HEAD:...` 加载原始代码到隔离模块，不修改工作树：
- prefill query `(1, 2, 3, 16)`，原输出 `(1, 2, 2, 16)`。
- compute_skip epsilon=0 与 SDPA 因 padding 产生 max_abs_error `0.313884437084198`。
- compute_skip 全 mask 输出 `isfinite=False`。
- half projection unpack：`expected m1 and m2 to have the same dtype, but got: float != c10::Half`。
- singleton adaptive bits：`isfinite=False`。
- adaptive quantization：`clamp() received an invalid combination of arguments - got (Tensor, int, Tensor)`。
- 同一确定性模型生成 `[2, 3, 4]`：原 PPL `22.660699892750284`，正确值 `1.128209475427866`。

扩展 CPU 回归（genception 预检）：

```text
PYTHONPATH=src OMP_NUM_THREADS=2 /comp_robot/home/xu.lin/miniconda3/envs/genception/bin/python -m pytest -q -o addopts='' tests/test_kv*.py tests/test_turboquant*.py tests/test_ppl*.py tests/test_spectral_predictive.py tests/test_hybrid_ultra.py
548 passed, 1 skipped, 2 warnings in 24.45s
```

两条 warning 来自借用环境未安装 pytest-asyncio（配置项未知）；正式 `.venv` 验证由主任务统一执行或通知后复跑。无需网络模型下载。

## 正式受支持环境验证

`.venv`：Python 3.11、torch 2.6 CPU、transformers 4.57.6、numpy 1.26。

```text
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q -o addopts='' --timeout=60 tests/test_kv*.py tests/test_turboquant*.py tests/test_ppl*.py tests/test_spectral_predictive.py tests/test_hybrid_ultra.py
547 passed, 2 skipped in 14.54s
```

本轮 4 个 src 文件和 2 个新增 test 文件运行 `.venv/bin/python -m ruff check` 全部通过，无需追加改动。

## 全量发现后补修：bench 公共导出

基线全量复现 27 个 bench CLI ImportError：`bench.__getattr__` 仍把已拆分的 `analyze_kv_quantization` 和三个 TurboQuantum 函数路由到 `runner`，而实现已经迁至 `runner_kv` / `runner_turboquant`。

修复 `src/vitriol/bench/__init__.py` 的惰性出口，继续使用原实现并保留 `__all__`。新增 `tests/test_bench_public_exports.py` 验证四个公共函数身份，以及独立 Python 进程禁止 torch/transformers/accelerate 导入时仍能导入 bench 和轻量工具。

```text
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q -o addopts='' --timeout=60 tests/test_bench_public_exports.py tests/test_cli_bench.py tests/test_cli_commands_misc.py tests/test_cli_optional_dependencies.py
64 passed in 12.16s
```

这两个补修文件 Ruff / diff whitespace 检查通过。
