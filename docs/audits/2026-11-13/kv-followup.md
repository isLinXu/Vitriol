# Vitriol 第二轮：KV 数值正确性与实际接入验证

日期：2026-11-13。接续首轮审计，在已有未提交改动上继续修复；未提交、推送或发布。环境沿用首轮的 Python 3.11.18、PyTorch 2.6.0+cpu、Transformers 4.57.6。本轮未下载模型权重。

后续状态：本文保留第二轮证据和计数。[第三轮](runtime-followup.md)已继续验证并修复 runtime preprocess cache 和补丁生命周期；模型测试扩展为缓存开/关、普通推理/inference_mode、连续两次请求。

## 1. 处理范围与修前证据

本轮从首轮遗留的非 2 次幂 Hadamard 问题出发，沿旋转、精度分配、缓存追加、滑窗淘汰和模型后端选择检查实际调用链。

| 问题 | 修复前可执行证据 | 修复 |
|---|---|---|
| 非 2 次幂旋转丢失信息 | seed=2026、形状 `[2,3,5,80]`，旋转两次的最大误差为 2.2096136，能量损失约 37.46% | 连续的 2 次幂分块，保留维度并保证正交、自逆 |
| 半精度中间和溢出 | 128 维、元素 1000 的 FP16 输入，最终结果可表示，中间蝶形累加却会溢出 | FP32 累加，输出恢复原 dtype |
| 自适应精度在错误坐标系评分 | Q 未旋转而 K 已旋转，开启旋转改变 attention entropy 和分配的 levels | 原始 Q/K/V 上计算分配，再执行量化旋转 |
| 追加 token 与 prefill 坐标/精度不一致 | 独立 adaptive 路径跳过追加量化；与 turbo 合用时未逆旋转；仅复用首个 batch 的 levels | 保存所有 batch 的 levels，对新 token 量化并逆旋转，保持旧前缀不变 |
| 重建后仍沿用旧 precision 状态 | quantization threshold 切换和 eviction 后，结果偏离重新 prefill 的参照 | 通过统一 prefill 流程重建，下一次 query 重新分配 |
| GQA 与不同 K/V 维度 | query/KV head 数不一致无法分配精度，K 扩展错误使用 V 的维度 | 按 KV head 聚合对应 query group 的重要性；分别使用 K/V 维度 |
| FP16 熵和零值统计产生 NaN | 大 logits、零 V 输入得到非有限 bits | logits、entropy、RMS 和量化计算使用 FP32 |
| 滑窗长度上限失效 | 9 个 token、上限 4、recent reserve 为 4 或 256 时仍保留 9 个 | reserve 不超过窗口；旧 token 配额为 0 时使用空切片 |
| 显式 SDPA 被兼容补丁覆盖 | 集成测试中模型变为 eager，KV patch 调用次数为 0 | 缺少 flash_attn 时仅回退显式 FlashAttention 2 请求，保留 SDPA/自动选择 |

最初新增的 54 项回归在修复前为 **40 failed / 14 passed**。扩展测试又独立复现 3 项滑窗失败，以及 2 项后端选择单元失败和 1 项真实模型集成失败。没有通过回滚工作区来制造对照：修前 codec/cache_store/turboquant 源码保存在临时目录中，数值对照通过独立模块载入。

## 2. 旋转算法与兼容范围

令维度的二进制分解为 `d = d₁ + … + dₙ`，各 `dᵢ` 都是 2 的幂且从大到小排列。使用归一化 Sylvester 矩阵组成 `R = diag(H_d₁, …, H_dₙ)`。因为每个分块满足 `HᵀH = I` 且 `H² = I`，所以整体也满足 `RᵀR = I`、`R² = I`。例如维度 80 使用 64 和 16 两个分块，不截断系数。

维度为 2 的幂时仍使用原标准变换；时间复杂度不超过 O(d log d)，不增加特征维度。FP16/BF16 增加 FP32 中间计算，未据此宣称吞吐提升。分块间不混合能量，正交性也不能证明实际语料量化质量优于其他旋转方案。

同一输入修复后的往返最大误差为 **3.5762787e-7**，FP32 测得能量损失为 **0.0**。回归覆盖 1、3、5、64、80、96、127、128 维，FP32/FP64/FP16/BF16、非连续输入、内积、范数、梯度、输入不被修改，以及标准 64 维稠密 Sylvester 矩阵对照。

本轮修改的是保持原维度的通用旋转及其兼容入口。TurboQuant/TurboQuantum 中保留完整 padding、显式携带 `orig_dim` 的 signed rotation 没有改变。使用旧版非 2 次幂通用旋转生成的实验缓存必须重建，不能把旧坐标系系数直接交给新逆变换。

## 3. 缓存与模型验证方法

连续解码回归同时覆盖保留/丢弃 raw cache、turbo 开/关、旋转开/关及 FP32/FP16/BF16。两个 batch 使用不同 attention 分布，确认 levels 确实不同；连续追加 3 次，逐元素比较旧前缀和新追加部分，并用独立逐 batch/head 的量化参照加 PyTorch SDPA 检查最终输出。K 维度为 80、V 为 48；另有 6 个 query heads 对 2 个 KV heads 的 GQA 参照。

阈值切换和滑窗重建分别与新建 store 的 prefill 结果比较。`keep_raw_cache=False` 时只能从已有近似值重建，会存在累积量化误差；本轮保证其行为一致，不宣称恢复原始 KV。

实际模型集成使用本地随机初始化的两层 Llama，hidden_size=160、2 个 query heads、1 个 KV head、head_dim=80。显式执行项目 `apply_all_patches()` 后，确认模型仍使用 SDPA。完成 3 token prefill 和 3 次单 token decode，每一步与完整前缀重算的最后位置 logits 比较，检查有限值、缓存长度，以及 **14 次实际 KV patch 调用**，最后恢复全局 SDPA 函数。

模型测试固定 8 bit 分配以隔离缓存与旋转正确性；可变精度另由单元测试覆盖。关闭 runtime preprocess cache 以避免重复利用预处理结果掩盖路径执行问题。这个测试没有证明模型任务精度、语料 PPL 或 runtime preprocess cache 的生命周期正确性。

## 4. 修改文件

- `src/vitriol/kv/codec.py`：可逆旋转、原坐标系分配、GQA、FP32 统计、完整 batch 元数据和固定 levels 追加量化。
- `src/vitriol/kv/cache_store.py`：统一 adaptive 追加流程、重建状态失效和独立 K/V 维度扩展。
- `src/vitriol/kv/hybrid_pipeline.py`：滑窗 reserve 和零旧 token 配额。
- `src/vitriol/patches/turboquant.py`：通用旋转入口复用 codec。
- `src/vitriol/patches/transformers_patches.py`：后端回退仅针对缺失的 FlashAttention 2。
- `tests/test_kv_rotation_streaming.py`：79 项数值与状态回归。
- `tests/test_attention_backend_selection.py`：8 项后端选择回归。
- `tests/integration/test_adaptive_kv_decode.py`：真实本地模型的 SDPA 接入与连续解码验证。

## 5. 尚未验证的范围

GPU/Triton、预训练模型长上下文语料质量、跨进程/多租户服务及 NAS/RL 遗留算法问题仍未由本轮验证。自适应路径返回浮点量化近似，不等同于已获得低比特打包的显存占用。未新增依赖，未调整测试通过门槛。

## 6. 最终验证结果

| 检查 | 结果 |
|---|---|
| 完整非集成套件，串行、原覆盖率门槛 | **2708 passed / 19 skipped / 0 failed**，53.63 秒 |
| 语句覆盖率 | **70.90%（17562 / 24770）**；首轮为 70.54% |
| 全部集成测试 + 后端选择回归 | **29 passed / 0 failed**，10.72 秒；包括 21 项集成和 8 项单元回归 |
| 新增测试数量 | **88 项**：87 项单元回归、1 项模型集成；8 项后端单元回归在上两条验证中均执行，不重复计入新增数量 |
| Ruff | 全部 `src/` 及本轮 3 个新测试文件通过 |
| CI 关键模块 mypy | api/server.py、cli/main.py、security/context.py 通过；仍提示未检查无类型函数体 |
| diff whitespace | `git diff --check` 通过 |

19 项跳过原因与首轮一致：远端模型访问关闭或本地模型资产缺失、当前 Transformers 不含 Qwen3.5、空张量评分暂不支持等。19 条 warnings 为既有依赖弃用/实验功能告警。

复现命令（仓库根目录）：

```bash
PYTHONPATH=src GRADIO_ANALYTICS_ENABLED=False OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests --ignore=tests/integration --timeout=60 -q -ra --tb=short

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/test_attention_backend_selection.py tests/integration \
  -o addopts='' --timeout=60 -q --tb=short

.venv/bin/ruff check src/ tests/test_kv_rotation_streaming.py \
  tests/test_attention_backend_selection.py tests/integration/test_adaptive_kv_decode.py
```

本地证据日志：`/tmp/vitriol-kv-phase2-before.log`、`/tmp/vitriol-kv-phase2-expanded.log`、`/tmp/vitriol-phase2-backend-before.log`、`/tmp/vitriol-phase2-final-full.log`、`/tmp/vitriol-phase2-final-coverage.xml`、`/tmp/vitriol-phase2-integration-final.log`。临时文件不作为仓库交付依赖；测试与本文记录保留在工作区。
