<p align="center">
  <h1 align="center">Vitriol</h1>
  <p align="center"><strong>LLM 架构探索、可视化与神经架构搜索平台</strong></p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-%3E%3D3.8-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/version-0.3.0-green" alt="Version">
  <img src="https://img.shields.io/badge/license-MIT-brightgreen" alt="License">
  <img src="https://img.shields.io/badge/torch-%3E%3D2.0-red?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/transformers-%3E%3D4.40-orange?logo=huggingface" alt="Transformers">
  <img src="https://img.shields.io/badge/source_files-150+-blueviolet" alt="Source Files">
</p>

<p align="center">
  <a href="README.md">English</a> · <a href="README_CN.md">中文</a>
</p>

---

Vitriol 是一个专为大语言模型 (LLM) 设计的综合性工具平台。它将**结构**与**权重**彻底解耦——让你在 MB 级体积下探索、可视化和优化模型架构，无需下载 GB 级真实权重。

## 核心能力

| 能力 | 描述 |
|------|------|
| **最小权重生成** | 13 种策略将 GB 级模型压缩至 MB/KB 级，完全兼容 `transformers` |
| **架构可视化** | 交互式 HTML + 3D 浏览器查看器，内置 **10 种专用分析器** |
| **神经架构搜索** | 面向 LLM 的 NAS，支持随机、进化、定向和**强化学习**搜索算法 |
| **架构进化** | 家族树、智能对比、性能模拟、创新时间线 |
| **压缩智能度** | 多维度评估框架 (CIS)，量化压缩策略的综合表现 |
| **量化推理** | TurboQuant KV Cache 压缩、**Triton GPU 加速内核**、PPL 困惑度评估 |
| **Web UI 与 REST API** | Gradio 图形界面 + 可选 FastAPI 服务端 |

## 核心亮点

- **GB → KB 跨数量级压缩**：13 种策略，从 Random 到 Ultra（stride=0 黑科技，1 个浮点数模拟任意大小张量）
- **零内存实例化超大模型**：利用 PyTorch Meta Device，在 8GB 内存机器上构建 397B 参数模型骨架
- **13 种权重生成策略**：Random、Compact、Ultra、HybridUltra、Sparse、StructuredSparse、Ternary、Binary、Quantized、LowRank、Learned、HybridLearned、Quantum
- **10 种架构分析器**：自动识别 GQA/MQA/MLA、RoPE、MoE（Shared+Routed Expert）、多模态等组件——**含 Qwen3.5 MoE 的 Linear/Full 注意力分层检测**
- **4 种 NAS 搜索算法**：随机搜索、进化算法 (GA)、定向搜索（约束优化 + 多目标帕累托优化）、**强化学习代理（实验性）**
- **插件化适配器系统**：自动发现注册表，快速扩展新模型系列
- **18 个 CLI 命令**：从生成到基准测试和推理的完整工具链
- **Web UI**：Gradio 界面集成进化树、对比、模拟和定向 NAS
- **REST API**：实验性 FastAPI 服务端，支持 HTTP 方式的模型生成与搜索
- **Triton GPU 加速**：FWHT 快速哈达玛变换、分块量化、位打包等 KV Cache 高性能内核
- **PPL 困惑度评估框架**：用真实困惑度指标替代旧的代理度量，量化推理效果验证更可信

## 设计理念：结构与数据解耦

Vitriol 的核心洞察是：**模型性能是结构与数据的混淆变量**。新模型不断刷榜，但一个根本问题越来越难以回答：

> *性能提升到底是因为更好的架构，还是更多的数据、更好的训练策略、或更长的训练时间？*

Vitriol 通过**彻底分离结构骨架与训练权重**来回答这个问题：

```
┌─────────────────┐       ┌──────────────────────┐       ┌──────────────────┐
│   结构层          │       │   桥接层              │       │   数据层          │
│                  │──────►│   init_empty_weights() │──────►│   generate_      │
│  config.json     │       │   from_config()        │       │   tensor(        │
│  (仅 KB)          │       │                        │       │     shape,       │
│                  │       │  param.shape ◄────────┼───────│     dtype,       │
│  hidden_size     │       │  param.dtype ◄────────┼───────│     name)        │
│  num_layers      │       │  named_parameters()   │       │                  │
│  num_heads       │       │                        │       │  13 种策略        │
│  model_type      │       │  无需 GPU、无需权重     │       │  纯算法生成        │
│                  │       │  下载                  │       │  无需训练          │
└─────────────────┘       └──────────────────────┘       └──────────────────┘
```

**三阶段解耦流水线：**

| 阶段 | 做什么 | 输入 | 输出 |
|------|--------|------|------|
| **1. 配置 → 结构** | 解析 `config.json`（下载约 KB 级） | HuggingFace 模型 ID | 包含所有架构属性的 `PretrainedConfig` |
| **2. 结构 → 骨架** | 在 `init_empty_weights()` 内用 `from_config()` 构建模型 | `PretrainedConfig` | 空壳模型，每个参数具有精确的 `(shape, dtype, name)` —— **零内存分配** |
| **3. 骨架 → 权重** | 通过策略的 `generate_tensor(shape, dtype, name)` 填充每个参数 | `(shape, dtype, name)` 三元组 | 结构完全兼容的权重文件 |

**为什么这对研究至关重要：**

- **零成本架构消融实验**：无需下载 140 GB 权重，即可比较 LLaMA-70B 与 Qwen-72B 的架构差异。70B 模型的骨架构建仅需约 5 秒（纯 CPU）。
- **隔离结构性贡献**：对不同模型生成结构相同但"数据"不同的权重（例如全部使用 `random` 策略），可以基准测试性能差距是源于架构还是训练。
- **无 GPU 的 CI/CD**：在纯 CPU 环境中运行 `transformers` 加载、分片验证和架构分析——无需 GPU、无需 100 GB 磁盘空间。
- **基于真实拓扑的 NAS**：神经架构搜索操作于真实的模型配置（而非简化代理），发现 `from_config()` 可实例化的架构。
- **公平基准测试**：生成相同架构但不同声称规模的 `random` 初始化模型，揭示所谓的"缩放定律"到底是结构性还是训练产物。

```python
# 示例：隔离架构与数据
# 相同结构，不同"数据"——纯算法生成
python -m vitriol.cli.main generate Qwen/Qwen2.5-72B --strategy random    # 全尺寸，真实分布
python -m vitriol.cli.main generate Qwen/Qwen2.5-72B --strategy compact   # 相同形状，零填充
python -m vitriol.cli.main generate Qwen/Qwen2.5-72B --strategy ultra     # 相同形状，1 个 float stride=0

# 三个模型结构完全相同（参数名、形状、数据类型一致）
# 仅"数据"填充策略不同——实现受控消融
```

## 快速开始

### 安装

```bash
git clone https://github.com/isLinXu/Vitriol.git
cd Vitriol

pip install -e .
# 开发依赖（测试/lint/类型检查）
pip install -e ".[dev]"
# Web UI
pip install -e ".[webui]"
# REST API（实验性）
pip install -e ".[api]"
```

### 常用命令

```bash
# 生成最小权重
python -m vitriol.cli.main generate Qwen/Qwen2.5-0.5B -o output/qwen-mini

# Ultra 极致压缩（最小体积）
python -m vitriol.cli.main generate Qwen/Qwen2.5-0.5B -o output/qwen-ultra --strategy ultra

# 架构可视化（交互式 HTML）
python -m vitriol.cli.main arch-viz Qwen/Qwen2.5-0.5B --html -o output/qwen_viz.html

# 3D 模型查看器
python -m vitriol.cli.main viz Qwen/Qwen2.5-0.5B --3d

# 神经架构搜索
python -m vitriol.cli.main nas --algorithm evolutionary --generations 10 --population 20

# 架构进化树
python -m vitriol.cli.main evolve tree -o output/evolution_tree.html

# 对比两个架构
python -m vitriol.cli.main evolve compare Qwen/Qwen2.5-7B DeepSeek-V3/DeepSeek-V3

# 性能模拟
python -m vitriol.cli.main evolve simulate Qwen/Qwen2.5-72B --gpu H100

# 启动 Web UI
python -m vitriol.cli.main webui
```

## CLI 命令参考

Vitriol 提供 **18 个命令**，统一入口：

```bash
python -m vitriol.cli.main <command> [options]
# 或
vitriol <command> [options]
```

| 命令 | 描述 |
|------|------|
| `generate` | 生成最小权重模型 |
| `validate` | 验证已生成模型（加载、分词器、推理） |
| `analyze` | 分析模型架构（层数、参数量、注意力类型） |
| `batch` | 从 YAML 配置批量生成 |
| `bench` | **KV Cache 压缩基准测试套件（6 个子命令）** |
| `export` | 导出生成的模型 |
| `visualize` | 生成权重可视化报告 |
| `viz` | 启动交互式模型查看器 (3D) |
| `arch-viz` | 从配置可视化架构拓扑 |
| `nas` | 神经架构搜索 |
| `vocab-viz` | 3D 词表可视化 |
| `weight-viz` | 3D 权重可视化 |
| `evolve` | 架构进化工具（树、对比、模拟、家族、时间线、推荐） |
| `exobrain` | External Brain 推理与蒸馏实验 |
| `hash` | 计算模型哈希指纹 |
| `infer` | **使用 TurboQuant 预设运行单条 prompt 推理** |
| `webui` | 启动 Gradio Web UI |

> **安全提示**：CLI 默认启用 `trust_remote_code=True` 以提高兼容性。如需更安全的 CI 环境，请传递 `--no-trust-remote-code`。

## 模型哈希指纹

`hash` 命令可为任意模型计算加密身份指纹，用于完整性验证、版本追踪和未授权修改检测。

```bash
# 完整指纹（架构 + 权重 + 行为）
python -m vitriol.cli.main hash /path/to/model

# 快速模式（仅架构，跳过权重扫描）
python -m vitriol.cli.main hash /path/to/model --fast
```

**三层哈希体系：**

| 哈希层级 | 输入 | 适用场景 |
|---------|------|----------|
| **架构哈希** | `config.json` 拓扑键值（hidden_size、层数、注意力头数、MoE 配置、多模态子配置） | 识别结构相同的模型；无需下载权重 |
| **权重分布哈希** | 前 50 个张量的统计特征（均值、标准差、L2 范数），读取 `.safetensors` 或 `.bin` 文件 | 检测微调、格式转换（fp16↔bf16）或未授权权重修改 |
| **行为 DNA 哈希** | 理论表达能力边界（表达能力因子、路由复杂度、注意力粒度、词表熵） | 模型行为能力的近似度量——无需前向传播 |
| **Vitriol 签名** | `arx_` + 三层哈希的 SHA-256 组合 | 模型追踪和模型市场验证的唯一 16 字符标识 |

支持标准 Transformers 模型和 Diffusers 流水线（`model_index.json` + UNet/VAE/TextEncoder 子组件）。

**编程式 API**（内存中操作，无需文件）：

```python
from vitriol.utils.fingerprint import FingerprintEngine, FingerprintRegistry

engine = FingerprintEngine()
fingerprint = engine.fingerprint(model, model_id="my-model")
# fingerprint.architecture_hash, .weights_hash, .content_hash, .signature

# 对比两个模型
comparison = engine.compare_models(model_a, model_b)
# → {"identical": bool, "same_architecture": bool, "weights_similarity": float}

# 追踪模型版本谱系
registry = FingerprintRegistry("fingerprints.json")
registry.register(model, metadata={"version": "v1.0"})
lineage = registry.get_lineage("my-model")  # 同一架构的所有版本
```

## 量化推理与 KV Cache 压缩

Vitriol 不仅仅生成权重——它实现了完整的**量化推理流水线**，包含 KV Cache 压缩，面向研究、基准测试和部署导向实验。

### TurboQuant（KV Cache 量化）

推理时对注意力计算中的 Key/Value 做 TurboQuant 风格的近似处理，直接 Monkey-patch `F.scaled_dot_product_attention`：

> 注意：`KVRuntimePatcher` 路径属于 **“近似推理 / 解码加速”**，它返回的仍是浮点张量，因此**不会把 KV cache 改造成 bit-packed 存储格式**，显存占用通常不会按 “bits/value” 线性下降。  
> 若你需要"KV 存储压缩"（packed KV + scales/mins 等元数据），请使用 `KVCacheStore` / `CacheHookPatcher` 那条 KVStore 路径（见下文"KV Cache 策略系统"）。

| 格式 | 有效位数 | 字节/值 | 相对 BF16 压缩比 |
|------|:---:|:---:|:---:|
| `turbo2` | 2.5 bits | 0.31 B | **6.4×** |
| `turbo3` | 3.5 bits | 0.44 B | **4.6×** |
| `turbo4` | 4.25 bits | 0.53 B | **3.8×** |

```python
from vitriol.patches.kv_runtime_patches import patch_kv_runtime, KVRuntimePatchConfig

cfg = KVRuntimePatchConfig(
    enable_turbo_quant=True,
    turbo_bits=3.5,              # K 对齐 3-bit、V 对齐 4-bit
    turbo_block_size=32,
    quantized_kv_start=2048,     # 短前缀保持精确，长上下文解码再做量化
)
patcher = patch_kv_runtime(cfg)
# 现在 model.generate() 自动使用量化 KV Cache
print(patcher.stats())  # calls_total, calls_patched, calls_bypassed
```

你也可以使用 `turbo_k_bits` / `turbo_v_bits` 显式指定，或继续使用 `turbo_format="turbo3"` 保持向后兼容。

### Adaptive KV Codec

基于注意力感知的自适应位宽分配——重要的 head/token 分配更多位，其余分配更少：

```python
from vitriol.patches.kv_runtime_patches import patch_kv_runtime, KVRuntimePatchConfig
from vitriol.kv.codec import AdaptiveKVCodec

cfg = KVRuntimePatchConfig(
    enable_adaptive_bits=True,
    adaptive_bits=AdaptiveKVCodec(
        min_bits=3.0, max_bits=5.0, target_avg_bits=3.5,
        k_share=0.65,               # K 获得位宽预算的 65%
        rotate_kurtosis_threshold=10.0,  # 自动 Walsh-Hadamard 旋转
    ),
)
patcher = patch_kv_runtime(cfg)
```

**核心特性：**
- **基于熵的分配**：使用注意力熵确定每个 head 的重要性
- **Walsh-Hadamard 旋转**：当 KV 峰度超过阈值时自动应用 FWHT（高斯化以改善量化效果）
- **逐 head 报告**：返回 `k_bits`、`v_bits` 和压缩统计

### Sparse V（注意力门控 KV 解码）

推理时跳过低注意力 V 块的内存加载，降低显存带宽消耗：

```python
cfg = KVRuntimePatchConfig(
    enable_sparse_v=True,
    sparse_v_threshold=0.01,   # 跳过注意力 < 1% 的 V 位置
)
```

### Compute Skip Attention

分块级注意力重要性评分——跳过 `attn_mass × ‖V‖ < ε × total` 的整个 KV 块：

```python
from vitriol.patches.kv_runtime_patches import patch_kv_runtime, KVRuntimePatchConfig

cfg = KVRuntimePatchConfig(
    enable_compute_skip=True,
    compute_skip=ComputeSkipConfig(block_size=128, epsilon=0.02),
)
```

### 🌟 TurboQuantum：量子增强的 KV Cache 压缩

> **Vitriol 的研究方向** — 将 Google Lab 的 TurboQuant 思路与量子启发的自适应位宽分配相结合。

TurboQuantum 将**注意力分布视为量子波函数**，根据每个 head 的注意力熵动态分配比特数：

| 量子概念 | KV Cache 映射 | 技术实现 |
|----------|--------------|---------|
| 波函数 ψ | Attention softmax | `compute_attention_entropy()` |
| 测量坍缩 | 低熵 → 少比特 | `quantum_bit_allocator()` |
| 叠加态 | 高熵 → 多比特 | 熵阈值 > 0.7 |
| 量子隧穿 | 关键 token 保护 | Top-2% 注意力质量保持全精度 |
| 纠缠 | 跨层误差关联 | `entanglement_residual_sketch()` |

**当前工作假设**：不再使用统一比特宽（如 turbo3 = 3.5 bpv），而是为每个 head **动态分配 1.5–5.0 bits**。这里应理解为实验性特性说明，而不是已经完全证明的论文级结论。

```python
from vitriol.kv.turboquantum import (
    TurboQuantumConfig,
    turboquantum_compress,
)

# 4 种内置模式: conservative / balanced / aggressive / ultra-long
config = TurboQuantumConfig(mode="balanced", target_avg_bits=3.0)
result = turboquantum_compress(q, k, v, config)

# 合成示例结果: 约 81% 存储节省，K 余弦相似度 > 0.87
print(f"有效 BPV: {result.report['effective_bpv']}")
print(f"K 余弦相似度: {result.report['k_cosine']:.4f}")
```

**CLI 使用（合成基准测试无需模型）:**
```bash
# 并排比较所有 4 种模式
vitriol bench turboquantum --compare-modes --format summary

# 在真实模型 KV cache 上运行
vitriol bench turboquantum-model Qwen/Qwen3.5-0.8B --mode balanced

# JSON 格式详细输出
vitriol bench turboquantum --mode aggressive -o tq_results.json
```

**基准测试结果 (8 heads × 256 seq × 128 dim):**

| 模式 | BPV | K 余弦 | V 余弦 | 节省率 |
|------|----:|-------:|-------:|------|
| conservative | 3.00 | 0.881 | 0.889 | 81.25% |
| balanced | 3.00 | 0.880 | 0.881 | 81.25% |
| aggressive | 2.50 | 0.771 | 0.806 | 68.75% |

完整分析报告: [docs/turboquantum_analysis.md](docs/turboquantum_analysis.md)

### KV Cache 策略系统

针对不同部署场景的预配置策略预设：

| 预设 | 策略 | 目标 |
|------|------|------|
| `safe` | 精确 KV cache | 质量优先部署 |
| `balanced` | 带延迟启动的 TurboQuant + 选择性 full-attention V 量化 | 长上下文默认基线 |
| `fast-balanced` | 关闭 residual sketch 的轻量版 balanced TurboQuant | 更快的论文风格 A/B |
| `aggressive` | 更早启用 TurboQuant，并在前几层 full-attention 上启用 Sparse-V | 实验性吞吐调优 |
| `ultra-long` | 面向超长上下文的 TurboQuant + Sparse-V + Compute-Skip | 实验性超长上下文调优 |

```python
from vitriol.kv.policy import KVPolicyPreset, Turbo3ExactKApproxVPolicy

policy = KVPolicyPreset.balanced_default()
# 或精细调优：
# Turbo3ExactKApproxVPolicy(
#     v_quantize_only_first_n_full_attention_layers=4,
#     quantized_kv_start=1024,
#     enable_sparse_v=True,
# )
```


`bench/` 模块提供自动化的 KV Cache 压缩基准测试，附带提示词套件：

```bash
vitriol bench kv-smoke Qwen/Qwen2.5-7B --preset balanced
vitriol bench kv-smoke Qwen/Qwen2.5-7B --preset fast-balanced
vitriol bench kv-smoke Qwen/Qwen2.5-7B --preset balanced --format summary
vitriol bench kv-smoke Qwen/Qwen2.5-7B --preset balanced --compare-preset aggressive --format markdown --output smoke-compare.md
vitriol bench kv-long Qwen/Qwen2.5-7B --preset ultra-long --prompt-tokens 131072
vitriol bench kv-long Qwen/Qwen2.5-7B --preset balanced --compare-preset ultra-long --format markdown --output long-compare.md
vitriol bench kv-suite Qwen/Qwen2.5-7B --preset aggressive --prompt-tokens 2048 --prompt-tokens 8192
vitriol bench kv-suite Qwen/Qwen2.5-7B --preset balanced --compare-preset ultra-long --format summary
vitriol bench kv-report Qwen/Qwen2.5-7B --preset balanced --compare-preset ultra-long --format markdown --output kv-report.md
vitriol bench kv-report Qwen/Qwen2.5-7B --preset balanced --compare-preset ultra-long --output-dir ./bench-artifacts
vitriol bench kv-plan Qwen/Qwen2.5-7B --preset balanced --compare-preset ultra-long --format summary --show-layers
vitriol bench kv-plan Qwen/Qwen2.5-7B --preset balanced --format json --output plan.json
vitriol bench kv-plan Qwen/Qwen2.5-7B --preset balanced --format markdown --output plan.md
vitriol bench kv-analyze Qwen/Qwen2.5-7B --preset balanced --compare-preset fast-balanced --prompt-tokens 1024 --format summary
vitriol bench kv-analyze Qwen/Qwen2.5-7B --preset balanced --compare-preset fast-balanced --prompt-tokens 1024 --format summary --show-layers
vitriol bench kv-analyze Qwen/Qwen2.5-7B --preset balanced --compare-preset fast-balanced --prompt-tokens 1024 --format summary --show-layers --sort-by logits_mse_delta
vitriol bench kv-analyze Qwen/Qwen2.5-7B --preset balanced --compare-preset fast-balanced --format markdown --output kv-analyze.md
vitriol bench kv-suite Qwen/Qwen2.5-7B --preset aggressive --format markdown --preset-param quantized_kv_start=1024 --output suite.md
```

`markdown` 导出现在会自动附带实验元信息头，包括生成时间、命令类型、输出路径、preset 覆盖参数，以及关键 benchmark 参数，便于把导出的 `.md` 直接作为实验记录保存。

`kv-suite --compare-preset ...` 会对同一组 prompt suite 连续运行两套 preset，并输出统一 diff 报告，里面同时包含逐 case 的速度差值和逐层策略变化。

`kv-long --compare-preset ...` 则面向单个长上下文样本生成 A/B 报告，适合在跑完整 suite 之前先做聚焦验证。

`kv-smoke --compare-preset ...` 则把同样的 A/B 流程带到了最快速的 sanity check 路径，便于你先筛选 preset，再决定是否继续跑更长的基准。

`kv-report` 会把 `kv-smoke`、`kv-long`、`kv-suite` 三段对比整合成一份统一报告，适合一次性产出某个模型与 preset 组合的实验快照。

`kv-report --output-dir ...` 会在目标目录里同时写出 `report.json` 和 `report.md`，适合把同一次运行同时沉淀成结构化结果和可读实验记录。

`kv-analyze` 会基于一次 prefill 生成的 KV cache 做离线量化误差分析，不需要等待完整 decode benchmark，就能直接查看逐层和平均的 `MSE`、余弦相似度、代理 attention logits 漂移、代理 attention 输出漂移，以及 residual correction gain。

`kv-analyze --show-layers` 会额外输出量化层表格，方便直接看出哪些 full-attention 层从 residual sketch 或 preset 变化里获益最大。

`kv-analyze --sort-by logits_mse_delta` 会按 compare preset 相对 base preset 增加了多少代理 attention logits 漂移来排序层级，非常适合快速定位 residual sketch 最重要的层。

常用输出字段：

- `preset.name`：当前生效的 preset 名称
- `chosen_v_quantize_only_first_n`：最终启用 V 量化的 full-attention 层数量
- `policy_insights.quantized_kv_start`：开始进入量化 KV 的 token 位置
- `policy_insights.counts`：按层类型和策略命中统计的汇总计数
- `policy_insights.layers`：每一层的 `turbo_k`、`turbo_v`、`sparse_v`、`compute_skip` 决策
- `estimated_kv_mb`：当前路径的 KV-only 内存估算
- `peak_device_mb`：整次运行的设备峰值内存
- `peak_minus_estimated_mb`：峰值内存中非 KV 开销的近似差值
- `results[]`：`kv-suite` 的逐 case 基准结果
- `delta_speedup`：`kv-long --compare-preset` 的整体速度差值
- `case_diffs[]`：`kv-suite --compare-preset` 的逐 case 对比结果
- `changed_layers[]`：`kv-plan --compare-preset` 的逐层策略差异
- `smoke` / `long` / `suite`：`kv-report` 返回的分段对比结果
- `base.summary` / `compare.summary`：`kv-analyze` 返回的平均离线 KV 误差指标
- `layers[]`：`kv-analyze` 返回的逐层离线 KV 误差结果

```python
from vitriol.bench.runner import default_prompt_suite, prefix_match_tokens
# 自动生成校准提示词，测量质量与压缩的权衡
```

### 推理

现在可以直接用和 benchmark 同一套 TurboQuant preset 路径做单条 prompt 推理：

```bash
# 将 <model_path> 替换为本地模型目录或 HuggingFace 模型 ID
vitriol infer <model_path> \
  --prompt "用一句话解释 TurboQuant。" \
  --preset balanced \
  --preset-param quantized_kv_start=0

vitriol infer <model_path> \
  --prompt "用一句话解释 TurboQuant。" \
  --preset balanced \
  --preset-param quantized_kv_start=0 \
  --format summary \
  --show-stats

vitriol infer <model_path> \
  --prompt-file ./prompt.txt \
  --preset fast-balanced \
  --format summary
```

## PPL 困惑度评估框架

`ppl_evaluator.py` 模块提供**基于真实困惑度的量化推理效果评估**，替代旧的代理指标（MSE、余弦相似度）：

**核心指标：**

| 指标 | 描述 |
|------|------|
| **Perplexity (PPL)** | `exp(平均负对数似然)` — 常用的端到端语言模型质量指标 |
| **Token Match Rate** | 与基线的精确匹配 / 前缀匹配百分比 |
| **Logit KL Divergence** | 逐层输出分布偏移量测量 |
| **KV Memory Estimate** | 基准路径报告的 KV-only 内存估算 |
| **Device Peak Memory** | 整次运行的设备峰值内存，包含非 KV 开销 |
| **Throughput** | 量化前后的 tokens/sec 吞吐量 |

```python
from vitriol.bench.ppl_evaluator import PPLEvaluator, PPLConfig

config = PPLConfig(model_id="Qwen/Qwen2.5-1.5B", max_new_tokens=64)
evaluator = PPLEvaluator(config)
results = evaluator.evaluate(kv_preset_override="balanced")
print(results.report())
```

**架构**: 基线 (无量化) → 生成 token → 对比 ← 调优后 (KV 量化) → 生成 token

**兼容契约**：调优分支与 `vitriol bench` 使用同一套 preset 解析路径，并通过 `vitriol.bench.runner._apply_vitriol_universal(..., v_quantize_only_first_n_layers=...)` 应用 KV hook。这样可以避免 hook 签名演进后，PPL 评估静默退化为“未压缩解码”。

**推荐回归检查：**

```bash
python -m pytest tests/test_ppl_evaluator.py -q
python -m pytest tests/test_cli_bench.py tests/test_cli_infer.py -q
```

### Triton GPU 加速内核

`kv/triton_kernels.py` 为 KV Cache 操作提供高性能 GPU 内核：

| 内核 | 功能 | 加速比 |
|------|------|--------|
| `triton_fwht` | 快速 Walsh-Hadamard 变换（O(n log n) 并行） | 10–50× |
| `triton_blockwise_quantize` | 分块 min-max 量化，全向量化 | 5–20× |
| `triton_pack` / `triton_unpack` | 亚字节位打包/解包（turbo 格式） | 5–15× |

所有内核自动检测 Triton 可用性，未安装 Triton 时回退到优化后的 PyTorch 实现。

## 权重生成策略

13 种策略覆盖不同研究和工程场景：

| 策略 | CLI 参数 | 原理 | 文件大小 | 适用场景 |
|------|----------|------|----------|----------|
| **Random** | `random` | 标准正态分布随机初始化 | 大 (≈原始) | 训练测试、梯度验证 |
| **Compact** | `compact` | 零填充 + 张量缓存 | 极小 | 加载测试、CI/CD |
| **Ultra** | `ultra` | Strided 张量 (stride=0) | 最小 | 存储敏感场景 |
| **Sparse** | `sparse` | 稀疏张量 | 较小 | 稀疏性研究 |
| **Structured Sparse** | `structured_sparse` | 结构化稀疏模式 | 较小 | 剪枝研究 |
| **Ternary** | `ternary` | 三值 (-1, 0, +1) | 小 | 量化研究 |
| **Binary** | `binary` | 二值 (±1) | 小 | 极端量化研究 |
| **Quantized** | `quantized` | INT8/FP8 量化 | 中等 | 量化部署测试 |
| **LowRank** | `lowrank` | 低秩矩阵分解 | 较小 | 压缩研究 |
| **Learned** | `learned` | 神经网络生成权重 | 中等 | 学习型压缩研究 |
| **Hybrid Learned** | `hybrid_learned` | 注意力/嵌入用 learned，其余用 compact | 小-中等 | 兼顾效率与学习性 |
| **Quantum** | `quantum` | 量子启发式策略 | 极小 | 量子计算探索 |

## 架构分析器

**10 种专用分析器**，覆盖当前主流 LLM 架构：

| 分析器 | 目标模型 | 特殊能力 |
|--------|---------|----------|
| TransformerAnalyzer | 通用 Transformer（LLaMA、Mistral 等） | GQA/MQA 识别、RoPE 检测 |
| QwenAnalyzer | Qwen 系列 | Qwen 特有配置处理 |
| DeepSeekAnalyzer | DeepSeek-V3 | MLA（多头潜在注意力）、Hybrid Dense+MoE |
| KimiAnalyzer | Kimi K2.5 | DeepSeek-V3 架构变体 |
| GLMAnalyzer | GLM-5 (MoE+DSA) | Hybrid MLP（Dense+Sparse 逐层切换） |
| ErnieAnalyzer | ERNIE 4.5 VL | Vision Encoder + MoE + 3D-RoPE |
| GPT2Analyzer | GPT-2 | 绝对位置编码、Conv1D 实现 |
| MiniMaxAnalyzer | MiniMax-M2.5 | MTP（多 Token 预测）、Hybrid Attention |
| InternS1Analyzer | Intern-S1-Pro | 三模态（Text+Vision+TimeSeries） |
| **Qwen35Analyzer** | **Qwen3.5 MoE (A3B/A17B)** | **Linear / Full 注意力分层检测、视觉编码器、Shared Expert** |

## 模型库 (Demo 配置)

Vitriol 内置 **3 个演示模型配置**，存放在 `docs/data/` 目录，可直接用于可视化和测试——无需下载真实权重：

| 模型 | 类型 | 架构 | 关键特征 |
|------|------|------|---------|
| **Qwen3.5-397B-A17B** | 多模态 MoE | `Qwen3_5MoeForConditionalGeneration` | 8 专家（2 激活），27 层视觉编码器，2 层文本，MLA |
| **Qwen3 Demo** | Dense Transformer | `Qwen2ForCausalLM` | 24 层，GQA (16H/8KV)，RoPE，hidden=2048 |
| **DeepSeek V3 Demo** | MoE | `DeepseekV3ForCausalLM` | 32 层，64 专家（6 激活），GQA (32H/8KV) |

> 每个演示仅包含一个 `config.json`（约 400B–3.6KB），可在 [3D 查看器](https://islinxu.github.io/Vitriol/viewer.html) 中即时查看。

**添加自定义模型：** 将任意 `config.json` 放入 `docs/data/<name>/`，访问 `viewer.html#?model=data/<name>` 即可。

```bash
# 本地快速预览
cd docs && python3 -m http.server 8000
# 打开 http://localhost:8000/viewer.html#?model=data/qwen3-demo
```

## 3D 可视化

Vitriol 提供**浏览器优先的 3D 架构查看器**，基于 Three.js + WebGL 构建——加载配置后无需 Python 后端。

**在线演示：** [isLinXu.github.io/Vitriol/viewer.html](https://islinxu.github.io/Vitriol/viewer.html)

### 功能特性

| 功能 | 描述 |
|------|------|
| **3D 模型探索** | 旋转、缩放、平移——每层以 3D 方块渲染，按类型着色 |
| **3D 数据流管道** | 二次贝塞尔曲线 + 动态粒子动画，展示层间张量流动 |
| **2D/3D 切换** | 平面分层图与空间 3D 视图无缝切换 |
| **右键菜单** | 折叠/展开分组、隔离子模块、跳转到指定层 |
| **键盘快捷键** | `R` 重置相机，`F` 聚焦选中，`H` 帮助，`Esc` 取消选择 |
| **搜索定位** | 按名称快速跳转到任意层（attention、mlp、embed、norm…） |
| **悬停提示** | 显示 shape、dtype、params、heads、KV heads 等技术细节 |
| **模型对比** | 多个架构并排对比 |
| **导出** | 截图 (PNG) 和完整配置 (JSON) 下载 |

### 命令行

```bash
# 生成独立 HTML 可视化文件
python -m vitriol.cli.main viz /path/to/model --output arch_viz.html

# 3D 模式（WebGL）——在浏览器中打开
python -m vitriol.cli.main viz /path/to/model --3d

# 简写
python -m vitriol.cli.main arch-viz /path/to/model
```

## 演示与截图

### 3D 架构查看器

<p align="center">
  <img src="docs/images/screenshot_3d_viewer.png" alt="3D 架构查看器" width="720">
</p>

> 交互式 3D MoE 模型探索——旋转、缩放、检查每一层。

### 架构分析报告

```bash
python -m vitriol.cli.main analyze /path/to/Qwen3.5-397B-A17B/config.json
```

示例输出：

```
╔══════════════════════════════════════════════════════════════╗
║  架构分析报告                                                ║
╠══════════════════════════════════════════════════════════════╣
║  模型: Qwen3.5-397B-A17B                                    ║
║  类型: MoE（多模态）                                         ║
║  总参数: 397.6B  激活参数: 35.6B                             ║
║  层数: 29  隐藏维度: 8192  注意力头: 32  KV头: 4              ║
╠══════════════════════════════════════════════════════════════╣
║  MoE: 8 个专家，每次激活 2 个                                  ║
║  注意力: MLA（多头潜在注意力）                                 ║
║  位置编码: RoPE                                              ║
╠══════════════════════════════════════════════════════════════╣
║  性能模拟 (H100, fp16):                                      ║
║  推理: ~12.3 tok/s  训练: ~1,850 tok/s                       ║
║  显存: ~72 GB (全精度) / ~18 GB (4-bit 量化)                  ║
╚══════════════════════════════════════════════════════════════╝
```

### NAS 基准测试

```bash
# 随机搜索
python -m vitriol.cli.main nas random --trials 50 --objective efficiency

# 进化搜索
python -m vitriol.cli.main nas evolutionary --generations 20 --population 30
```

### 权重生成立演示

```bash
# 为 72B 模型生成超紧凑权重——输出仅 ~3KB
python -m vitriol.cli.main generate Qwen/Qwen2.5-72B --strategy ultra --output ./tiny-qwen-72b

# 生成并一步验证
python -m vitriol.cli.main generate Qwen/Qwen2.5-7B --strategy compact --validate
```

### Hy3 Preview 实操流程

```bash
# 使用 ultra 策略导出 Tencent Hy3 preview
python -m vitriol.cli.main --trust-remote-code generate tencent/Hy3-preview --strategy ultra --output output/hy3_preview_ultra_final

# 基于导出目录刷新架构可视化产物
python -m vitriol.cli.main --trust-remote-code arch-viz output/hy3_preview_ultra_final --html --output output/hy3_preview_ultra_final/architecture.html
python -m vitriol.cli.main --trust-remote-code arch-viz output/hy3_preview_ultra_final --block --output output/hy3_preview_ultra_final/architecture.png
python -m vitriol.cli.main --trust-remote-code arch-viz output/hy3_preview_ultra_final --detail --output output/hy3_preview_ultra_final/architecture_detail.png

# 执行一次最小本地生成烟测
python -m vitriol.cli.main --trust-remote-code infer output/hy3_preview_ultra_final --prompt "Hello" --preset safe --max-new-tokens 12 --format summary --show-stats
```

> `ultra` 导出的是最小壳权重。只要本地生成成功，就说明 tokenizer 和模型链路可加载；但生成文本质量不代表原始模型能力。

> **提示：** 所有可视化功能无需下载真实权重。`docs/data/` 中的演示配置即可体验完整功能。

## 架构进化工具

```bash
# 查看已知模型家族
python -m vitriol.cli.main evolve families

# 生成进化树
python -m vitriol.cli.main evolve tree -o output/evolution_tree.html

# 对比两个模型
python -m vitriol.cli.main evolve compare Qwen/Qwen2.5-7B DeepSeek-V3/DeepSeek-V3

# 性能模拟
python -m vitriol.cli.main evolve simulate Qwen/Qwen2.5-72B --gpu H100

# 创新时间线
python -m vitriol.cli.main evolve timeline

# 架构推荐
python -m vitriol.cli.main evolve recommend --use-case chat --max-vram 24
```

## NAS（神经架构搜索）

```bash
# 随机搜索
python -m vitriol.cli.main nas --algorithm random --iterations 20

# 进化算法 + 数据集
python -m vitriol.cli.main nas \
    --algorithm evolutionary \
    --generations 10 --population 20 \
    --dataset wikitext --dataset-config wikitext-2-v1 \
    --n-samples 100

# 定向搜索（约束优化）
python -m vitriol.cli.main nas --algorithm targeted --target-vram 24
python -m vitriol.cli.main nas --algorithm targeted --target-params 70 --objective maximize-efficiency

# 强化学习代理（实验性）
python -m vitriol.cli.main nas --algorithm rl --episodes 50

# 断点续搜
python -m vitriol.cli.main nas --algorithm evolutionary --output-dir output/nas_evo --resume
```

**4 种搜索算法：**

| 算法 | 类名 | 描述 |
|------|------|------|
| 随机搜索 | `RandomSearcher` | 均匀随机采样搜索空间 |
| 进化搜索 | `EvolutionarySearcher` | 遗传算法（交叉+变异） |
| 定向搜索 | `TargetedNASEvaluator` | 约束优化 + 多目标帕累托最优 |
| **强化学习** | **`RLSearcher`** | **基于 RL 的架构搜索（实验性）** |

NAS 搜索空间为所有搜索器提供稳定兼容层：

- `ArchitectureGene.to_config()` 生成 HuggingFace 风格配置字典。
- `ArchitectureGene.from_config()` 在 RL/controller 修改配置后重建 gene。
- `LLMSearchSpace.sample_random()` 是面向 RL 的随机采样别名。
- `LLMSearchSpace.validate_gene()` 拒绝超出离散搜索空间的 gene。

修改 NAS 内部后建议运行：

```bash
python -m pytest tests/test_nas_rl_compat.py tests/test_cli_nas_rl.py -q
```

## 压缩智能度评分 (CIS)

基于「压缩即智能」理论的多维度评估框架：

```
Ψ(S) = α·η_info + β·η_storage + γ·η_express + δ·T_train
```

四个维度：信息保留、存储效率、表达能力、可训练性。

```python
from vitriol.metrics import CompressionIntelligenceScorer, generate_score_comparison_table

scorer = CompressionIntelligenceScorer()
scores = scorer.score_all_strategies()
table = generate_score_comparison_table()
```

## Web UI

通过 Gradio 启动图形界面：

```bash
python -m vitriol.cli.main webui
python -m vitriol.cli.main webui --port 8080 --share
```

功能：模型对比、进化树、定向 NAS、性能模拟、架构评分卡。

## REST API（实验性）

通过 FastAPI 提供编程式访问：

```bash
pip install -e ".[api]"
python -m vitriol.api.server
```

端点：模型生成、架构搜索、模型分析、系统监控。

## GitHub Pages

`docs/` 目录下内置可直接部署到 GitHub Pages 的静态站点：

- 首页：`docs/index.html`
- 3D 查看器：`docs/viewer.html`
- 模型索引：`docs/viz-models/`
- 词表索引：`docs/vocab-viz/`

查看器支持通过 hash 选择模型：

```text
viewer.html#?model=data/Qwen3.5-397B-A17B-Vitriol-ultra-dummy
viewer.html#?model=data/qwen3-demo
viewer.html#?model=data/deepseek-demo
```

本地预览：

```bash
cd docs && python3 -m http.server 8000
```

## Python API

```python
from vitriol import MinimalWeightGenerator
from vitriol.config.manager import GenerationConfig

config = GenerationConfig(strategy="compact", max_shard_size="2GB")
generator = MinimalWeightGenerator(
    model_id="Qwen/Qwen2.5-0.5B",
    output_dir="output/qwen-compact",
    config=config,
)
generator.generate()
```

## 配置系统

三层配置体系（优先级：CLI 参数 > YAML 文件 > 环境变量）：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_shard_size` | str | `"5GB"` | 单个分片最大体积 |
| `dtype` | str | `"bfloat16"` | 权重数据类型 |
| `strategy` | str | `"random"` | 生成策略名称 |
| `auto_validate` | bool | `true` | 生成后自动验证 |
| `n_bits` | int | `8` | 量化位数 (quantized 策略) |
| `rank` | int | `16` | 矩阵秩 (lowrank 策略) |
| `sparsity` | float | `0.5` | 稀疏率 (structured_sparse 策略) |

## 项目结构

```
Vitriol/
├── src/vitriol/                         # 核心源码 (140+ Python 文件)
│   ├── __init__.py                     # 版本 (v0.3.0), 惰性导入
│   ├── core/                           # 核心引擎
│   │   ├── generator.py                # MinimalWeightGenerator 主生成器
│   │   ├── validator.py                # 模型验证器 (加载/推理/内存测试)
│   │   ├── analyzer.py                 # 架构分析器
│   │   ├── batch.py                    # 批量生成
│   │   ├── exporter.py                 # 模型导出
│   │   ├── hasher.py                   # 模型哈希指纹
│   │   ├── incremental.py              # 断点续传 / Checkpoint
│   │   ├── parallel_generator.py       # 并行生成
│   │   ├── adaptive_sharder.py         # 自适应分片
│   │   ├── smart_initializer.py        # 智能初始化
│   │   ├── shard_manager.py            # 分片管理
│   │   ├── config_processor.py         # 配置处理
│   │   └── pipeline/                   # Pipeline 生成器
│   │       ├── pipeline.py             # Pipeline 编排
│   │       ├── context.py              # Pipeline 上下文
│   │       └── steps/                  # Pipeline 步骤
│   ├── strategies/                     # 13 种权重生成策略
│   │   ├── base.py                     # WeightGenerationStrategy 抽象基类
│   │   ├── random.py                   # 随机初始化
│   │   ├── compact.py                  # 紧凑压缩
│   │   ├── ultra.py                    # 极致压缩 (stride=0)
│   │   ├── sparse.py                   # 稀疏策略
│   │   ├── structured_sparse.py        # 结构化稀疏
│   │   ├── ternary.py                  # 三值量化 (-1, 0, +1)
│   │   ├── binary.py                   # 二值量化 (±1)
│   │   ├── quantized.py                # INT8/FP8 量化
│   │   ├── lowrank.py                  # 低秩分解
│   │   ├── learned.py                  # Learned + HybridLearned
│   │   └── quantum.py                  # 量子启发式
│   ├── arch_viz/                       # 架构可视化引擎
│   │   ├── analyzers.py                # 10 种专用分析器
│   │   ├── visualizer.py               # 可视化入口
│   │   ├── core.py / parser.py         # 数据结构 & 配置解析
│   │   └── renderers/                  # Block, Detail, HTML 渲染器
│   ├── nas/                            # 神经架构搜索
│   │   ├── search_space.py             # ArchitectureGene 搜索空间
│   │   ├── searcher.py                 # 随机 / 进化 / RL 搜索
│   │   ├── evaluator.py                # 混合评估器 (Zero-Cost + Few-Shot)
│   │   ├── targeted_nas.py             # 约束优化 + 多目标 NAS
│   │   ├── rl_agent.py                 # 强化学习代理（实验性）
│   │   └── controller.py               # NAS 控制器
│   ├── evolution/                      # 架构进化模块 (v0.4.0)
│   │   ├── tree_builder.py             # EvolutionTree, ArchNode
│   │   ├── tree_visualizer.py          # D3.js HTML 可视化
│   │   ├── compare.py                  # 架构对比器
│   │   ├── simulator.py                # VRAM/FLOPs/延迟估算
│   │   ├── recommender.py              # 架构推荐器
│   │   └── timeline.py                 # 创新时间线
│   ├── metrics/                        # 压缩智能度
│   │   └── compression_intelligence.py # CIS 评分框架
│   ├── adapters/                       # 模型适配器 (自动发现)
│   │   ├── base.py                     # ModelAdapter 抽象基类
│   │   ├── registry.py                 # AdapterRegistry 注册中心
│   │   ├── llama.py                    # LLaMA / Mistral 适配器
│   │   ├── qwen.py                     # Qwen 系列适配器
│   │   └── deepseek.py                 # DeepSeek 适配器
│   ├── patches/                        # 兼容性补丁 (10 个模块)
│   │   ├── transformers_patches.py     # 通用 transformers 补丁
│   │   ├── model_family_patches.py     # 模型家族注册表
│   │   ├── kv_runtime_patches.py       # KV 运行时补丁
│   │   ├── cache_hooks.py              # 缓存钩子补丁
│   │   ├── qwen35_*.py                 # Qwen3.5 专用补丁（KV Store、缓存、注意力）
│   │   ├── turboquant.py               # TurboQuant 补丁
│   │   └── ...                         # detectron2 mock, 动态模型补丁等
│   ├── kv/                             # KV Cache 系统 (7 个模块)
│   │   ├── backend.py / codec.py       # 后端 & AdaptiveKVCodec 编解码
│   │   ├── cache_store.py              # 缓存存储 (L1/L2/L3)
│   │   ├── policy.py                   # KVPolicyPreset 淘汰策略
│   │   └── triton_kernels.py           # Triton GPU 加速内核（FWHT、量化、位打包）
│   ├── cli/                            # CLI (18 个命令)
│   │   ├── main.py                     # Click Group 入口
│   │   └── commands/                   # 子命令实现（含 bench、infer）
│   ├── webui/                          # Gradio Web UI
│   │   └── app.py                      # Gradio 应用
│   ├── api/                            # REST API（实验性）
│   │   └── server.py                   # FastAPI 服务端
│   ├── viz/                            # 可视化模板
│   │   ├── dashboard.py                # 可视化仪表板
│   │   └── *.html                      # 4 个 HTML 可视化器 (3D)
│   ├── config/                         # 配置管理
│   │   ├── manager.py                  # GenerationConfig
│   │   └── settings.py                 # 应用设置
│   ├── bench/                          # 基准测试
│   │   ├── runner.py                   # 基准测试运行器（KV Cache 预设）
│   │   ├── autokv.py                   # AutoKV 基准
│   │   └── ppl_evaluator.py            # 困惑度评估框架（PPL、Token 匹配、KL 散度）
│   ├── distributed/                    # 分布式生成
│   │   └── coordinator.py              # 任务协调器
│   ├── plugins/                        # 插件系统
│   │   └── base.py                     # 插件基类
│   ├── resilience/                     # 容错机制
│   │   └── checkpoint.py               # 检查点管理
│   ├── ai/                             # AI 驱动功能
│   │   └── recommender.py              # AI 推荐器
│   ├── registry/                       # 模型注册表
│   │   └── model_store.py              # 模型存储
│   ├── tools/                          # 独立工具
│   │   └── comparator.py               # 架构对比工具
│   └── utils/                          # 工具模块
│       ├── logging.py / logger.py      # 日志
│       ├── exceptions.py               # 自定义异常
│       ├── fingerprint.py              # 模型指纹
│       └── config_cache.py             # 配置缓存
├── docs/                               # GitHub Pages 静态站点
│   ├── index.html                      # 项目展示首页
│   ├── viewer.html                     # 3D 模型查看器
│   ├── data/                           # 演示模型配置
│   └── manifests/                      # 可视化 & 词表清单
├── scripts/                            # 工具脚本
│   ├── install_dev_cpu.sh              # CPU 开发环境安装
│   ├── prepare_test_assets.py          # 测试资产准备
│   └── sync_github_pages_assets.py     # Pages 资产同步
├── pyproject.toml                      # 项目元数据与依赖
├── requirements.txt                    # pip 依赖
└── README.md                           # 英文文档
```

## 与同类工具的对比

| 工具 | 最小权重生成 | 架构可视化 | LLM NAS | LLM 语义理解 | 生态位 |
|------|:---:|:---:|:---:|:---:|------|
| **Vitriol** | ✅ 13 种策略 | ✅ 10 种分析器 | ✅ 4 种算法 | ✅ MoE/GQA/MLA | LLM 架构探索专用 |
| HuggingFace Transformers | ❌ | ❌ | ❌ | ✅ | 模型训练/推理框架 |
| `torch.nn.utils.skip_init` | 部分 (仅跳过 init) | ❌ | ❌ | ❌ | PyTorch 底层工具 |
| NNI / AutoGluon | ❌ | ❌ | ✅ (面向 CV) | ❌ | 通用 NAS 框架 |
| Netron | ❌ | ✅ (通用) | ❌ | ❌ | 通用模型可视化 |
| vLLM / FlexGen | ❌ | ❌ | ❌ | ✅ | 推理优化框架 |

> **Vitriol 的独特生态位**：开源社区中唯一同时提供「LLM 最小权重生成 + 架构可视化 + NAS」三合一能力的工具平台。

## 实际价值与成本节约

### 💾 存储成本节约

Vitriol 的核心价值在于将 GB 级模型下载转化为 MB/KB 级操作。以下是具体对比：

| 模型 | 原始大小 | Compact 策略 | Ultra 策略 | **节约** |
|------|:---:|:---:|:---:|:---:|
| Qwen2.5-0.5B | ~1 GB | ~200 MB | ~100 KB | **90%–99.99%** |
| LLaMA-3-8B | ~16 GB | ~3.2 GB | ~1.6 MB | **90%–99.99%** |
| Qwen2.5-72B | ~144 GB | ~28.8 GB | ~14.4 MB | **90%–99.99%** |
| DeepSeek-V3 (671B) | ~1.3 TB | ~260 GB | ~130 MB | **90%–99.99%** |
| Qwen3.5-397B-A17B | ~756 GB | ~151 GB | ~75.6 MB | **90%–99.99%** |

> **Quantum 策略**可实现最高 **99.22%** 的压缩率（float32 的 1/128），支持逐层自适应位宽。**Compact 策略**通过零填充张量 + zip/gzip 实现 **99%** 压缩。**Ultra 策略**利用 stride=0 技巧实现 **99.99%** 压缩（1 个浮点数代表任意大小张量）。

### ⏱️ 时间成本节约

| 任务 | 不使用 Vitriol | 使用 Vitriol | 加速比 |
|------|:---:|:---:|:---:|
| 下载 72B 模型权重 | ~2–4 小时 (144 GB) | ~5 秒 (仅配置) | **~2,000–3,000×** |
| 下载 397B 模型权重 | ~10–20 小时 (756 GB) | ~5 秒 (仅配置) | **~7,000–14,000×** |
| 探索门控模型的架构 | 数小时 (下载 + 环境搭建) | 数秒 (配置获取) | **即时** |
| 测试新模型的加载流程 | 先下载完整权重 | 生成最小权重 | **分钟级 vs 小时级** |
| CI/CD 流水线每次测试 | 每次运行都下载 + 加载 | 生成 compact 一次，缓存复用 | **10–100× 更快** |

### 🧠 研究与工程价值

1. **零成本架构探索**：研究人员无需下载 TB 级权重，即可研究 DeepSeek-V3 的 MLA（多头潜在注意力）、MoE 路由机制或 Qwen3.5 的混合架构。`analyze` 命令 + **10 种专用分析器**可自动识别 GQA/MQA/MLA、RoPE、MoE、多模态组件——**含 Qwen3.5 的 Linear/Full 注意力分层检测**。

2. **神经架构搜索 (NAS)**：内置 NAS 模块支持 **4 种算法**（随机搜索、进化算法、定向搜索、**强化学习代理**），在 VRAM/FLOPs 约束下搜索最优架构——无需真实权重。`targeted_nas` 支持约束优化（如"在 24 GB 显存下找最优架构"）。`rl_agent` 通过 `nas --algorithm rl --episodes N` 提供实验性的强化学习架构搜索。

3. **性能模拟**：`evolve simulate` 命令可估算任意架构在 H100/A100/V100 GPU 上的 VRAM、FLOPs、延迟和吞吐量，支持 MoE 活跃参数追踪、GQA/MQA/MLA FLOPs 公式和 KV Cache 大小估算。无需 GPU——纯解析式计算。

4. **训练流水线验证**：使用 `random` 策略生成的模型支持完整梯度反向传播，无需昂贵的真实权重即可验证训练循环、数据加载器和梯度累积逻辑。

5. **「压缩即智能」框架**：CIS（压缩智能度评分）提供 4 维评估：Ψ(S) = α·η_info + β·η_storage + γ·η_express + δ·T_train，量化信息保留、存储效率、表达能力和可训练性。

6. **教育与可复现性**：架构可视化（HTML + 3D）、进化树和创新时间线使 LLM 架构的演进（从 GPT-2 到现代 MoE 模型）易于教学和复现。

### 💰 成本估算（云 GPU 场景）

| 场景 | 不使用 Vitriol | 使用 Vitriol | 节约 |
|------|:---:|:---:|:---:|
| 存储 (S3, 100 个模型 × 平均 72B) | ~14.4 TB/月 × ¥0.17/GB = **¥2,448/月** | ~28.8 GB (compact) = **¥4.9/月** | **99.8%** |
| 带宽 (每天下载 10 个模型) | ~1.44 TB/天 × ¥0.65/GB = **¥936/天** | ~2.88 GB/天 = **¥1.87/天** | **99.8%** |
| GPU 时间 (流水线测试) | 1× A100 80GB × 8h = **¥90/次** | 仅 CPU，30 秒 = **≈¥0** | **≈100%** |
| CI/CD (每天 50 个模型测试) | 50 × ¥90 = **¥4,500/天** | 仅 CPU = **≈¥0** | **≈100%** |

## 测试

```bash
# 快速、离线优先的测试套件
python -m pytest -m "not slow and not network" tests/ --ignore=tests/integration -v
```

NAS/PPL 兼容层的重点发布检查：

```bash
python -m pytest tests/test_ppl_evaluator.py tests/test_nas_rl_compat.py tests/test_cli_nas_rl.py -q
python -m ruff check src/vitriol/bench/ppl_evaluator.py src/vitriol/nas/search_space.py src/vitriol/nas/controller.py src/vitriol/cli/commands/nas.py tests/test_ppl_evaluator.py tests/test_nas_rl_compat.py tests/test_cli_nas_rl.py
```

大型外部测试资产（可选）：参见 `docs/TEST_ASSETS.md`。完整发布门禁：参见 `docs/release-validation.md`。

## 发布到 GitHub

1. 推送代码到 GitHub
2. 启用 Pages：`Settings → Pages → Source: GitHub Actions`
3. 更新公开地址：
   - 仓库：`https://github.com/isLinXu/Vitriol`
   - Pages：`https://isLinXu.github.io/Vitriol/`

## 常见问题 (FAQ)

<details>
<summary><strong>Q: 生成的模型能用于推理吗？</strong></summary>

可以运行推理 (forward pass)，但输出是无意义的随机值。生成的模型主要用于验证模型结构、测试加载逻辑和评估显存占用，**不能用于实际的文本生成**。
</details>

<details>
<summary><strong>Q: 生成的模型能用于训练吗？</strong></summary>

是的。使用 `random` 策略生成的权重支持完整的梯度反向传播，可以用于测试训练流程是否正常运行。
</details>

<details>
<summary><strong>Q: 支持哪些模型架构？</strong></summary>

支持所有基于 HuggingFace `transformers` 的模型架构。内置适配器针对 LLaMA、Qwen、DeepSeek 做了专项优化。**10 种专用分析器**覆盖 GQA、MQA、MLA、MoE、**含 Qwen3.5 MoE** 等多模态等。
</details>

<details>
<summary><strong>Q: 如何处理需要登录才能访问的模型 (如 Llama-2)？</strong></summary>

请先运行 `huggingface-cli login`。Vitriol 通过 `huggingface_hub` 访问模型配置。
</details>

<details>
<summary><strong>Q: 生成 70B+ 模型需要多少内存？</strong></summary>

几乎为零——Vitriol 使用 Meta Device 构建模型骨架，内存仅在权重张量生成和分片写入时消耗。
</details>

## 贡献指南

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/amazing-feature`
3. 编写代码并添加测试
4. 确保测试通过：`python -m pytest tests/ -v`
5. 提交更改：`git commit -m 'feat: add amazing feature'`
6. 推送分支：`git push origin feature/amazing-feature`
7. 创建 Pull Request

## 许可证

本项目基于 [MIT License](LICENSE) 开源。

---

<p align="center">
  <sub>Made with care for the LLM research community</sub><br>
  <sub><a href="README.md">English Version</a></sub>
</p>
