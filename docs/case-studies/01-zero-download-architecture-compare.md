# Case Study 01: 零下载 70B 架构对比

> **目标**：在不下载真实权重的前提下，对比两个大模型的架构差异，并生成可分享的 HTML 报告。  
> **适用读者**：架构研究员、MLOps 工程师、技术传播  
> **所需环境**：Python 3.9+、`pip install -e ".[viz]"`、网络可访问 HuggingFace（或本地 config 目录）

---

## 背景

当 LLaMA-70B 与 Qwen-72B 同台竞技时，常见问题是：性能差距来自架构还是训练？传统流程需要先下载 **140GB+** 权重才能加载模型。Vitriol 的 **Structure–Data Decoupling** 让你只下载 **KB 级 config**，即可完成架构分析与可视化。

---

## 方法一：一键 Golden Path（推荐）

```bash
# 模型 A：结构检查 + 报告
vitriol check meta-llama/Llama-3.1-70B -o report/llama-70b --fast

# 模型 B
vitriol check Qwen/Qwen2.5-72B -o report/qwen-72b --fast
```

`--fast` 跳过推理验证与权重分布哈希，适合纯架构研究场景。

**产出目录**：

```text
report/llama-70b/
├── index.html              # 总览报告（浏览器打开）
├── check-report.json       # 机器可读结果
├── analysis.json           # 参数量、层数、特殊组件
├── architecture.html       # 交互式架构图
└── weights/                # compact 策略生成的最小权重（用于 CI 验证）
```

---

## 方法二：分步对比（更细粒度控制）

### 1. 架构分析

```bash
vitriol analyze meta-llama/Llama-3.1-70B
vitriol analyze Qwen/Qwen2.5-72B
```

关注输出中的 `Total Params`、`Layers`、`Hidden Size`、`Special Features`（GQA/MLA/MoE 等）。

### 2. 交互式可视化

```bash
vitriol arch-viz meta-llama/Llama-3.1-70B --html -o report/llama_arch.html
vitriol arch-viz Qwen/Qwen2.5-72B --html -o report/qwen_arch.html
```

### 3. 智能对比（进化模块）

```bash
vitriol evolve compare meta-llama/Llama-3.1-70B Qwen/Qwen2.5-72B
```

### 4. 性能代理估算（无需 GPU）

```bash
vitriol evolve simulate meta-llama/Llama-3.1-70B --gpu H100
vitriol evolve simulate Qwen/Qwen2.5-72B --gpu H100
```

---

## 预期结论框架

| 维度 | 你可以回答的问题 |
|------|------------------|
| **拓扑** | 层数、hidden size、GQA/MQA 头数是否不同？ |
| **注意力** | 是否使用 MLA、滑动窗口、Linear-Full 混合？ |
| **FFN / MoE** | SwiGLU vs 标准 FFN？是否 MoE？专家数？ |
| **资源** | 相同 GPU 上的 FLOPs/VRAM 代理估算 |
| **CI** | `weights/` 目录能否在 CPU 环境通过 `vitriol validate`？ |

---

## 离线 / CI 场景

若模型 config 已缓存或放在本地目录：

```bash
vitriol --offline check ./models/local-qwen -o report/local --fast
```

---

## 下一步

- [Case Study 02: CI 模型 config 验证](./02-ci-model-config-validation.md)
- [Vitriol Check 设计说明](../release-validation.md)
- [在线 Demo](https://islinxu.github.io/Vitriol/viewer.html)
