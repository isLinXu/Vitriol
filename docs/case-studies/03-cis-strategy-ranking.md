# Case Study 03: CIS 压缩策略排行

> **目标**：用 Compression Intelligence Score (CIS) 在统一框架下对比 13 种权重生成策略。  
> **适用读者**：量化/压缩研究者、架构消融实验设计者  
> **公式**：Ψ(S) = α·η_info + β·η_storage + γ·η_express + δ·T_train

---

## 背景

Vitriol 提供 13 种 `generate --strategy` 选项，从 **random**（高表达、零压缩）到 **ultra**（极致体积、低可训练性）。  
单靠「文件体积比」无法回答：**哪种策略更适合 CI 加载测试？哪种更适合训练管线探针？**

CIS 把四个维度合成单一 PSI 分数，并支持 **理论排行** 与 **实证打分** 两种模式。

---

## 方法一：理论排行（零依赖、秒级）

```bash
# 终端表格
vitriol cis rank

# JSON（适合 CI / 仪表盘）
vitriol cis rank --json

# Markdown 完整对比表
vitriol cis table

# 合并报告（Top 5 + 全表）
vitriol cis report -o report/cis-strategies.md
```

**典型结论（理论 PSI，默认权重 α=0.3, β=0.3, γ=0.25, δ=0.15）**：

| 场景 | 推荐策略 | 原因 |
|------|----------|------|
| CI 加载验证 | `compact` | η_storage 高，体积可控 |
| 训练管线探针 | `random` | η_info / T_train 高 |
| 极限体积 demo | `ultra` / `hybrid_ultra` | η_storage ≈ 1.0 |
| 平衡研究 | `learned` / `hybrid_learned` | 四维较均衡 |

> 运行 `vitriol cis rank` 获取当前版本的精确排序。

---

## 方法二：实证 CIS（基于已生成权重）

对同一模型用不同策略生成权重，再分别打分：

```bash
MODEL=Qwen/Qwen2.5-0.5B

vitriol generate "$MODEL" -o out/random   --strategy random
vitriol generate "$MODEL" -o out/compact  --strategy compact
vitriol generate "$MODEL" -o out/ultra    --strategy ultra

vitriol cis score out/random  --strategy random  -o scores/random.json
vitriol cis score out/compact --strategy compact -o scores/compact.json
vitriol cis score out/ultra   --strategy ultra   -o scores/ultra.json
```

每个 JSON 包含：`psi`、四维分数、`radar_vector`、参与打分的 tensor 数量。

---

## 方法三：与 PPL 评估结合（Beta）

KV / 量化路径的质量验证请使用 bench 子系统的 PPL 评估（见 `vitriol bench kv-*`）。  
CIS 评估的是 **权重填充策略的结构-统计特性**，不替代语言建模困惑度。

推荐工作流：

1. `vitriol cis rank` — 初选策略  
2. `vitriol generate` — 生成候选  
3. `vitriol validate` — 确认可加载  
4. （可选）PPL bench — 验证推理质量代理

---

## Python API

```python
from vitriol.metrics import CompressionIntelligenceScorer, generate_score_comparison_table

scorer = CompressionIntelligenceScorer()
for name, psi in scorer.score_all_strategies():
    print(name, psi)

print(generate_score_comparison_table())
```

---

## 相关文档

- [CIS 框架说明](../cis_framework_explained.html)
- [Case Study 01: 零下载架构对比](./01-zero-download-architecture-compare.md)
- [Case Study 02: CI config 验证](./02-ci-model-config-validation.md)
