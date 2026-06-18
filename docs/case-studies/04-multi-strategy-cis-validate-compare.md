# Case Study 04: 多策略实证 CIS + Validate 对比

> **目标**：对同一模型用多种权重策略生成、验证加载，并对比实证 CIS 与体积。  
> **适用读者**：压缩策略研究者、需要选型 `generate --strategy` 的工程师  
> **一条命令**：`vitriol cis compare`

---

## 背景

Case Study 03 介绍了 **理论 CIS 排行**。真实场景还需要回答：

- 某模型上 `compact` 与 `random` 的 **实证 PSI** 差多少？
- 哪种策略 **validate 能通过** 且 **体积最小**？
- 理论排序与实测是否一致？

Vitriol v0.3.1 提供 **`vitriol cis compare`**，自动完成：

```
对每个 strategy:
  generate → validate → empirical CIS → 记录体积/耗时
→ compare-report.json + compare-report.md
```

---

## 快速实验

### 本地 / 离线 fixture

```bash
vitriol --offline cis compare ./models/tiny-llama \
  -o report/strategy-compare \
  --strategies compact,random
```

### HuggingFace 小模型（需网络）

```bash
vitriol cis compare Qwen/Qwen2.5-0.5B \
  -o report/qwen-compare \
  --strategies random,compact,ultra
```

默认策略 trio：`random,compact,ultra`。可通过 `--strategies` 自定义。

---

## 产出解读

```text
report/strategy-compare/
├── compare-report.json    # 机器可读（CI 门禁）
├── compare-report.md      # 人类可读表格
├── compact/               # compact 策略生成物
├── random/
└── ultra/                 # 若包含在 --strategies 中
```

**compare-report.md 列说明**：

| 列 | 含义 |
|----|------|
| Empirical PSI | 基于生成权重的 CIS 实测分 |
| Theoretical PSI | 理论矩阵预估分 |
| Validate | `vitriol validate` 是否通过 |
| Loadable | 模型是否可加载 |
| Size (MB) | 输出目录总体积 |
| Time (s) | 单策略 generate+validate+score 耗时 |

---

## 典型结论模式

| 策略 | 体积 | 实证 PSI | Validate | 适用场景 |
|------|------|----------|----------|----------|
| `ultra` | 最小 | 较低 | 通常通过* | 体积极限 demo |
| `compact` | 小 | 中等 | 通过 | **CI 加载验证** |
| `random` | 大 | 较高 | 通过 | 训练管线探针 |

\* Ultra 使用 stride=0，需 `.bin` 格式；部分环境需 `--trust-remote-code` 加载自定义架构。

---

## 与 `vitriol check` 的关系

| 命令 | 用途 |
|------|------|
| `vitriol check` | 单策略 Structure-First 闭环 + HTML 报告 |
| `vitriol cis compare` | **多策略**横向对比 |

推荐流程：

1. `vitriol check` — 确认模型 config 与 adapter 链路 OK  
2. `vitriol cis compare` — 在候选策略中选型  
3. `vitriol cis rank` — 对照理论排行做 sanity check  

---

## CI 门禁示例

```bash
vitriol --offline cis compare "$MODEL" -o compare --strategies compact,random
python - <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path("compare/compare-report.json").read_text())
if not payload.get("success"):
    sys.exit(payload)
PY
```

---

## 相关文档

- [Case Study 03: CIS 理论排行](./03-cis-strategy-ranking.md)
- [Case Study 02: CI config 验证](./02-ci-model-config-validation.md)
- [Composite Action README](../../.github/actions/vitriol-check/README.md)
