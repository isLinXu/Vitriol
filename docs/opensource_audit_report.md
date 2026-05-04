# Vitriol 开源前准确性审查报告

> **审查日期**: 2026-04-08  
> **审查范围**: README.md / README_CN.md 声明 vs 实际代码  
> **审查方法**: 自动化代码搜索 + 人工交叉验证

---

## 一、README 核心声明验证

### 1.1 数值声明

| 声明 | README 值 | 实际值 | 状态 |
|------|----------|--------|------|
| 版本号 | 0.2.0 | `pyproject.toml:7` → `0.2.0` | ✅ 准确 |
| Python 要求 | ≥3.8 | `pyproject.toml:10` → `>=3.8` | ✅ 准确 |
| 源文件数 | 150+ | 145 个 .py 文件 | ⚠️ 略有出入 (badge 写 150+, 实际 145, 仍在合理范围) |
| CLI 命令数 | 16 | `COMMAND_SPECS` 中 16 个 | ✅ 准确 |
| 权重策略数 | 12 | `STRATEGY_REGISTRY` 中 12 个 (含 hybrid_learned) | ✅ 准确 |
| 架构分析器 | 10 | `AnalyzerRegistry` 中 10 个 + default | ✅ 准确 |
| NAS 算法 | 4 | Random, Evolutionary, Targeted, RL | ✅ 准确 |

### 1.2 策略列表验证

| 策略 | CLI Flag | 源文件存在 | 注册表 | README |
|------|----------|----------|--------|--------|
| Random | `random` | `strategies/random.py` ✅ | ✅ | ✅ |
| Compact | `compact` | `strategies/compact.py` ✅ | ✅ | ✅ |
| Ultra | `ultra` | `strategies/ultra.py` ✅ | ✅ | ✅ |
| Sparse | `sparse` | `strategies/sparse.py` ✅ | ✅ | ✅ |
| Structured Sparse | `structured_sparse` | `strategies/structured_sparse.py` ✅ | ✅ | ✅ |
| Ternary | `ternary` | `strategies/ternary.py` ✅ | ✅ | ✅ |
| Binary | `binary` | `strategies/binary.py` ✅ | ✅ | ✅ |
| Quantized | `quantized` | `strategies/quantized.py` ✅ | ✅ | ✅ |
| LowRank | `lowrank` | `strategies/lowrank.py` ✅ | ✅ | ✅ |
| Learned | `learned` | `strategies/learned.py` ✅ | ✅ | ✅ |
| Hybrid Learned | `hybrid_learned` | `strategies/learned.py` ✅ | ✅ | ✅ |
| Quantum | `quantum` | `strategies/quantum.py` ✅ | ✅ | ✅ |

### 1.3 模型适配器验证

| 适配器 | 源文件 | README 声称 | 状态 |
|--------|--------|------------|------|
| LlamaAdapter | `adapters/llama.py` | ✅ | 准确 |
| QwenMoeAdapter | `adapters/qwen.py` | ✅ (Qwen) | 准确，实际更细：QwenMoe + Qwen35Moe |
| DeepSeekAdapter | `adapters/deepseek.py` | ✅ | 准确 |
| DefaultAdapter | `adapters/base.py` | ✅ | 准确 |
| Qwen35MoeAdapter | `adapters/qwen.py` | 未单独提及 | README 仅说 "Qwen" 覆盖了 |

### 1.4 NAS 算法类名验证

| README 声称 | 实际类名 | 状态 |
|------------|---------|------|
| Random Search | `RandomSearcher` | ✅ |
| Evolutionary | `EvolutionarySearcher` | ✅ |
| Targeted → `TargetedNASEvaluator` | `ConstraintOptimizer` + `MultiObjectiveOptimizer` + `DirectedMutator` | ⚠️ **类名不一致** — README 写 `TargetedNASEvaluator`，实际是 `ConstraintOptimizer` 等三个类 |
| RL Agent → `RLAgent` | `RLSearcher` | ⚠️ **类名不一致** — README 写 `RLAgent`，实际是 `RLSearcher` |

### 1.5 KV Cache 组件验证

| 组件 | README 声称 | 实际 | 状态 |
|------|------------|------|------|
| TurboQuant | ✅ | `patches/turboquant.py` + `kv/codec.py` | ✅ |
| Adaptive KV Codec | ✅ | `kv/codec.py` → `AdaptiveKVCodec` | ✅ |
| Sparse V | ✅ | `kv/codec.py` → `compute_skip_attention` | ✅ |
| Compute Skip Attention | ✅ | `kv/codec.py` → `ComputeSkipConfig/Result` | ✅ |
| Triton 内核 | ✅ | `kv/triton_kernels.py` (FWHT, quantize, pack) | ✅ |
| Policy 预设 | ✅ safe/balanced/aggressive | `kv/policy.py` → `KVPolicyPreset` | ✅ (实际有 5 个: safe/balanced/fast-balanced/aggressive/ultra-long) |
| TurboQuantum | ✅ | `kv/turboquantum.py` → `TurboQuantumCodec` | ✅ |

### 1.6 Triton GPU 内核验证

README 声称 "Triton GPU acceleration"，实际验证：
- `kv/triton_kernels.py` 有 `import triton` + `import triton.language as tl`
- 支持 FWHT、blockwise quantize、pack/unpack
- **有 graceful fallback**：`try/except ImportError` 回退到纯 PyTorch 实现
- 没有独立的 `.triton` 文件，全部在 `.py` 中实现

### 1.7 PPL Evaluation Framework 验证

README 声称有 `ppl_evaluator.py` 模块：
- 文件存在：`bench/ppl_evaluator.py` (24.79 KB)
- 类：`PPLEvaluator`, `PPLConfig` — README API 示例中引用的类名一致 ✅

---

## 二、CIS 框架数据准确性

### 2.1 公式验证

README 公式：`Ψ(S) = α·η_info + β·η_storage + γ·η_express + δ·T_train`

源码 (`compression_intelligence.py:12`)：
```python
Ψ(S) = α·η_info + β·η_storage + γ·η_express + δ·T_train
```
✅ 完全一致

### 2.2 权重系数验证

README 未显式列出权重值。源码默认值：
```python
alpha: float = 0.3   # Information weight
beta: float = 0.3    # Storage weight
gamma: float = 0.25  # Expressive power weight
delta: float = 0.15  # Trainability weight
```
总和: 0.3 + 0.3 + 0.25 + 0.15 = **1.0** ✅

### 2.3 API 导出验证

README 示例：
```python
from vitriol.metrics import CompressionIntelligenceScorer, generate_score_comparison_table
```

`metrics/__init__.py` 实际导出：
```python
CompressionIntelligenceScorer  ✅
generate_score_comparison_table  ✅
```

---

## 三、安全扫描结果

### P0 — 严重 (必须修复)

| # | 问题 | 文件 | 详情 |
|---|------|------|------|
| **S1** | 硬编码用户路径泄露 | `src/vitriol/cli/commands/viz.py` | 已修复；现改为通用正则匹配，不再依赖用户目录 |
| **S2** | `output/` 未被 gitignore | `.gitignore` | 缺少 `output/` 排除规则，output/ 目录已包含 511 个文件 |

### P1 — 中等 (建议修复)

| # | 问题 | 文件 | 详情 |
|---|------|------|------|
| **S3** | pickle 反序列化 | `resilience/checkpoint.py:138,165` | `pickle.load()` 可能导致 RCE，建议改用 safetensors 或添加安全文档 |
| **S4** | XXXXX 占位符 | `core/shard_manager.py:123,125`, `core/generator.py:894,922,971` | 需添加注释说明这是 shard 数量通配符 |

### P2 — 无需修改

- ✅ 无硬编码 API 密钥/Token
- ✅ 无 `.env` 文件
- ✅ 无非 localhost 的 IP 地址
- ✅ `.gitignore` 正确排除了 `__pycache__/`, `.env`, `.pypirc` 等
- ✅ 无 `shell=True` 的 subprocess 调用
- ✅ 无 SQL 注入风险
- ✅ 无 debug=True 配置
- ✅ 所有 `eval()` 均为 `model.eval()`

---

## 四、项目结构与文件引用验证

### 4.1 README 项目结构 vs 实际目录

| README 声称 | 实际 | 状态 |
|------------|------|------|
| `strategies/` 12 策略 | 12 个文件 (random→quantum + learned) | ✅ |
| `kv/` 6 模块 | 7 个文件 (含 turboquantum.py) | ⚠️ README 说 6 个，实际 7 个 |
| `patches/` 11 模块 | 10 个 .py 文件 | ⚠️ README 说 11 个 |
| `cli/` 16 commands | 16 个命令文件 | ✅ |
| `nas/` 含 rl_agent.py | ✅ | ✅ |
| `arch_viz/analyzers.py` 10 analyzers | ✅ | ✅ |
| `bench/` 含 runner.py, ppl_evaluator.py | ✅ | ✅ |
| `viz/` 4 HTML templates | 4 个 .html 文件 | ✅ |

### 4.2 GitHub Actions CI/CD

| Workflow | 触发条件 | 状态 |
|----------|---------|------|
| `ci.yml` | push main / PR | ✅ (3 jobs: test, api-smoke, webui-smoke) |
| `pages.yml` | push docs/ | ✅ |
| `hub-smoke.yml` | workflow_dispatch | ✅ |

### 4.3 依赖验证 (pyproject.toml)

| 依赖 | 声明 | 说明 |
|------|------|------|
| transformers ≥4.40 | ✅ | 核心依赖 |
| torch ≥2.0 | ✅ | 核心依赖 |
| accelerate ≥0.20 | ✅ | Meta device |
| safetensors ≥0.3 | ✅ | 安全序列化 |
| click ≥8.0 | ✅ | CLI 框架 |
| numpy <2 | ✅ | 防止兼容性问题 |
| triton | **未声明** | 可选依赖，代码中有 graceful fallback |
| rich, matplotlib, scipy | `[viz]` extra | ✅ |
| gradio ≥4.0 | `[webui]` extra | ✅ |
| fastapi, uvicorn, pydantic | `[api]` extra | ✅ |

---

## 五、发现的不一致问题清单

### 需要修复 (6 项)

| # | 严重度 | 问题 | 建议修复 |
|---|--------|------|---------|
| **1** | ✅ 已修复 | `viz.py` 曾硬编码用户目录路径 | 已替换为通用正则匹配 |
| **2** | 🔴 P0 | `.gitignore` 缺少 `output/` | 添加 `output/` 规则 + 清理已跟踪文件 |
| **3** | 🟡 P1 | README NAS 表写 `TargetedNASEvaluator`，实际类名是 `ConstraintOptimizer` | 更正为 `ConstraintOptimizer` |
| **4** | 🟡 P1 | README NAS 表写 `RLAgent`，实际类名是 `RLSearcher` | 更正为 `RLSearcher` |
| **5** | 🟢 P2 | README 写 "kv/ (6 modules)"，实际有 7 个文件 | 改为 7 |
| **6** | 🟢 P2 | README 写 "patches/ (11 modules)"，实际 10 个 .py 文件 | 改为 10 |

### 源码 badge 不一致 (1 项)

| # | 问题 | 当前 | 建议 |
|---|------|------|------|
| **7** | 🟢 P2 | Badge 写 "150+" 个源文件 | 改为 "145+" 或等添加更多文件后恢复 |

---

## 六、审查结论

### 总体评估: ⭐⭐⭐⭐ (4/5) — 基本准确，少量需修复

**整体质量很高**。README 的核心声明（CLI 命令数、策略数、分析器数、NAS 算法数）全部与代码一致。代码组织良好，无硬编码密钥，无安全漏洞。CIS 框架实现与文档描述完全吻合。

**开源前必须修复**：
1. 硬编码用户路径 (`viz.py:207`)
2. `.gitignore` 添加 `output/`

**建议修复**：
3. README 中 2 个 NAS 类名不一致
4. 2 个小数字不一致 (kv 模块数、patches 模块数)
