# Vitriol CLI API 稳定性契约

> **版本**: v0.3.1  
> **生效日期**: 2026-06-19  
> **状态**: 本文件自 v0.3.1 起生效，所有标记为 **Stable** 的 CLI 命令遵循 [Semantic Versioning](https://semver.org/) 兼容性承诺。

---

## 1. 稳定性等级定义

| 等级 | 标识 | CLI 承诺 | 兼容范围 |
|------|------|----------|----------|
| **Stable** | 🟢 | 参数名、顺序、默认值在 **major 版本** 内保持不变 | v0.x → v0.(x+1) 向后兼容 |
| **Beta** | 🟡 | 参数名可能微调，提供 **1 个 minor 版本** 的弃用期 | 弃用参数保留并警告 |
| **Experimental** | 🔴 | 无稳定性保证，随时重构或删除 | 不建议脚本依赖 |

---

## 2. 命令稳定性矩阵

### 🟢 Stable 命令（v0.3.1+ 锁定）

以下命令的 CLI 接口在 **v0.x 版本** 内保持向后兼容。新增参数为可选；必填参数和默认值不删除/不修改。

| 命令 | 子命令 | 参数承诺 | 说明 |
|------|--------|----------|------|
| `check` | — | `MODEL_ID`, `--output`, `--strategy`, `--format` | Golden path 入口，项目核心工作流 |
| `generate` | — | `MODEL_ID`, `--output`, `--strategy`, `--batch` | 权重生成核心 |
| `validate` | — | `MODEL_PATH`, `--trust-remote-code` | 模型验证 |
| `analyze` | — | `MODEL_ID`, `--format`, `--output` | 架构分析 |
| `batch` | — | `CONFIG_PATH` | 批量生成 |
| `cis` | `rank`, `table`, `score`, `report`, `compare` | 所有子命令 | CIS 压缩智能评分 |
| `export` | — | `MODEL_PATH`, `--format` | 导出格式 |
| `visualize` | — | `MODEL_PATH`, `--output`, `--format` | 可视化报告 |
| `viz` | — | `MODEL_PATH` | 交互式浏览器可视化 |
| `arch-viz` | — | `MODEL_ID`, `--output` | 架构可视化 |
| `hash` | — | `MODEL_PATH` | 模型指纹 |
| `evolve` | `tree`, `compare`, `simulate`, `timeline` | 所有子命令 | 架构演化工具 |
| `vocab-viz` | — | `MODEL_ID` | 词表可视化 |
| `weight-viz` | — | `MODEL_PATH` | 权重 3D 可视化 |

### 🟡 Beta 命令

| 命令 | 子命令 | 说明 |
|------|--------|------|
| `nas` | `targeted`, `evolutionary` | 可用，但搜索参数可能微调 |
| `bench` | `kv-smoke`, `kv-long`, `kv-suite`, `kv-plan`, `kv-analyze`, `kv-report` | 基准测试套件 |

### 🔴 Experimental 命令

| 命令 | 子命令 | 说明 |
|------|--------|------|
| `infer` | — | 单提示推理，需与 vLLM/TensorRT-LLM 集成 |
| `exobrain` | `infer`, `distill` | 外部大脑推理与知识蒸馏 |
| `trace` | — | 离线 trace 生成 |
| `webui` | — | Gradio Web UI |
| `nas` | `rl` | RL-based 搜索 |
| `bench` | `turboquantum` | 量子启发 KV 基准 |

---

## 3. 兼容性规则

### 3.1 Stable 命令的 semver 承诺

在 `v0.x` 版本内（即 v0.3.1 → v0.4.0 → v0.5.0...）：

- **✅ 允许**：添加新的可选参数（`--new-flag`）
- **✅ 允许**：添加新的输出格式（如 `--format json` 基础上增加 `yaml`）
- **✅ 允许**：修复 bug，不改变行为契约
- **❌ 禁止**：删除或重命名必填参数
- **❌ 禁止**：修改参数的默认值（如 `--strategy` 默认从 `random` 改为 `compact`）
- **❌ 禁止**：删除子命令

### 3.2 Beta 命令的演进规则

- 参数变更需保留旧参数名 **1 个 minor 版本**，并输出 `DeprecationWarning`
- 子命令可能合并或拆分，提前在 CHANGELOG 中公告

### 3.3 Experimental 命令的免责声明

- 不遵循 semver 兼容性
- 命令名、参数、行为随时可能重构
- 不建议在 CI/CD 脚本或生产工具链中依赖

---

## 4. 版本升级指南

### 从 v0.3.x 到 v0.4.0

所有 🟢 Stable 命令的 CLI 接口 **100% 向后兼容**。可直接升级。

### 从 v0.3.x 到 v1.0.0

v1.0.0 将锁定当前所有 🟢 Stable 命令的接口。v1.0.0 之前任何破坏性变更都会提前 2 个 minor 版本公告。

---

## 5. 贡献者指南

向 Stable CLI 命令添加新参数时：

1. 新参数必须是 **可选**（有默认值）
2. 不修改现有参数的默认值
3. 不删除现有参数或子命令
4. 在 CHANGELOG 中记录新增参数

---

> **注意**: 本契约仅覆盖 CLI 命令行接口（参数、子命令、输出格式）。Python 程序化 API（`vitriol.core.generator.MinimalWeightGenerator` 等）的稳定性遵循 `STABILITY.md` 中的模块级别声明。
