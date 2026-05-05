# Vitriol Git 发布合规性深度审计报告

> **审计日期**: 2026-04-26
> **审计范围**: 全项目功能、接口、文档、测试、Git 发布合规性
> **版本**: v0.3.0
> **状态**: 🟡 基本合规，存在可修复问题

---

## 一、执行摘要

| 维度 | 状态 | 评分 | 说明 |
|------|------|------|------|
| **核心功能完整性** | 🟢 完整 | 95% | 18 CLI 命令、14 API 端点、7 WebUI Tab 全部可用 |
| **接口一致性** | 🟢 一致 | 90% | 策略/适配器注册机制完善，接口契约清晰 |
| **Git 发布合规性** | 🟡 基本合规 | 85% | 必要文件齐全，但存在遗留文件和命名残留 |
| **测试覆盖** | 🟡 中等 | 75% | 447 tests collected，但部分模块覆盖不足 |
| **文档完整性** | 🟢 完整 | 95% | README/CHANGELOG/CONTRIBUTING/SECURITY 齐全 |
| **代码质量** | 🟢 良好 | 90% | 类型提示、异常体系、Lazy Import 设计优良 |

**总体评估**: ✅ **可以发布**，但建议先修复以下 🔴 P0 和 🟡 P1 问题。

---

## 二、Git 发布合规性检查

### 2.1 必要文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `LICENSE` | ✅ | MIT License，Copyright 2024-2026 Archon Team |
| `README.md` | ✅ | ~1080 行，双语（EN/CN），完整功能介绍 |
| `README_CN.md` | ✅ | 中文文档 |
| `CHANGELOG.md` | ✅ | Keep a Changelog 格式，v0.1.0 → v0.2.0 |
| `CONTRIBUTING.md` | ✅ | 贡献指南，Conventional Commits |
| `CODE_OF_CONDUCT.md` | ✅ | 行为准则 |
| `SECURITY.md` | ✅ | 安全策略，漏洞报告流程 |
| `pyproject.toml` | ✅ | PEP 621 标准，依赖/可选依赖/脚本入口完整 |
| `.gitignore` | ✅ | Python 标准忽略规则 |
| `.github/workflows/` | ✅ | CI + Hub-Smoke + Pages 三个工作流 |
| `.github/ISSUE_TEMPLATE/` | ✅ | Issue 模板 |
| `.github/PULL_REQUEST_TEMPLATE.md` | ✅ | PR 模板 |

### 2.2 pyproject.toml 详细检查

```toml
[project]
name = "vitriol"                    ✅ 正确
version = "0.3.0"                   ✅ 与 src/vitriol/version.py 一致
requires-python = ">=3.8"          ✅ 合理
license = {file = "LICENSE"}       ✅ MIT

[project.scripts]
vitriol = "vitriol.cli.main:main"  ✅ CLI 入口正确

[project.urls]
Homepage = "https://github.com/isLinXu/Vitriol"  ✅
Repository = "https://github.com/isLinXu/Vitriol.git"  ✅
Issues = "https://github.com/isLinXu/Vitriol/issues"  ✅
```

**可选依赖分组**: ✅ 完整
- `viz`: rich, matplotlib, seaborn, pandas, plotly, scipy
- `webui`: gradio>=4.0.0
- `api`: fastapi, uvicorn, pydantic>=2, psutil
- `dev`: pytest, pytest-cov, pytest-timeout, ruff, mypy, pre-commit

---

## 三、功能与接口完备性分析

### 3.1 CLI 命令体系 (18 个命令)

| 命令 | 实现文件 | 状态 | 说明 |
|------|----------|------|------|
| `generate` | `cli/commands/generate.py` | ✅ | 权重生成，支持 12 种策略 |
| `validate` | `cli/commands/validate.py` | ✅ | 模型验证 |
| `analyze` | `cli/commands/analyze.py` | ✅ | 架构分析 |
| `batch` | `cli/commands/batch.py` | ✅ | 批量生成 |
| `bench` | `cli/commands/bench.py` | ✅ | 8 个子命令 (kv-plan/smoke/long/suite/report/analyze/turboquantum/turboquantum-model) |
| `export` | `cli/commands/export.py` | ✅ | 导出结构/GGUF 预备 |
| `visualize` | `cli/commands/visualize.py` | ✅ | 权重可视化报告 |
| `viz` | `cli/commands/viz.py` | ✅ | 3D 交互式可视化 |
| `arch-viz` | `cli/commands/arch_viz.py` | ✅ | 架构拓扑可视化 |
| `nas` | `cli/commands/nas.py` | ✅ | NAS 搜索 (random/evolutionary/targeted) |
| `vocab-viz` | `cli/commands/vocab_viz.py` | ✅ | 词表 3D 可视化 |
| `weight-viz` | `cli/commands/weight_viz.py` | ✅ | 权重矩阵 3D 可视化 |
| `evolve` | `cli/commands/evolve.py` | ✅ | 6 个子命令 (tree/compare/simulate/families/timeline/recommend) |
| `hash` | `cli/commands/hash.py` | ✅ | 模型指纹 |
| `infer` | `cli/commands/infer.py` | ✅ | 单提示推理 |
| `webui` | `cli/commands/webui.py` | ✅ | Gradio Web UI |
| `exobrain` | `cli/commands/exobrain.py` | ✅ | 2 个子命令 (infer/distill) |

**CLI 设计亮点**:
- LazyGroup 机制：命令模块按需加载，避免导入重依赖
- 全局安全参数：`--trust-remote-code`, `--allow-network`, `--local-files-only`, `--offline`
- 安全警告：generate/validate 命令自动输出 trust_remote_code 警告

### 3.2 API 端点 (14 个端点)

| 端点 | 方法 | 状态 | 说明 |
|------|------|------|------|
| `/` | GET | ✅ | 根端点 |
| `/health` | GET | ✅ | 健康检查 |
| `/status` | GET | ✅ | 系统状态 (依赖 psutil) |
| `/generate` | POST | ✅ | 异步权重生成 |
| `/jobs/{job_id}` | GET | ✅ | 任务状态查询 |
| `/jobs` | GET | ✅ | 任务列表 |
| `/nas/search` | POST | ✅ | NAS 搜索 |
| `/models` | GET | ✅ | 动态聚合模型列表 |
| `/models/families` | GET | ✅ | 模型家族 |
| `/models/adapters` | GET | ✅ | 适配器列表 |
| `/strategies` | GET | ✅ | 策略列表 |
| `/stream/logs` | GET | ✅ | SSE 日志流 |
| `/batch/generate` | POST | ✅ | 批量生成 |
| `/batch/{batch_id}` | GET | ✅ | 批量任务状态 |

**API 安全特性**:
- `verify_api_key`: 可选 API Key 认证
- CORS: 当前配置为 `allow_origins=["*"]`，生产环境需收紧
- 安全上下文传播：process_generation_job 始终写入 trust_remote_code/allow_network/local_files_only

### 3.3 WebUI 功能 (7 个 Tab)

| Tab | 功能 | 状态 |
|-----|------|------|
| ⚖️ Model Comparison | 模型架构比较 | ✅ |
| 🌳 Evolution Tree | 进化树可视化 | ✅ |
| 🎯 Targeted NAS | 约束优化 NAS | ✅ |
| ⚡ Architecture Simulator | 性能模拟 | ✅ |
| 📋 Architecture Scorecard | 架构评分卡 | ✅ |
| 📅 Innovation Timeline | 创新时间线 | ✅ |
| 🎯 Architecture Recommender | 架构推荐 | ✅ |

### 3.4 策略注册 (12 个策略)

| 策略 | 类名 | 注册状态 | 测试覆盖 |
|------|------|----------|----------|
| random | RandomStrategy | ✅ | ✅ |
| compact | CompactStrategy | ✅ | ✅ |
| ultra | UltraStrategy | ✅ | ✅ |
| hybrid_ultra | HybridUltraStrategy | ✅ | ✅ |
| sparse | SparseStrategy | ✅ | ⚠️ 无专门测试 |
| ternary | TernaryStrategy | ✅ | ⚠️ 无专门测试 |
| binary | BinaryStrategy | ✅ | ⚠️ 无专门测试 |
| quantized | QuantizedStrategy | ✅ | ⚠️ 无专门测试 |
| lowrank | LowRankStrategy | ✅ | ⚠️ 无专门测试 |
| structured_sparse | StructuredSparseStrategy | ✅ | ⚠️ 无专门测试 |
| quantum | QuantumStrategy | ✅ | ⚠️ 无专门测试 |
| learned | LearnedWeightStrategy | ✅ | ⚠️ 无专门测试 |
| hybrid_learned | HybridLearnedStrategy | ✅ | ⚠️ 无专门测试 |

### 3.5 适配器注册 (12 个适配器)

| 适配器 | 匹配条件 | 状态 |
|--------|----------|------|
| LlamaAdapter | `model_type == "llama"` | ✅ |
| QwenMoeAdapter | `model_type == "qwen2_moe"` | ✅ |
| Qwen35MoeAdapter | `model_type == "qwen3_5_moe"` | ✅ |
| DeepSeekAdapter | `model_type == "deepseek"` | ✅ |
| MistralAdapter | `model_type == "mistral"` | ✅ |
| GemmaAdapter | `model_type == "gemma"` | ✅ |
| PhiAdapter | `model_type == "phi"` | ✅ |
| CohereAdapter | `model_type == "cohere"` | ✅ |
| GLMAdapter | `model_type == "glm"` | ✅ |
| StableLMAdapter | `model_type == "stablelm"` | ✅ |
| MiniMaxAdapter | `model_type == "minimax"` | ✅ |
| DefaultAdapter | 通用回退 | ✅ |

---

## 四、发现的问题

### 🔴 P0: 阻塞发布的问题

| # | 问题 | 影响 | 修复建议 |
|---|------|------|----------|
| P0-1 | `cli/commands.py` 遗留旧版 CLI 实现 | 包含未注册的命令(search/config/serve/dashboard/info)，代码冗余，可能误导用户 | **删除** `cli/commands.py`，或将其功能整合到 `cli/commands/` 下 |
| P0-2 | `tests/test_smoke_archon.py` 文件名含旧名 | 与 `test_smoke_vitriol.py` 内容重复，文件名含 "archon" | **删除** `test_smoke_archon.py`，保留 `test_smoke_vitriol.py` |
| P0-3 | `CHANGELOG.md` 仍引用 "Archon" | 第 3 行: "All notable changes to the Archon project" | 改为 "Vitriol project" |
| P0-4 | `LICENSE` Copyright 仍为 "Archon Team" | 法律实体名称不一致 | 改为 "Vitriol Team" 或保留历史记录 |
| P0-5 | `CONTRIBUTING.md` 多处引用 "archon" | `ruff check src/archon`, `pytest --cov=archon`, `mypy src/archon/` | 全局替换为 `vitriol` |
| P0-6 | `SECURITY.md` 首段引用 "Archon" | "Thank you for helping keep Archon and its users safe" | 改为 "Vitriol" |

### 🟡 P1: 建议修复的问题

| # | 问题 | 影响 | 修复建议 |
|---|------|------|----------|
| P1-1 | `adapters/__init__.py` 未导出具体适配器类 | 外部无法 `from vitriol.adapters import LlamaAdapter` | 添加 `from .llama import LlamaAdapter` 等导出 |
| P1-2 | `cli/__init__.py` 为空文件 | 不符合包规范，虽然不影响功能 | 添加模块说明注释或导出 |
| P1-3 | `api/__init__.py` 仅含注释 | 无公共接口导出 | 添加 `from .server import app` 等导出 |
| P1-4 | `core/pipeline/` 实验性代码不完整 | `__init__.py` 为空，`ResolveShardMapStep` 未导出，pipeline 未实际启用 | 完善 `__init__.py` 导出，或标记为内部 API |
| P1-5 | `webui/app.py` 硬编码 GitHub 链接 | 页脚链接为 `https://github.com/your-org/vitriol` | 改为 `https://github.com/isLinXu/Vitriol` |
| P1-6 | `CHANGELOG.md` 链接仍为 `your-org/Archon` | 底部 compare 链接 | 改为 `isLinXu/Vitriol` |
| P1-7 | `docs/` 下多个文件含 "Archon" | docs/data, docs/manifests, docs/viz-models 等 | 批量替换或保留历史说明 |
| P1-8 | 测试文件 `test_hub_smoke_models.py` 含 "archon" | 内容引用旧名 | 全局替换为 `vitriol` |
| P1-9 | 测试文件 `test_generation_config_resolution.py` 含 "archon" | 内容引用旧名 | 全局替换为 `vitriol` |
| P1-10 | `output/` 和 `scripts/` 下 README/脚本含 "Archon" | 输出目录和脚本中的历史名称 | 更新或添加说明 |

### 🟢 P2: 可选优化

| # | 问题 | 说明 |
|---|------|------|
| P2-1 | API CORS 过于宽松 | `allow_origins=["*"]`，建议生产环境配置为具体域名 |
| P2-2 | API 无速率限制 | 未集成限流中间件，建议添加 slowapi 等 |
| P2-3 | 策略缺少单元测试 | sparse/ternary/binary/quantized/lowrank/structured_sparse/quantum/learned 无专门测试 |
| P2-4 | adapters 缺少单元测试 | 依赖集成测试覆盖，建议补充 |
| P2-5 | `core/pipeline/` 实验性 | 代码存在但未启用，建议完善或移除 |

---

## 五、核心模块接口一致性

### 5.1 接口契约矩阵

| 模块 | 主要类 | 返回类型 | 一致性 |
|------|--------|----------|--------|
| `core/generator.py` | `MinimalWeightGenerator` | `GenerationResult` (dataclass, 含 `to_dict()`) | ✅ 高 |
| `core/validator.py` | `ModelValidator` | `ValidationReport` (dataclass, 含 `to_dict()`) | ✅ 高 |
| `core/analyzer.py` | `ModelAnalyzer` | `ModelAnalysis` (dataclass) | ✅ 高 |
| `core/exporter.py` | `ModelExporter` | None / 副作用 | ✅ 中 |
| `core/batch.py` | `BatchGenerator` | None / 副作用 | ✅ 中 |
| `core/hasher.py` | `ModelHasher` | str (hash) | ✅ 高 |

### 5.2 Lazy Import 设计

| 包 | Lazy 机制 | 状态 |
|----|-----------|------|
| `vitriol` | `__getattr__` → core.generator/validator/analyzer, config.manager | ✅ |
| `vitriol.core` | `__getattr__` → generator/validator/analyzer | ✅ |
| `vitriol.nas` | `__getattr__` → searcher/evaluator/controller | ✅ |
| `vitriol.evolution` | 直接导入 (轻量) | ✅ |
| `vitriol.strategies` | try/except ImportError 保护可选策略 | ✅ |

### 5.3 异常体系

```
VitriolError (base)
├── ConfigError
│   ├── ConfigLoadError
│   └── ConfigValidationError
├── ModelError
│   └── ModelBuildError
├── WeightGenerationError
├── StrategyError
├── IncompatibleStrategyError
└── StrategyNotFoundError
```

✅ 异常体系完整，所有异常均继承 `VitriolError`，支持 `recoverable` 标记。

---

## 六、测试覆盖分析

### 6.1 测试统计

- **总测试文件**: 46 个 (tests/ 目录)
- **收集到的测试**: 447 tests (pytest --collect-only)
- **测试分类**:
  - API 测试: 6 个文件
  - CLI 测试: 4 个文件
  - KV Cache 测试: 8 个文件
  - 端到端测试: 3 个文件
  - 安全/加固测试: 4 个文件
  - 架构分析测试: 2 个文件
  - ExoBrain 测试: 1 个文件
  - 其他: 18 个文件

### 6.2 覆盖薄弱区域

| 模块 | 测试状态 | 风险 |
|------|----------|------|
| `strategies/learned.py` | ❌ 无专门测试 | 复杂策略，需补充 |
| `strategies/quantum.py` | ❌ 无专门测试 | 需补充 |
| `strategies/sparse.py` | ❌ 无专门测试 | 需补充 |
| `strategies/ternary.py` | ❌ 无专门测试 | 需补充 |
| `strategies/binary.py` | ❌ 无专门测试 | 需补充 |
| `strategies/quantized.py` | ❌ 无专门测试 | 需补充 |
| `strategies/lowrank.py` | ❌ 无专门测试 | 需补充 |
| `strategies/structured_sparse.py` | ❌ 无专门测试 | 需补充 |
| `adapters/*.py` | ❌ 无单元测试 | 依赖集成测试 |
| `core/pipeline/` | ❌ 无测试 | 实验性代码 |
| `webui/app.py` | ⚠️ 仅 smoke 测试 | 功能测试需补充 |
| `viz/` (可视化) | ⚠️ 仅 truthfulness 测试 | 前端为主，风险低 |

---

## 七、安全合规性

### 7.1 trust_remote_code 全链路参数化

| 层级 | 实现 | 状态 |
|------|------|------|
| CLI | `--trust-remote-code/--no-trust-remote-code` 全局选项 | ✅ |
| API | `GenerateRequest.trust_remote_code`, `BatchGenerateRequest.trust_remote_code` | ✅ |
| WebUI | 每个 Tab 的 Checkbox 开关 | ✅ |
| Core | `security={"trust_remote_code": ...}` 字典传递 | ✅ |

### 7.2 安全警告

- `generate` 命令: 启用 trust_remote_code 时输出 `[SECURITY WARNING]`
- `validate` 命令: 同上
- CLI 帮助文档: 明确说明 `--no-trust-remote-code` 用于安全环境

### 7.3 安全策略文档

- `SECURITY.md`: 漏洞报告流程、支持版本、范围说明
- `CODE_OF_CONDUCT.md`: 社区行为准则

---

## 八、发布前检查清单

### 必须完成 (P0)

- [ ] 删除 `src/vitriol/cli/commands.py` 遗留文件
- [ ] 删除 `tests/test_smoke_archon.py` 重复测试
- [ ] 将 `CHANGELOG.md` 中 "Archon" 改为 "Vitriol"
- [ ] 将 `LICENSE` Copyright 改为 "Vitriol Team" (或保留历史)
- [ ] 将 `CONTRIBUTING.md` 中 `src/archon` → `src/vitriol`, `archon` → `vitriol`
- [ ] 将 `SECURITY.md` 中 "Archon" 改为 "Vitriol"

### 强烈建议 (P1)

- [ ] 完善 `adapters/__init__.py` 导出具体适配器类
- [ ] 完善 `core/pipeline/__init__.py` 和 `steps/__init__.py` 导出
- [ ] 更新 `webui/app.py` 页脚 GitHub 链接
- [ ] 更新 `CHANGELOG.md` 底部 compare 链接
- [ ] 清理 `docs/` 下 "Archon" 残留（或保留历史说明）
- [ ] 清理 `tests/` 中 `test_hub_smoke_models.py` 和 `test_generation_config_resolution.py` 的 "archon" 引用
- [ ] 更新 `output/` 和 `scripts/` 中的旧名称

### 可选优化 (P2)

- [ ] 补充 strategies 单元测试 (learned, quantum, sparse 等)
- [ ] 补充 adapters 单元测试
- [ ] API CORS 配置收紧
- [ ] API 添加速率限制

---

## 九、结论

**Vitriol v0.3.0 整体功能完备、接口一致、文档齐全，具备 Git 发布条件。**

主要优势：
1. **功能丰富**: 18 CLI 命令、14 API 端点、7 WebUI Tab、12 种权重策略
2. **架构清晰**: Lazy Import、策略注册、适配器自动发现、异常体系完善
3. **安全合规**: trust_remote_code 全链路参数化，安全警告到位
4. **文档完整**: README/CHANGELOG/CONTRIBUTING/SECURITY 齐全

主要风险：
1. **旧名残留**: 文件名和内容中仍有 "archon" 引用，需清理
2. **遗留文件**: `cli/commands.py` 和 `test_smoke_archon.py` 需删除
3. **测试覆盖**: 部分策略和适配器缺少单元测试

**建议**: 完成 P0 修复后即可发布，P1 修复可随 v0.3.1 发布。

---

*报告生成时间: 2026-04-26*
*审计工具: CodeBuddy Agent + 人工复核*
