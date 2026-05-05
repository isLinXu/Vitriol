# Vitriol 项目深度分析（功能完备性 + 功能说明｜详细版）
> Generated: 2026-04-15  
> Scope: 静态代码审阅（仓库内代码与文档），未运行完整端到端任务  
> Repo: `vitriol`（Python 包，源码位于 `src/vitriol/`）

---

## 目录
- [1. 项目定位与结论概览](#1-项目定位与结论概览)
- [2. 技术栈与交付形态](#2-技术栈与交付形态)
- [3. 目录结构与模块地图（代码事实）](#3-目录结构与模块地图代码事实)
- [4. 主要功能是否完备：结论与证据](#4-主要功能是否完备结论与证据)
- [5. 功能详细说明（按用户视角）](#5-功能详细说明按用户视角)
  - [5.1 CLI 总入口与全局安全开关](#51-cli-总入口与全局安全开关)
  - [5.2 最小权重生成（generate）](#52-最小权重生成generate)
  - [5.3 生成结果验证（validate）](#53-生成结果验证validate)
  - [5.4 架构分析（analyze）](#54-架构分析analyze)
  - [5.5 架构可视化（arch-viz）](#55-架构可视化arch-viz)
  - [5.6 交互式可视化器（viz）](#56-交互式可视化器viz)
  - [5.7 权重可视化报告（visualize）](#57-权重可视化报告visualize)
  - [5.8 词表可视化（vocab-viz）](#58-词表可视化vocab-viz)
  - [5.9 KV Cache / TurboQuant 基准测试（bench ...）](#59-kv-cache--turboquant-基准测试bench-)
  - [5.10 单条推理（infer）](#510-单条推理infer)
  - [5.11 神经架构搜索 NAS（nas）](#511-神经架构搜索-nasnas)
  - [5.12 架构进化工具（evolve ...）](#512-架构进化工具evolve-)
  - [5.13 模型指纹（hash）](#513-模型指纹hash)
  - [5.14 批量生成（batch）](#514-批量生成batch)
  - [5.15 导出（export）](#515-导出export)
  - [5.16 Web UI（webui）](#516-web-uiwebui)
  - [5.17 REST API（实验性）（vitriol.api.server）](#517-rest-api实验性vitriolapiserver)
- [6. 核心内部模块说明（按工程视角）](#6-核心内部模块说明按工程视角)
- [7. 工程完备性评估：测试、CI/CD、文档、发布](#7-工程完备性评估测试cicd文档发布)
- [8. 关键缺口/风险点与改进建议（按优先级）](#8-关键缺口风险点与改进建议按优先级)

---

## 1. 项目定位与结论概览

### 1.1 项目定位（从代码与 README 抽象）
Vitriol 是一个围绕 **“结构—权重解耦”** 思想构建的 LLM 工程/研究工具箱，核心目标是：
1. **只依赖模型配置（KB 级）** 就能实例化模型结构骨架（Meta device），再用多种算法生成“最小/替代权重”，产出结构兼容的权重文件；
2. 以此为基础提供：架构分析、架构可视化、KV Cache 压缩（TurboQuant / TurboQuantum）实验与基准测试、架构进化与推荐、NAS 搜索等能力；
3. 提供 CLI 作为主要交付形态，并可选启动 Web UI / REST API。

### 1.2 主要功能是否完备（结论）
从“项目自身声明的核心能力”角度（generate / validate / analyze / viz / bench / infer / evolve / nas / webui / api），**主体功能链条是闭环的**，且具备：
- 清晰的 CLI 命令组织（惰性加载命令，16 个主命令）
- 主要模块均有实现与测试用例（`tests/`）
- CI 工作流覆盖基础测试 + WebUI/API smoke

但从“产品化/工程化完备”角度仍存在 **明确缺口**（详见[第 8 节](#8-关键缺口风险点与改进建议按优先级)），最关键的包括：
- REST API 的 NAS 搜索实现目前为**模拟/占位**逻辑（非真实 NAS），`/models` 也是硬编码样例；
- 多处子命令/模块对 `trust_remote_code` 的处理不完全一致：顶层支持 `--no-trust-remote-code`，但部分路径仍在内部硬编码 `trust_remote_code=True`；
- 仓库内存在大量 `output/`、`__pycache__/` 等大体量/缓存类目录（如果这是“发布仓库”，会显著影响可维护性与分发；若仅为本地快照则另当别论）。

---

## 2. 技术栈与交付形态

### 2.1 语言与依赖（核心）
- Python：`>=3.8`
- 核心依赖：`transformers`, `torch`, `accelerate`, `safetensors`, `huggingface_hub`, `click`, `numpy`

### 2.2 可选能力（extras）
- `vitriol[webui]`：Gradio Web UI
- `vitriol[api]`：FastAPI + Uvicorn API Server
- `vitriol[viz]`：rich/matplotlib/plotly 等可视化增强（用于 hash/可视化报告等）

### 2.3 交付入口
- CLI：`vitriol = vitriol.cli.main:main`（见 `pyproject.toml`）
- WebUI：`vitriol webui ...`（内部调用 `vitriol.webui.launch()`）
- API Server：`python -m vitriol.api.server`（或调用其 `main()`）

---

## 3. 目录结构与模块地图（代码事实）

### 3.1 关键目录（相对路径）
- `src/vitriol/cli/`：CLI 命令入口与子命令（16 个主命令，且使用 LazyGroup 惰性加载）
- `src/vitriol/core/`：核心生成/验证/分析/导出/分片/流水线等
- `src/vitriol/strategies/`：权重生成策略（random/compact/ultra/...）
- `src/vitriol/adapters/`：模型家族适配器（自动发现 + 注册类）
- `src/vitriol/arch_viz/`：架构拓扑可视化（block/detail/html 渲染）
- `src/vitriol/kv/`：KV Cache 压缩与策略（含 TurboQuantum）
- `src/vitriol/bench/`：基准测试与评估（smoke/long/suite/report + PPL evaluator）
- `src/vitriol/nas/`：NAS 搜索空间/搜索器/控制器/定向优化/（含 RL 搜索器）
- `src/vitriol/evolution/`：架构进化树、对比、模拟、时间线、推荐
- `src/vitriol/webui/`：Gradio 应用
- `src/vitriol/api/`：FastAPI（实验性）
- `docs/`：GitHub Pages 静态站点 + demo data
- `tests/`：覆盖 CLI、API、WebUI、KV、Hub smoke 等测试
- `scripts/`：演示/资产同步/批量校验脚本

### 3.2 “大型目录/缓存目录”提醒（工程风险）
当前仓库快照中可见：
- `output/`：包含大量预生成模型权重/资源（体量巨大）
- `src/vitriol/**/__pycache__/`：包含 `.pyc`
如果这是准备发布/共享的主仓库，建议严格通过 `.gitignore` 排除并清理历史追踪；否则会造成：
1) clone/CI 时间暴涨；2) 安全审计成本上升；3) PyPI sdist/wheel 体积异常；4) PR diff 噪音。

---

## 4. 主要功能是否完备：结论与证据

下表按“项目对外承诺的功能”给出实现状态（✅完备 / ⚠️部分完备 / 🧪实验性 / ❌缺失）。

| 功能域 | 对外入口 | 状态 | 证据（关键代码路径） | 备注 |
|---|---|---:|---|---|
| 最小权重生成 | `vitriol generate` | ✅ | `src/vitriol/core/generator.py` + `src/vitriol/cli/commands/generate.py` | 支持策略参数、分片、shrink_config 等 |
| 生成后验证 | `vitriol validate` | ✅ | `src/vitriol/core/validator.py` | 支持模型/分词器加载与简易推理 smoke |
| 架构分析 | `vitriol analyze` | ⚠️ | `src/vitriol/core/analyzer.py` | 部分路径硬编码 `trust_remote_code=True`，估算逻辑为启发式 |
| 架构拓扑可视化（静态） | `vitriol arch-viz` | ✅ | `src/vitriol/arch_viz/*` | block/detail/html 三类输出 |
| 交互式架构查看器 | `vitriol viz` | ✅ | `src/vitriol/cli/commands/viz.py` + `src/vitriol/viz/*.html` | 本地 HTTP server + 内联 config 注入 |
| 权重可视化报告 | `vitriol visualize` | 🧪 | `src/vitriol/visualization/*` + `cli/commands/visualize.py` | 依赖 `vitriol[viz]`；更多偏实验/研究 |
| 词表可视化 | `vitriol vocab-viz` | 🧪 | `src/vitriol/vocab_viz/*` + `cli/commands/vocab_viz.py` | 支持 3D vocab viewer，本地服务 |
| KV/TurboQuant 基准 | `vitriol bench ...` | ✅ | `src/vitriol/bench/*` + `cli/commands/bench.py` | 输出 json/summary/markdown，支持对比 |
| 单条推理 | `vitriol infer` | ✅ | `src/vitriol/cli/commands/infer.py` + `bench/runner.py` | preset + chat template + stats |
| NAS（CLI） | `vitriol nas` | ✅ | `src/vitriol/nas/*` + `cli/commands/nas.py` | random/evolutionary/targeted + artifact 输出 |
| 架构进化/对比/模拟 | `vitriol evolve ...` | ✅ | `src/vitriol/evolution/*` + `cli/commands/evolve.py` | 但部分 `trust_remote_code` 处理不统一 |
| 模型指纹 | `vitriol hash` | ✅ | `src/vitriol/core/hasher.py` + `cli/commands/hash.py` | 依赖 `vitriol[viz]` 的 rich 输出 |
| Batch 生成 | `vitriol batch` | ✅ | `src/vitriol/core/batch.py` + `cli/commands/batch.py` | 从 YAML/配置批量生成 |
| 导出 | `vitriol export` | ⚠️ | `src/vitriol/core/exporter.py` + `cli/commands/export.py` | gguf 为 “prep” 级别，非完整转换 |
| WebUI | `vitriol webui` | ✅ | `src/vitriol/webui/app.py` | Gradio 多 tab 功能齐全 |
| REST API | `python -m vitriol.api.server` | ⚠️🧪 | `src/vitriol/api/server.py` | `/generate` 可用；`/nas/search` 为模拟；`/models` 为硬编码 |

---

## 5. 功能详细说明（按用户视角）

### 5.1 CLI 总入口与全局安全开关
**入口：** `vitriol`  
**实现：** `src/vitriol/cli/main.py`

关键点：
- 使用 `LazyGroup`：只有在用户调用某个命令时才 import 对应模块，减少启动成本。
- 全局开关：`--trust-remote-code/--no-trust-remote-code`  
  - 设计目标：当 compatibility 允许时，在 CI/共享环境更安全地运行（详见 `SECURITY.md`）
  - 现状问题：部分子模块仍内部写死 `trust_remote_code=True`（详见第 8 节建议）。

---

### 5.2 最小权重生成（generate）
**命令：**
```bash
vitriol generate <model_id> -o <output_dir> [options]
```
**实现：**
- CLI：`src/vitriol/cli/commands/generate.py`
- 核心：`src/vitriol/core/generator.py`
- 策略：`src/vitriol/strategies/*`

**用途：**
将 HuggingFace 模型的权重生成从 “下载真实 weights（GB/TB）” 变为 “算法生成替代 weights（MB/KB）”，仍保持：
- 参数名、shape、dtype 结构兼容；
- sharding/index 文件结构兼容（便于加载与验证）。

**关键参数：**
- `--strategy`：random / compact / ultra / sparse / ternary / binary / quantized / lowrank / structured_sparse / learned / hybrid_learned / quantum
- `--max-shard-size`：分片上限
- `--shrink/--no-shrink`：是否“缩小 config”（默认 ultra 可能启用）
- `--n-bits` / `--rank` / `--sparsity`：策略参数

**输出（典型）：**
- `<output_dir>/config.json`（或保留/生成 dummy config）
- 权重 shards（`.safetensors` / `.bin` 等，视策略与实现而定）
- index/manifest 等元信息文件（用于追踪分片、生成参数、reconcile 信息等）

---

### 5.3 生成结果验证（validate）
**命令：**
```bash
vitriol validate <output_dir> [--no-inference]
```
**实现：** `src/vitriol/core/validator.py`

**验证维度：**
1) 模型能否从生成目录加载（优先 CausalLM，失败则尝试 AutoModel）  
2) tokenizer 能否加载  
3)（可选）是否能跑一次极小推理/forward（不追求语义正确，只验证管线可用）  
4) 估算内存占用（参数 + buffer 的字节数）

**工程特性：**
- 对低内存场景有 `max_memory` 与 `offload_folder` 的降级路径。

---

### 5.4 架构分析（analyze）
**命令：**
```bash
vitriol analyze <model_id>
```
**实现：** `src/vitriol/core/analyzer.py`

**输出：**
打印模型架构摘要（architecture/type、参数量估计、层数、hidden size、vocab size、特征识别、策略文件大小估算）。

**注意：**
- 当前分析器对 `trust_remote_code` 处理为硬编码 True（存在安全/一致性问题）。
- 参数量计算优先尝试 meta 初始化拿到 `model.num_parameters()`，失败则回退启发式估算。

---

### 5.5 架构可视化（arch-viz）
**命令：**
```bash
vitriol arch-viz <model_id> [--block] [--detail] [--html] [-o <path_or_dir>] [--all]
```
**实现：**
- CLI：`src/vitriol/cli/commands/arch_viz.py`
- 引擎：`src/vitriol/arch_viz/*`

**输出类型：**
- Block diagram：粗粒度模块/层级块图
- Detail diagram：更细节的层/参数标注
- HTML：交互式页面（适合在浏览器查看）

---

### 5.6 交互式可视化器（viz）
**命令：**
```bash
vitriol viz [<model_path_or_hf_id>] [--2d|--3d] [--port 8765] [--no-open]
```
**实现：** `src/vitriol/cli/commands/viz.py` + `src/vitriol/viz/*.html`

**能力：**
- 本地起一个静态 HTTP 服务
- 将模型 config（可选读取 `meta-config.json` / `config_meta.json`）注入到 HTML 中
- 支持 2D/3D 两种可视化页面

**边界：**
- `Diffusers` 路径有 “解析 model_index.json” 的特殊处理，但部分字段仍是 placeholder（如 total_params 粗估/占位）。

---

### 5.7 权重可视化报告（visualize）
**命令：**
```bash
vitriol visualize <model_dir> [-o <out_dir>] [--layer-pattern <regex>] [--limit N]
```
**实现：** `src/vitriol/visualization/*` + `cli/commands/visualize.py`

**说明：**
读取模型目录中部分权重张量（可按 layer 正则过滤、按数量 limit），生成综合可视化报告。

**依赖：** 需要安装 `vitriol[viz]`（缺失会提示安装命令）。

---

### 5.8 词表可视化（vocab-viz）
**命令（节选）：**
```bash
vitriol vocab-viz --3d --model-id <hf_model_id_or_local_dir>
```
**实现：** `src/vitriol/vocab_viz/*` + `cli/commands/vocab_viz.py`

**能力：**
- 读取 tokenizer vocab（优先 HF tokenizer；失败可读本地 `tokenizer.json`）
- 为 token 归类（Special / Latin / Chinese / Digits / Cyrillic / Other）
- 启动本地 3D viewer（HTTP server）

---

### 5.9 KV Cache / TurboQuant 基准测试（bench ...）
**入口：** `vitriol bench <subcommand> ...`  
**实现：** `src/vitriol/cli/commands/bench.py` + `src/vitriol/bench/*` + `src/vitriol/kv/*`

**关键子命令：**
- `kv-plan`：输出某个 preset 的逐层策略决策（也可对比两个 preset）
- `kv-analyze`：离线量化误差分析（可输出 per-layer rows）
- `kv-smoke`：短 prompt 的快速 sanity benchmark（支持 compare）
- `kv-long`：长上下文 benchmark（支持 compare）
- `kv-suite`：一组 prompt_tokens 的短套件 benchmark（支持 compare）
- `kv-report`：组合 smoke/long/suite 的一键报告（支持 `--output-dir` 同时产出 json+md）
- `turboquantum`：合成 KV 张量的 TurboQuantum 测试（可 compare-modes）
- `turboquantum-model`：对真实模型 KV cache 运行 TurboQuantum

**输出格式：**
支持 `json / summary / markdown`，其中 markdown 还能包含实验元信息（利于沉淀实验记录）。

---

### 5.10 单条推理（infer）
**命令：**
```bash
vitriol infer <model_id> --prompt "..." --preset balanced --show-stats --format summary
```
**实现：** `src/vitriol/cli/commands/infer.py`

**能力：**
- 使用 `bench.runner` 的 preset 路径跑一次单 prompt 推理
- 支持 chat 模式：使用 tokenizer chat template 渲染（`--chat` / `--system-prompt` / `--assistant-prefix`）
- 支持 Qwen chat shortcut：`--preset qwen-chat` 会映射到 aggressive preset，并默认去掉 `<think>...</think>` 等

---

### 5.11 神经架构搜索 NAS（nas）
**命令：**
```bash
vitriol nas --algorithm random|evolutionary|targeted [options]
```
**实现：** `src/vitriol/nas/*` + `src/vitriol/cli/commands/nas.py`

**支持模式：**
- Random：按 iteration 采样搜索空间
- Evolutionary：按 generation + population 运行进化搜索
- Targeted：约束优化（max_vram / max_params）+ objective（min params / min vram / max efficiency）

**输出：**
搜索产物写入 `--output-dir`（含 checkpoint/结果 json 等，视实现而定）。

---

### 5.12 架构进化工具（evolve ...）
**入口：** `vitriol evolve <subcommand>`  
**实现：** `src/vitriol/evolution/*` + `src/vitriol/cli/commands/evolve.py`

**子命令：**
- `tree`：构建架构进化树并输出 HTML
- `compare`：对比两个模型架构，输出 markdown/json/html
- `simulate`：估算 VRAM/FLOPs/吞吐等
- `families`：列出已知模型家族
- `timeline`：创新时间线 HTML
- `recommend`：按约束推荐架构

**注意：**
部分实现对 `trust_remote_code` 仍写死 True（建议统一到全局开关）。

---

### 5.13 模型指纹（hash）
**命令：**
```bash
vitriol hash <model_path> [--fast]
```
**实现：** `src/vitriol/core/hasher.py` + `cli/commands/hash.py`

**指纹层：**
1) Architecture hash  
2) Behavioral DNA hash  
3) Weight distribution hash（`--fast` 可跳过）  
并可组合生成 `arx_` 前缀签名（部分条件下）。

---

### 5.14 批量生成（batch）
**命令：**
```bash
vitriol batch <config_file>
```
**实现：** `src/vitriol/core/batch.py` + `cli/commands/batch.py`

**说明：**
从 YAML/配置文件批量生成多个模型的最小权重，适合做一键 demo 或回归。

---

### 5.15 导出（export）
**命令：**
```bash
vitriol export <input_dir> -o <output_path> --format json|gguf
```
**实现：** `src/vitriol/core/exporter.py` + `cli/commands/export.py`

**说明：**
- `json`：导出结构信息
- `gguf`：当前更像 “GGUF 准备/对接” 的导出路径，不等同于完整的权重格式转换（需要结合 `core/exporter.py` 的实际实现确认功能深度）。

---

### 5.16 Web UI（webui）
**命令：**
```bash
vitriol webui --port 7860 [--share] [--debug]
```
**实现：** `src/vitriol/webui/app.py`

**功能页（Tab）概览：**
- Model Comparison：对比两个模型架构（输出 markdown report）
- Evolution Tree：构建/展示进化树（HTML）
- Targeted NAS：约束优化搜索（JSON + markdown summary）
- Architecture Simulator：性能估算（JSON + markdown summary）
- Architecture Scorecard：架构打分卡（更偏 UI 演示）
- Innovation Timeline：创新时间线（HTML）
- Architecture Recommender：按约束推荐（JSON + markdown summary）

**注意：**
WebUI 内部加载 config 使用 `trust_remote_code=True`（建议与 CLI 全局开关对齐）。

---

### 5.17 REST API（实验性）（vitriol.api.server）
**启动：**
```bash
python -m vitriol.api.server
```
**实现：** `src/vitriol/api/server.py`

**现有端点（节选）：**
- `GET /`：服务信息
- `GET /health`：健康检查
- `GET /status`：系统状态（cpu/mem/disk 等，依赖 psutil）
- `POST /generate`：异步生成任务（写入 active_jobs + background task）
- `GET /jobs/{job_id}`、`GET /jobs`：任务状态查询
- `POST /batch/generate`、`GET /batch/{batch_id}`：批量生成任务
- `GET /strategies`：从 `STRATEGY_REGISTRY` 枚举策略能力
- `GET /models`：**硬编码样例**（非真实 registry）
- `POST /nas/search`：**模拟进度 + 返回占位结果**（非真实 NAS）

**鉴权：**
提供 `api_key` query 参数校验框架（是否启用由配置决定），但默认策略需结合 `config/settings.py` 的实现与部署方式确认。

---

## 6. 核心内部模块说明（按工程视角）

> 本节用于帮助读代码/做二次开发：每个模块“负责什么、与谁协作、核心对象是什么”。

### 6.1 `core/`（核心引擎）
- `core/generator.py`：MinimalWeightGenerator（骨架构建、策略生成、分片写入、shrink_config、patch 注入等）
- `core/validator.py`：ModelValidator（加载/推理/内存）
- `core/analyzer.py`：ModelAnalyzer（结构特征识别 + 参数/大小估算）
- `core/hasher.py`：ModelHasher（多层 hash）
- `core/batch.py`：BatchGenerator（配置驱动批量生成）
- `core/exporter.py`：ModelExporter（结构导出/格式 prep）
- `core/pipeline/*`：Pipeline 化的生成/步骤编排（更利于扩展与断点续作）

### 6.2 `strategies/`（权重生成策略）
- `STRATEGY_REGISTRY` 维护策略名 → 类映射（`get_strategy()` 统一构造）
- 不同策略强调不同 trade-off：体积、可训练性、结构兼容性、速度

### 6.3 `adapters/`（模型家族适配）
- `AdapterRegistry` 会自动发现并 import `adapters/*.py`，按 LIFO 注册优先级匹配
- 适配器负责：注册/patch 对应模型家族所需的类、处理特定 config 字段差异等

### 6.4 `patches/` + `kv/` + `bench/`
这是 TurboQuant/TurboQuantum 与 KV cache 策略的核心三件套：
- `patches/*`：对 transformers/模型实现进行 monkey patch 或兼容修复
- `kv/*`：KV 编解码、存储、策略与（可选）triton kernels
- `bench/*`：把上述策略组合成可重复实验（支持 markdown 实验记录输出）

### 6.5 `evolution/` 与 `nas/`
两个面向“架构设计空间”的模块：
- evolution：偏“已知模型谱系/对比/推演/推荐”
- nas：偏“搜索空间 + 搜索算法”，包含 targeted constraint optimizer 与 RL 搜索器等

---

## 7. 工程完备性评估：测试、CI/CD、文档、发布

### 7.1 测试（tests/）
仓库存在大量 pytest 用例，覆盖面包括：
- CLI 基础与可选依赖行为
- WebUI smoke
- API server smoke
- KV cache / TurboQuant / TurboQuantum 的多维回归
- Hub smoke（可选，依赖 HF_TOKEN）
- 离线安全模式（例如 `allow_network=False` 时加载本地模型目录）

### 7.2 CI/CD（.github/workflows）
- `ci.yml`：主 CI（matrix: trust_remote_code true/false），并含 API/WebUI smoke job
- `pages.yml`：部署 docs 到 GitHub Pages
- `hub-smoke.yml`：手动触发的 HuggingFace Hub smoke（带 gate summary）

### 7.3 安全与使用建议
仓库包含 `SECURITY.md`，明确提示：
- `trust_remote_code=True` 存在执行远程代码风险
- CI/共享环境建议使用 `--no-trust-remote-code`，尽量使用可信本地模型路径

---

## 8. 关键缺口/风险点与改进建议（按优先级）

### P0（高优先级：会影响“功能真实性/安全一致性”）
1. **REST API 的 NAS 端点为模拟实现**
   - 现状：`process_nas_job()` 仅 sleep + 返回占位 best_architecture
   - 建议：复用 `NASController` 或 `ConstraintOptimizer`，使 `/nas/search` 真正调用 NAS 引擎，并产出可查询的 artifacts
2. **`trust_remote_code` 全局开关未在全链路贯彻**
   - 现状：顶层 CLI 提供开关，但 `core/analyzer.py`、`cli/commands/evolve.py`、`webui/app.py` 等仍硬编码 True
   - 建议：统一使用同一来源（ctx.obj / GenerationConfig.security）下发到所有 AutoConfig/AutoTokenizer/AutoModel 调用点

### P1（中优先级：会影响“工程体验/可维护性”）
3. **仓库包含大体量 output/ 与 __pycache__（若是发布仓库）**
   - 建议：`.gitignore` 严格排除；并对历史追踪进行清理（git rm --cached）
4. **API `/models` 为硬编码样例**
   - 建议：从 `AdapterRegistry` 或已知 family matrix / evolution families 中动态生成或返回 “支持能力矩阵”

### P2（低优先级：优化项）
5. 部分模块的异常处理使用静默 except（可读性/调试成本）
6. 可视化/演示模块中存在 placeholder 估算字段（如 diffusion total_params）
7. 进一步补齐“配置文件/环境变量/命令行”三层配置的一致性文档（尤其是安全相关项）

---

## 附：建议的“功能闭环验收清单”（可用于你们自己的 QA）
1. `vitriol generate gpt2 -o output/gpt2-compact --strategy compact` 生成成功  
2. `vitriol validate output/gpt2-compact` 加载与 tokenizer 验证通过  
3. `vitriol analyze gpt2` 能输出结构摘要  
4. `vitriol arch-viz gpt2 --html -o output/gpt2-arch.html` 生成 HTML  
5. `vitriol viz output/gpt2-compact --2d` 本地打开可视化器  
6. `vitriol bench kv-smoke gpt2 --preset balanced` 产出 json/markdown 报告  
7. `vitriol infer gpt2 --prompt "hello" --preset balanced --format summary --show-stats` 输出结果与统计  
8. `vitriol webui` 打开并能成功加载/对比两个模型配置  
9. （可选）启动 `python -m vitriol.api.server`，验证 `/generate` 创建 job 并完成

