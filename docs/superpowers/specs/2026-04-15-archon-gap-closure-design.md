# Archon 功能缺口补全与闭环（方案 A）设计稿
> Generated: 2026-04-15  
> Status: Draft（已获用户口头确认“按默认推进”，待 spec review 后进入实现计划）  
> Scope: 本次设计聚焦“功能真实性 + 安全一致性 + API 信息可信”三条闭环，不做大规模重构与 UI 美化。

---

## 1. 背景与问题陈述

基于对当前仓库的静态审阅，项目主干功能（CLI / WebUI / API）已具备较完整的模块化实现，但存在三类影响“可用闭环”的缺口：

1) **安全一致性缺口**：仓库整体提供了 `trust_remote_code` 的安全提示与 CLI 全局开关，但部分内部加载路径仍直接写死 `trust_remote_code=True`，导致：
   - CLI 传 `--no-trust-remote-code` 时并不能全链路生效
   - WebUI/API 与 CLI 行为不一致

2) **功能真实性缺口（API NAS）**：FastAPI 的 `/nas/search` 当前仅模拟进度并返回占位结果，不满足“功能闭环可验证”的要求。

3) **信息可信缺口（API models）**：FastAPI 的 `/models` 当前返回硬编码样例，无法反映真实支持范围/家族信息。

---

## 2. 目标与非目标

### 2.1 目标（闭环定义 + 验收口径）

**G1. 安全一致性闭环：**  
同一个 `trust_remote_code` 开关/策略在 **CLI / WebUI / API** 下表现一致，且所有 `AutoConfig/AutoTokenizer/AutoModel*` 的加载路径都遵从该开关。

**G2. NAS API 真实性闭环：**  
FastAPI 的 `/nas/search` 变为“真实 NAS 计算 + 可查询结果 + 产物落盘”。

**G3. Models API 信息可信闭环：**  
FastAPI 的 `/models` 返回动态生成的“模型家族/支持矩阵”信息，且来源与项目内事实一致。

### 2.2 非目标（明确不做）
- 不在本次设计中做大范围架构重构（例如拆分 API server、重写 job system）
- 不保证 NAS 的绝对性能/最优性（只保证“真实执行、可复现、可观测”）
- 不对所有 WebUI 页面做交互/视觉优化（仅补充必要开关与信息展示）
- 不强制清理仓库中 `output/` 等历史文件（可作为后续“仓库发布闭合”单独工作流）

---

## 3. 方案概览（方案 A：优先补“真实性 + 一致性”）

实施顺序建议：
1) **统一 `trust_remote_code` 一致性（先改最小面）**
2) **API：NAS 从模拟改真实（targeted NAS 优先）**
3) **API：/models 动态化（家族/能力信息）**
4) （可选）补测试：新增/更新 pytest 覆盖新行为

---

## 4. 详细设计

### 4.1 安全一致性：`trust_remote_code` 统一策略

#### 4.1.1 单一事实来源（Source of Truth）

设计原则：所有“模型加载”都通过统一函数/参数传递得到 `trust_remote_code`。

优先级（从高到低）：
1. **API request override**（如果 endpoint 显式提供 `trust_remote_code` 字段）
2. **API server default config**（例如 `config/settings.py` 提供的系统默认）
3. **CLI ctx.obj**（由 `archon.cli.main` 注入的全局开关）
4. **GenerationConfig.security.trust_remote_code**（用于生成/批量/内部 pipeline）
5. 默认值 True（保持兼容性），但所有入口都应“可控地改为 False”

#### 4.1.2 需要改造的模块与接口

**A) `core/analyzer.py`**
- 现状：`AutoConfig.from_pretrained(... trust_remote_code=True)` 写死
- 目标：`ModelAnalyzer` 构造函数增加参数 `trust_remote_code: bool = True`，并在内部所有 HF 加载点使用该参数
- 同时调整 `cli/commands/analyze.py`：从 ctx.obj 读取并注入

**B) `cli/commands/evolve.py`**
- 现状：`AutoConfig.from_pretrained(... trust_remote_code=True)` 写死
- 目标：所有 `AutoConfig.from_pretrained` 改为使用 `ctx.obj["trust_remote_code"]`（保持与 CLI 一致）

**C) `webui/app.py`**
- 现状：`load_model_config()` 写死 `trust_remote_code=True`
- 目标：提供 UI 开关（例如 checkbox：`trust_remote_code`），默认值可从环境变量读取（例如 `ARCHON_UI_TRUST_REMOTE_CODE`）或延用 True
- 所有 `AutoConfig/AutoTokenizer` 加载都使用 UI 选择值

**D) `api/server.py`**
- 现状：`GenerateRequest` 已提供可选 `trust_remote_code`；但需确保所有加载点遵从 job 的最终 effective 值
- 目标：在 job.request → build_generation_config(overrides) 阶段明确写入 `trust_remote_code`，后续所有路径不应覆盖

#### 4.1.3 用户可见行为
- CLI：`archon --no-trust-remote-code analyze ...` / `archon --no-trust-remote-code evolve ...` 实际生效
- WebUI：新增开关，可让用户明确选择“是否信任远程代码”
- API：每个 job 可覆盖 trust_remote_code；不传则走 server default

---

### 4.2 API：NAS 从模拟到真实

#### 4.2.1 选择“targeted NAS”作为第一阶段
原因：
- 资源可控（迭代次数可限）
- 与现有 CLI/WebUI 使用的 `ConstraintOptimizer` 一致
- 结果结构化（gene + metrics + constraints + objective）

#### 4.2.2 API 契约（输入/输出）

**输入：** 扩展 `NASRequest`
- `algorithm`: 先仅允许 `targeted`（保守收敛，避免 scope 膨胀）
- `n_iterations`: 10–200（建议进一步收紧上限）
- `constraints`:（沿用已有字段或扩展）例如：
  - `target_vram_gb?: float`
  - `target_params_m?: float`
- `objective`: `minimize-params|minimize-vram|maximize-efficiency`
- `output_dir?: str`（可选，默认为 `<ARCHON_API_OUTPUT_DIR>/nas/<job_id>`；若未设置环境变量则使用 `./api_output/nas/<job_id>`）
  - 安全约束：仅允许相对路径或落在服务端允许的根目录下（防止 path traversal / 任意文件写入）。

**输出：** job result 中包含：
- `best_gene`: `ArchitectureGene.to_dict()`
- `metrics`: 优化器返回 metrics（params_millions, vram_gb, flops_per_param, ...）
- `score`: optimization score
- `artifacts_path`: 产物目录
- `status/progress`: 可观测

#### 4.2.3 后台任务行为
替换 `process_nas_job()`：
1. 从 job.request 读取参数
2. 执行 `ConstraintOptimizer.optimize(LLMSearchSpace(), ...)`
3. 持久化结果到 `artifacts_path/result.json`
4. 写回 job.result（并标记 completed）

#### 4.2.4 进度（progress）策略
由于 `ConstraintOptimizer.optimize` 可能是纯函数式循环且内部不提供 callback：
- 第一阶段可采用“粗粒度进度”：启动 → 50%（开始 optimize）→ 100%（写入结果）
- 若代码结构允许，第二阶段再引入 callback 或每 N 次迭代更新一次 job.progress

---

### 4.3 API：`/models` 动态化

#### 4.3.1 返回内容定位
`/models` 的目标不是“列出所有 HF 模型”，而是返回 Archon 内置的：
- 已知家族（evolution tree families）
- 已知 adapter/兼容性信息（工程支持范围）
- 可能的能力矩阵（例如是否支持某些分析器/补丁路径）

#### 4.3.2 数据来源（可组合）
优先从以下模块汇总：
- `archon.evolution.tree_builder.DEFAULT_FAMILIES`
- `archon.compat.family_matrix`（如果能提供更工程化的矩阵）
- `archon.adapters.registry.AdapterRegistry`（用于列 adapter module 清单/可用性）

#### 4.3.3 API 响应结构（建议）
```json
{
  "families": [{"name": "...", "root": "...", "members": [...], "members_count": 12}],
  "adapters": [{"name": "LlamaAdapter"}, {"name": "QwenMoeAdapter"}],
  "notes": {"trust_remote_code": "...", "source": "..."}
}
```

兼容性建议：
- 为避免现有使用方依赖旧的 `{"models":[...]}` 结构，第一阶段可在响应中**同时保留**：
  - `models`（旧字段，可能为空或映射为 families 摘要）
  - `families` / `adapters`（新字段，作为权威输出）

---

## 5. 测试策略（最小闭环）

新增/调整测试（建议）：
1. `tests/test_api_server.py`：新增 “NAS job returns real payload keys（best_gene/metrics/artifacts_path）” 的断言（不验证最优性，只验证非占位）
2. `tests/test_cli_optional_dependencies.py` 或新增测试：确保 `--no-trust-remote-code` 能传递到 analyze/evolve 路径（可以通过 monkeypatch 断言 AutoConfig 调用参数）
3. WebUI smoke：至少验证 UI app 能启动、且 `load_model_config` 不再写死（可做静态单测或轻量导入测试）

---

## 6. 兼容性与迁移
- CLI 命令与参数保持不变（新增的仅为内部透传与一致性修复）
- API：
  - `/nas/search` 的语义改变（从模拟变真实），但响应结构仍保持 `job_id/status/message`
  - `/models` 输出结构会变化（从样例变真实），属于 breaking-change 风险；建议在响应中保留旧字段一段时间或通过版本号区分（如果已有 API version 约定）

---

## 7. 风险与缓解

1) **真实 NAS 计算资源占用**
   - 缓解：收紧 iterations 上限；仅支持 targeted；必要时增加 server-side 超时与并发限制
2) **`trust_remote_code=False` 导致兼容性下降**
   - 缓解：默认仍 True；但确保用户显式关闭时真的生效；并在输出/日志提示可能兼容性风险
3) **WebUI 引入新开关造成使用困惑**
   - 缓解：在 UI 文案提示“更安全但可能不兼容”，默认保持现状（True）

---

## 8. 实施里程碑（Definition of Done）

M1（安全一致性）：analyze/evolve/webui/api 所有加载点均不再写死 trust_remote_code=True  
M2（API NAS 真实性）：/nas/search 返回真实 gene/metrics；job artifacts 落盘；失败可追踪  
M3（API models 可信）：/models 动态生成 families + adapters 信息  
M4（测试闭环）：至少新增/调整 2 个测试覆盖上述变更
