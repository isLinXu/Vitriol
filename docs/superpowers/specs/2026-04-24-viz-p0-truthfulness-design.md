Generated: 2026-04-24T00:00Z
Topic: P0 fixes — Visualization accuracy/truthfulness hardening

# 目标
对可视化系统落地 P0 级修复，优先解决“隐式 Demo/硬编码回退导致的伪真展示”、统计结果不可复现、以及字段语义误导（display estimate 冒充 total）。

# 范围（In-Scope）
1. 3D 模型结构可视化：`src/vitriol/viz/model_3d_visualizer.html`
2. 权重统计与 3D 权重可视化：
   - `src/vitriol/viz/weight_inspector.py`
   - `src/vitriol/cli/commands/weight_viz.py`
3. 权重统计采样可复现：
   - `src/vitriol/viz/weight_inspector.py`
   - `src/vitriol/visualization/visualizer.py`

（不在本次范围：P1 项，如 entropy 精确性修正、arch_viz 全面估算标识体系、Dashboard schema 单位体系等。）

# 术语
- **真实（REAL）**：来自 meta-config / analyzer / 权重文件的可追溯结果。
- **估算（ESTIMATED）**：启发式计算、推导或采样统计导致的不确定值。
- **演示（DEMO）**：仅用于演示的静态/硬编码数据。
- **阻断（BLOCK）**：当关键输入不可用时，不再自动生成任何看似真实的关键指标。

# 现状问题（P0）
1. 3D 模型可视化在加载失败时会回退到 `getDefaultConfigForPath()`，并硬编码 `397B/7B` 参数量等，且 UI 不够显式标识为 Demo。
2. `weight_inspector.generate_viz_data()` 的 `total_params` 更接近“展示层估算参数量”，字段语义易误导为全模型参数。
3. 统计采样使用随机采样但无 seed 机制，导致结果不可复现，且输出缺少 sampling 元数据。

# 设计方案

## P0-1：3D 模型可视化严格阻断隐式 Demo 回退
### 行为变更
当 `config.json/meta-config.json` 加载失败时：
1. 不再调用 `getDefaultConfigForPath()`；
2. 页面进入 **错误态（BLOCK）**：
   - Total Params / Layers 等关键指标展示为 `N/A`；
   - 展示可操作错误提示（路径/服务可用性检查建议）；
3. 提供 **显式 Demo 模式入口**：
   - 仅当 URL hash 参数包含 `demo=1`（例如 `#?demo=1`）时，才允许加载 demo config；
   - Demo 模式下必须显著标识（watermark/Badge：`DEMO`）。

### 验收标准
- config/meta 加载失败时，页面上不出现 397B/7B 等硬编码数值；
- Demo 仅在用户显式开启时出现，并清晰标识为 DEMO。

## P0-2：权重统计输出字段语义修正（避免 total_params 误导）
### 行为变更
在 `weight_inspector.generate_viz_data()` 输出中新增/调整字段：
- `model_total_params`: int（优先来自 `ArchitectureAnalyzer`；若失败则为 0）
- `display_params_estimate`: int（当前 layers_data 之和）
- `total_params`: int（为兼容旧消费方，设为 `model_total_params`，若不可得则退化为 `display_params_estimate`）
- `params_source`: string（`analyzer` / `config_derived` / `weights_sampled` / `unknown`）
- `sampling`: object
  - `enabled`: bool
  - `method`: string（`uniform_random`）
  - `sample_size`: int（默认 1_000_000）
  - `seed`: int（默认 42）

### 验收标准
- `total_params` 语义等于“模型总参数量”（可得时），不再默认为展示层估算；
- 输出包含 `params_source` 与 `sampling` 元数据。

## P0-3：采样可复现（默认 seed=42，可 CLI 覆盖）
### 行为变更
1. `weight_inspector._compute_tensor_stats()` 支持传入 seed（默认 42），采样使用确定性 RNG；
2. `WeightVisualizer._flatten_weights()` 支持 seed（默认 42）；
3. `vitriol weight-viz` 新增 `--seed` 参数并透传至统计。

### 验收标准
- 同一输入 + 同一 seed，多次运行输出的统计结果一致（或数值完全一致）；
- 输出中记录 seed。

# 测试计划（最小 P0）
1. 新增/更新单测：
   - 验证 3D HTML 在加载失败分支不会出现 `397000000000`/`7000000000` 文本（或 `getDefaultConfigForPath` 不再用于错误分支）。
   - 验证 `weight_inspector.generate_viz_data()` 输出包含 `model_total_params/display_params_estimate/params_source/sampling`。
   - 验证固定 seed 时统计输出稳定（至少对一个小张量样例 deterministic）。

# 回滚计划
所有改动是向后兼容的：
- `total_params` 字段继续存在；
- Demo 仍可显式开启；
若出现用户强依赖旧回退行为，可在短期内通过 `demo=1` 或环境开关恢复（不建议默认开启）。

