# P0 可视化真实性修复 — PR 说明/变更清单

Generated: 2026-04-25  
Scope: P0（严格阻断隐式 Demo 回退、参数量口径修正、采样可复现）

## 1. 背景 & 目标（Why）
本次 P0 的目标是消除“可视化输出看起来很真实但实际上来自默认/硬编码/隐式回退”的风险，并让统计结果具备可复现性与可追溯性（provenance）。

## 2. 用户可见行为变化（What）

### 2.1 3D 模型结构可视化：加载失败 **不再** 自动展示默认模型/硬编码参数
**文件：** `src/vitriol/viz/model_3d_visualizer.html`

变更要点：
- 当 `config.json/meta-config.json` 加载失败时：进入 **BLOCKED** 错误态，不再生成任何“看似真实”的关键指标。
- 仅当用户在 URL hash 显式启用 `demo=1`（例如：`#?demo=1`）时，才进入 DEMO 模式，并显示明确标识。

验收要点：
- 不存在 `getDefaultConfigForPath()` 的隐式回退逻辑（也就不会出现 397B/7B 等硬编码参数量）。
- BLOCKED 时 UI 明确提示如何修复输入问题（路径/HTTP server）。

### 2.2 权重统计：明确区分“模型总参数量” vs “展示层估算参数量”，并输出来源元数据
**文件：** `src/vitriol/viz/weight_inspector.py`

`generate_viz_data()` 输出新增/调整字段：
- `model_total_params`: int（优先来自 `ArchitectureAnalyzer.total_params`）
- `display_params_estimate`: int（仅对“展示层/采样层”求和的估算值）
- `total_params`: int（向后兼容：优先等于 `model_total_params`，不可得时退化为 `display_params_estimate`）
- `params_source`: `"analyzer"` / `"config_derived"`（参数量来源）
- `sampling`: `{enabled, method, sample_size, seed}`（统计采样元数据）

### 2.3 采样可复现（默认 seed=42，可 CLI 覆盖）
涉及：
- `src/vitriol/viz/weight_inspector.py`：统计采样改为确定性采样（固定 seed）
- `src/vitriol/visualization/visualizer.py`：大张量采样改为确定性 + 固定遍历顺序（排序 keys）
- `src/vitriol/cli/commands/weight_viz.py`：新增 `--seed` 参数并透传

示例：
```bash
vitriol weight-viz -m /path/to/model --seed 42
```

## 3. 文件级变更清单（Files changed）

### P0-1（前端）
- 修改：`src/vitriol/viz/model_3d_visualizer.html`
  - 删除隐式回退 `getDefaultConfigForPath()`
  - 新增：`isDemoEnabled()`、`showDemoIndicator()`、`showBlockedIndicator()`
  - 配置加载失败时：仅在 `demo=1` 下允许 DEMO；否则 BLOCKED + 抛错
  - diffusion 分支：不再使用 `1000000000` 作为“看似真实”的占位 total_params

### P0-2/P0-3（后端/CLI/统计）
- 修改：`src/vitriol/viz/weight_inspector.py`
  - `generate_viz_data(..., seed=42, sample_size=1_000_000)` 支持 seed 与采样大小
  - 输出新增 `model_total_params/display_params_estimate/params_source/sampling`
  - `_compute_tensor_stats()` 采样逻辑改为确定性（`torch.Generator` + `randint`）
  - 增加 safetensors header-only 读取能力（torch 不可用时仍可读取 shape/numel）
- 修改：`src/vitriol/visualization/visualizer.py`
  - `WeightVisualizer` 增加 `seed/sample_size`，并对大张量做确定性采样
- 修改：`src/vitriol/cli/commands/weight_viz.py`
  - 新增 `--seed` 参数，透传到 `generate_viz_data()`
- 修改：`src/vitriol/cli/commands/viz.py`
  - 同步 `generate_viz_data()` 新签名（目前固定使用 seed=42）

### 测试
- 新增：`tests/test_viz_p0_truthfulness.py`
  - 覆盖：3D 视觉化不再隐式回退、demo=1 为显式入口、weight_inspector 元数据字段存在、seed 可复现

## 4. 测试与验证（How we know）
在当前环境中建议的验收命令：
```bash
python -m pytest tests/test_viz_p0_truthfulness.py -q
PYTHONPATH=src python -m pytest tests/test_weight_inspector_metadata.py::test_weight_inspector_reads_safetensors_header_without_loading_torch -q
```

说明：
- 本仓库存在大量与本次改动无关的既有测试失败（运行时依赖/torch 相关），因此本次 P0 以**关键路径用例**作为验收基线。

## 5. 兼容性 & 升级注意事项
- `weight_inspector.generate_viz_data()` 新增关键字参数 `seed/sample_size`，已同步主要调用方（`weight_viz`/`viz`）。
- 仍保留 `total_params` 字段以兼容旧前端/脚本，但语义变更为“模型总参数量优先”。
- DEMO 不再隐式触发：需要显式 `demo=1`。

## 6. 回滚策略
- 若出现用户强依赖旧“默认回退展示”的行为，可临时恢复为显式 demo（`#?demo=1`），不建议再回到隐式回退。

## 7. 后续建议（P1，不在本次范围）
- entropy 等统计指标的口径更正（密度 vs 概率）
- arch_viz 中 approx/placeholder 的统一显式标识体系
- Dashboard 指标 schema_version/单位/来源展示

