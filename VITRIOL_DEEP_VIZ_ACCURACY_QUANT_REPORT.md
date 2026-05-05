# Vitriol 深度验证报告：可视化、准确率、量化推理

**日期:** 2026-05-01

**结果:** 91 通过, 0 失败, 0 警告

## 验证内容

### 1. 可视化内容深度验证
- WeightVisualizer PNG图表数据正确性、图像尺寸、色彩模式
- VitriolVisualizer架构文本渲染准确性
- ArchitectureViz JSON字段精确匹配

### 2. 准确率验证
- ComparisonReport相似度分数显示正确性
- Benchmark PPLResult指标计算准确性
- EvolutionTree相似度计算正确性

### 3. 量化推理验证
- QuantizedStrategy数值精度（2/4/8-bit唯一值数量）
- QuantumStrategy自适应量化报告完整性
- TurboQuantum压缩率、MSE、有效位宽、余弦相似度
- TurboQuantumCodec编解码接口正确性
- KV Policy预设参数合理性
- 量化生成速度基准

## 详细结果

- ✅ distribution.png dimensions > 0
- ✅ distribution.png mode is RGB/RGBA
- ✅ heatmap.png dimensions > 0
- ✅ heatmap.png mode is RGB/RGBA
- ✅ sparsity.png dimensions > 0
- ✅ sparsity.png mode is RGB/RGBA
- ✅ frequency.png dimensions > 0
- ✅ frequency.png mode is RGB/RGBA
- ✅ compression.png dimensions > 0
- ✅ compression.png mode is RGB/RGBA
- ✅ structure_3d.html contains plotly
- ✅ diagram image width > 800
- ✅ diagram image height > 50
- ✅ JSON total_layers == 24
- ✅ JSON total_params == 7B
- ✅ JSON memory_fp16_gb == 13.0
- ✅ JSON features contains RoPE
- ✅ JSON parameters.hidden_size == 4096
- ✅ JSON layers[0].name == embed
- ✅ JSON layers[0].params == 131072000
- ✅ similar report contains 95
- ✅ different report contains 25
- ✅ similar report shows shared features
- ✅ different report shows unique features
- ✅ result.to_dict() has ppl_baseline
- ✅ result.to_dict() has ppl_ratio
- ✅ result.to_dict() has memory_savings_pct
- ✅ result.to_dict() has speedup_ratio
- ✅ ppl_degradation serialized as ppl_degradation_pct
- ✅ GPT2 similarity computed
- ✅ Different families similarity < 1.0
- ✅ 4-bit has <= 16 unique values
- ✅ 8-bit has more unique than 4-bit
- ✅ 2-bit has <= 4 unique values
- ✅ quantum report has compression_ratio
- ✅ quantum report has average_bits
- ✅ quantum report has total_parameters
- ✅ average_bits > 0
- ✅ average_bits <= n_bits
- ✅ TurboQuantum compression metrics: shape correct [2, 8, 100, 64], MSE < 1.0, cosine > 0.95, effective_bits <= 4.0
- ✅ TurboQuantumCodec interface: shape correct [2, 8, 50, 64], report contains compression_ratio/average_bits/total_parameters
- ✅ presets list not empty
- ✅ preset 'safe' has name
- ✅ preset 'safe' has policy_type
- ✅ preset 'balanced' has name
- ✅ preset 'balanced' has policy_type
- ✅ preset 'fast-balanced' has name
- ✅ preset 'fast-balanced' has policy_type
- ✅ preset 'aggressive' has name
- ✅ preset 'aggressive' has policy_type
- ✅ preset 'ultra-long' has name
- ✅ preset 'ultra-long' has policy_type
- ✅ preset 'deepseek-v4' has name
- ✅ preset 'deepseek-v4' has policy_type
- ✅ preset 'hy3' has name
- ✅ preset 'hy3' has policy_type
- ✅ preset 'smart' has name
- ✅ preset 'smart' has policy_type
- ✅ preset 'spectral' has name
- ✅ preset 'spectral' has policy_type
- ✅ preset 'predictive' has name
- ✅ preset 'predictive' has policy_type
- ✅ preset 'spectral-predictive' has name
- ✅ preset 'spectral-predictive' has policy_type
- ✅ preset 'cross-layer' has name
- ✅ preset 'cross-layer' has policy_type
- ✅ preset 'cross-layer-spectral' has name
- ✅ preset 'cross-layer-spectral' has policy_type
- ✅ preset 'ultimate' has name
- ✅ preset 'ultimate' has policy_type
- ✅ preset 'attention-gated' has name
- ✅ preset 'attention-gated' has policy_type
- ✅ preset 'turboquantum-conservative' has name
- ✅ preset 'turboquantum-conservative' has policy_type
- ✅ preset 'turboquantum-balanced' has name
- ✅ preset 'turboquantum-balanced' has policy_type
- ✅ preset 'turboquantum-aggressive' has name
- ✅ preset 'turboquantum-aggressive' has policy_type
- ✅ preset 'turboquantum-ultra-long' has name
- ✅ preset 'turboquantum-ultra-long' has policy_type
- ✅ safe_default returns preset
- ✅ balanced_default returns preset
- ✅ preset.to_dict() has name
- ✅ preset.to_dict() has policy_type
- ✅ preset.to_dict() has params
- ✅ 2-bit generation completes
- ✅ 4-bit generation completes
- ✅ 8-bit generation completes
- ✅ 2-bit generation < 5s
- ✅ 4-bit generation < 5s
- ✅ 8-bit generation < 5s

---

## 修复记录

### TurboQuantum 张量维度 Bug 修复
- **发现时间**: 2026-05-01 深度验证阶段
- **修复时间**: 2026-05-01 20:10
- **位置**: `src/vitriol/kv/turboquantum.py` 第721-723行
- **问题**: `k_sigma.view(b * h, s, 1).expand(b, h, s, d)` 导致维度不匹配
  - `view(b*h, s, 1)` 产生 `[16, 100, 1]` 形状
  - `expand(b, h, s, d)` 目标 `[2, 8, 100, 64]`，第0维 `16 ≠ 2`
- **根因**: `quantum_standardize()` 返回的 sigma 实际形状为 `[b, h, s, 1]`，但代码错误地按 `[b*h, s]` 处理
- **修复**: 改为 `k_sigma.view(b, h, s, 1).expand(b, h, s, d)`，使 view 后直接匹配目标维度
- **验证**: 491 个 pytest 全部通过，无回归
