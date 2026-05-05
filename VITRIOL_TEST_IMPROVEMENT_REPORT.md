# Vitriol 测试与验证完善报告

**日期**: 2026-05-01
**状态**: 全部通过

---

## 1. 测试覆盖完善

### 新增测试文件（8个）

| 测试文件 | 覆盖模块 | 测试数量 |
|----------|----------|----------|
| `tests/test_core_utils.py` | core/incremental, core/hasher, core/smart_initializer | 34 |
| `tests/test_utils_capabilities.py` | utils/model_capabilities, strategy_discovery, fingerprint | 71 |
| `tests/test_metrics_compression.py` | metrics/compression_intelligence | 52 |
| `tests/test_registry_model_store.py` | registry/model_store | 28 |
| `tests/test_evolution_timeline.py` | evolution/timeline | 15 |
| `tests/test_viz_dashboard.py` | viz/dashboard | 23 |
| `tests/test_tools_comparator.py` | tools/comparator | 15 |
| `tests/test_boundary_conditions.py` | 边界条件/异常处理/极端值 | 39 |
| **合计** | | **277** |

### 测试总数变化

- **原始**: 507 个测试
- **新增**: 260 个有效测试（部分文件有少量重叠）
- **最终**: **767 个测试全部通过**，17 个跳过，0 个失败

---

## 2. 发现的代码 Bug 及修复（7个）

### Bug 1: TurboQuantum 张量维度不匹配
- **文件**: `src/vitriol/kv/turboquantum.py` 第721-723行
- **问题**: `k_sigma.view(b*h, s, 1).expand(b, h, s, d)` 中 `b*h != b`
- **修复**: 改为 `k_sigma.view(b, h, s, 1).expand(b, h, s, d)`

### Bug 2: PyTorch SVD API 变更
- **文件**: `src/vitriol/metrics/compression_intelligence.py`
- **问题**: `torch.linalg.svd().singular_values` 在新版 PyTorch 中不存在
- **修复**: 改为 `.S`（3处：svd_preservation_score, rank_score, gradient_flow_score）

### Bug 3: 存储效率分数计算错误
- **文件**: `src/vitriol/metrics/compression_intelligence.py`
- **问题**: `storage_score_from_ratio(0.01)` 返回 0.5，但 99% 压缩应得高分
- **修复**: 修正归一化公式，0.01 -> ~1.0

### Bug 4: Entropy 计算返回负值
- **文件**: `src/vitriol/metrics/compression_intelligence.py`
- **问题**: `np.histogram(..., density=True)` 导致 uniform 数据 entropy 为负
- **修复**: 先计算 count histogram，再归一化为概率分布

### Bug 5: PSI 分数可能超过 1.0
- **文件**: `src/vitriol/metrics/compression_intelligence.py`
- **问题**: `compute_info_preservation` 中 entropy 未归一化，psi 可达 3.5+
- **修复**: entropy 除以 `ln(50)` 归一化到 [0, 1]

### Bug 6: CriticalPointDetector 默认逻辑反了
- **文件**: `src/vitriol/metrics/compression_intelligence.py`
- **问题**: `is_above_critical_point(0.95)` 返回 False（默认 `< 0.9`）
- **修复**: 改为 `compression_ratio > 0.9`

### Bug 7: cfg_int 对 None 值处理错误
- **文件**: `src/vitriol/utils/model_capabilities.py`
- **问题**: `cfg_int({"n": None}, "n", 3)` 返回 0 而非 3
- **修复**: 显式检查 None，返回 default

### Bug 8: strategy_discovery 未递归遍历
- **文件**: `src/vitriol/utils/strategy_discovery.py`
- **问题**: 仅遍历 AST 顶层节点，if 语句内的 `STRATEGY_REGISTRY["sparse"] = ...` 被忽略
- **修复**: 递归遍历所有子节点

---

## 3. 验证结果

```
============================= test session starts ==============================
platform darwin -- Python 3.11.8
pytest: 767 passed, 17 skipped, 0 failed
```

### 覆盖模块
- **core/**: incremental, hasher, smart_initializer
- **utils/**: model_capabilities, strategy_discovery, fingerprint
- **metrics/**: compression_intelligence (含 4 个指标类别)
- **registry/**: model_store, model_registry
- **evolution/**: timeline
- **viz/**: dashboard
- **tools/**: comparator
- **边界条件**: 空输入、None、极端值、异常维度

---

## 4. 产出文件

- `tests/test_core_utils.py`
- `tests/test_utils_capabilities.py`
- `tests/test_metrics_compression.py`
- `tests/test_registry_model_store.py`
- `tests/test_evolution_timeline.py`
- `tests/test_viz_dashboard.py`
- `tests/test_tools_comparator.py`
- `tests/test_boundary_conditions.py`
