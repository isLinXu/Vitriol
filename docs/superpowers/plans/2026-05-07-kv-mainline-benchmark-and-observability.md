# KV Mainline Benchmark + Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Goal:** 先把 Vitriol 的 KV 主链路（Store/Hook + KVCacheStore packed）收敛为默认可回归基准；再补齐 run_id + metrics/trace/dashboard 闭环；最后审查 hermes-agentic-rl 是否满足基于 hermes-agent 的 RL 优化要求。
>
> **Architecture:** Phase 1 以 `CacheHookPatcher + UniversalAttentionPatcher + KVStoreBackend + KVCacheStore` 为默认推理/压缩路径，并形成稳定 JSON 基准输出；Phase 2 引入 `RunContext(run_id)` 与统一 metrics 导出；Phase 3 仅做 readiness audit 报告与差距清单，不改算法实现。
>
> **Tech Stack:** Python, PyTorch, transformers, click CLI, pytest, Prometheus text metrics, stdlib HTTP/SSE dashboard

---

## 0) 代码范围地图（会被修改/新增的文件）

### Phase 1（KV 主链路收敛 + 基准输出）
**Modify**
- `src/vitriol/kv/cache_store.py`（KVCacheStoreConfig / raw 缓存策略 / seq_len 推导）
- `src/vitriol/bench/runner.py`（run_smoke / run_generate_preset / memory stats 输出字段）
- `src/vitriol/cli/commands/infer.py`（json 输出稳定字段；与 runner 对齐）
- `tests/`（新增 KVStore/KVCacheStore 的单测；新增 benchmark 输出字段的回归测试）

**Create**
- `tests/test_kv_store_keep_raw_cache.py`（KVCacheStore keep_raw_cache 行为单测）
- `tests/test_bench_run_summary_fields.py`（runner 返回结构回归测试）

### Phase 2（run_id + metrics/trace/dashboard 闭环）
**Modify**
- `src/vitriol/cli/commands/infer.py`（run_id 注入；可选 metrics 输出）
- `src/vitriol/cli/commands/trace.py`（trace.v1 增加 run_id + kv_summary 可选）
- `src/vitriol/bench/runner.py`（run_id 注入；统一 stats 聚合口径）
- `src/vitriol/telemetry/metrics.py`（增加 run_id label 支持/辅助 API）
- `src/vitriol/viz/dashboard.py`（显示当前 run_id；最小过滤能力）

**Create**
- `src/vitriol/telemetry/run_context.py`（RunContext 定义与生成）
- `src/vitriol/telemetry/kv_stats.py`（把 cache_hooks/kv_backend/turboquant/runtime_patch 统一成一份 dict）
- `tests/test_run_context.py`
- `tests/test_trace_includes_run_id.py`

### Phase 3（hermes-agentic-rl readiness audit）
**Create**
- `docs/superpowers/reports/2026-05-07-hermes-agentic-rl-readiness-audit.md`

---

## Phase 1：KV 主链路收敛 + 压测基准（先做）

### Task 1：定义并输出 KV 路径口径（compute_path / storage_path）

**Files:**
- Modify: `src/vitriol/bench/runner.py:1240-1434`（`run_smoke` / `run_generate_preset`）
- Modify: `src/vitriol/cli/commands/infer.py:197-304`（infer 命令 json 输出）
- Test: `tests/test_bench_run_summary_fields.py`（新增）

**目标：**
- 在 `run_smoke` 与 `run_generate_preset` 的返回结构中稳定输出：
  - `kv.compute_path`：固定为 `"store_hook"`（当前 runner 使用 `_apply_vitriol_universal`）
  - `kv.storage_path`：当启用 TurboQuant packed 时为 `"packed"`；否则 `"raw"`
  - `kv.estimated_kv_bytes` 与 `kv.layer_stats`：从现有 `tuned_memory` 迁移/补齐到 `kv` 下（保留旧字段一段时间以兼容）。

- [ ] **Step 1: 写回归测试（先失败）**

创建 `tests/test_bench_run_summary_fields.py`：

```python
from vitriol.bench.runner import _benchmark_memory_stats


def test_benchmark_memory_stats_shape_is_stable() -> None:
    # 这是纯结构测试：不依赖真实模型推理，避免 CI 环境波动。
    # 只验证我们在 runner 里承诺输出的关键字段形状稳定。
    dummy = {"_final_past_key_values": None}
    out = _benchmark_memory_stats(dummy, backend=None, device=type("D", (), {"type": "cpu"})())
    assert "estimated_kv_bytes" in out
    assert "layer_stats" in out
```

> 注：上面 dummy device 是最小替身；若实现中需要 torch.device，届时在实现步骤里把该测试调整为直接使用 `torch.device("cpu")`。

- [ ] **Step 2: 运行测试，确认失败（或提示需要调整）**

Run:
```bash
pytest -q tests/test_bench_run_summary_fields.py -q
```

Expected:
- 初版可能 FAIL（例如 dummy device 不兼容），按实现改成最小可用形态后再让其 PASS。

- [ ] **Step 3: 在 runner 增加 kv 字段并保持向后兼容**

在 `src/vitriol/bench/runner.py`：
- `run_smoke` 返回 dict（约 `1319-1334`）添加：

```python
"kv": {
    "compute_path": "store_hook",
    "storage_path": "packed" if bool(tuned_cfg.enable_turbo_quant) else "raw",
    "estimated_kv_bytes": int((tuned_memory or {}).get("estimated_kv_bytes", 0) or 0),
    "layer_stats": (tuned_memory or {}).get("layer_stats", {}) or {},
},
```

- `run_generate_preset` 返回 dict（约 `1415-1434`）同样添加 `kv` 字段。

并暂时保留旧字段：
- `tuned_memory`（用于旧版 CLI 输出/兼容）

- [ ] **Step 4: infer 命令 json 输出保持稳定**

在 `src/vitriol/cli/commands/infer.py`（约 `283-285`）：
- 保持 `--format json` 直接输出 runner 的 dict（现状）
- 但增加一个“稳定字段兜底”：如果未来字段缺失，至少保证 `kv` key 存在（实现时用 `setdefault`）。

示例（实现时按项目风格写）：
```python
if fmt == "json":
    result.setdefault("kv", {})
    click.echo(json.dumps(result, ensure_ascii=False, indent=2))
    return
```

- [ ] **Step 5: 运行相关测试**

Run:
```bash
pytest -q tests/test_bench_run_summary_fields.py -q
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/vitriol/bench/runner.py src/vitriol/cli/commands/infer.py tests/test_bench_run_summary_fields.py
git commit -m "feat(kv): add stable kv compute/storage path fields to bench outputs"
```

---

### Task 2：引入 keep_raw_cache 并修复“raw 常驻抵消压缩收益”的口径问题

**Files:**
- Modify: `src/vitriol/kv/cache_store.py`（`KVCacheStoreConfig` + `KVCacheStore.set_prefill/append/seq_len`）
- Modify: `src/vitriol/kv/backend.py`（如需：在 stats 中增加 storage_path 或额外字段）
- Test: `tests/test_kv_store_keep_raw_cache.py`（新增）

**目标：**
- 为 `KVCacheStoreConfig` 增加 `keep_raw_cache: bool = True`
- 当 `keep_raw_cache=False` 且 KV 已完成编码（`_k_enc/_v_enc` 存在）时：
  - `_k_raw/_v_raw` 允许被丢弃（置 None），避免“统计显示节省但实际没省”的情况
  - `seq_len` 必须仍可正确推导（从 encoded 或显式计数器）
- 最小可用策略（Phase 1 允许保守）：当 `keep_raw_cache=False` 且启用 eviction/需要重建时，可选择：
  - A) 直接禁止 eviction（抛出清晰异常或自动关闭 eviction）
  - B) decode → concat → re-encode（更慢但正确）
  
本计划推荐 **A（先正确口径，后续再优化）**：keep_raw_cache=False 时自动禁用 eviction，并在文档/日志里说明。

- [ ] **Step 1: 写单测（先失败）**

创建 `tests/test_kv_store_keep_raw_cache.py`：

```python
import torch

from vitriol.kv.cache_store import KVCacheStore, KVCacheStoreConfig


def _make_kv(b=1, h=2, s=4, d=8, device="cpu"):
    k = torch.randn(b, h, s, d, device=device, dtype=torch.float32)
    v = torch.randn(b, h, s, d, device=device, dtype=torch.float32)
    return k, v


def test_keep_raw_cache_false_drops_raw_after_prefill() -> None:
    cfg = KVCacheStoreConfig(
        enable_turbo_quant=True,
        turbo_k_format="turbo3",
        turbo_v_format="turbo3",
        turbo_block_size=8,
    )
    # 实现后将支持 keep_raw_cache=False
    cfg.keep_raw_cache = False  # type: ignore[attr-defined]

    store = KVCacheStore(cfg)
    k, v = _make_kv()
    store.set_prefill(k, v)

    # raw 应可被丢弃；encoded 仍存在
    assert store.estimated_kv_bytes() > 0
    assert getattr(store, "_k_raw", None) is None
    assert getattr(store, "_v_raw", None) is None
    assert store.seq_len == 4
```

- [ ] **Step 2: 运行测试，确认失败**

Run:
```bash
pytest -q tests/test_kv_store_keep_raw_cache.py -q
```

Expected: FAIL（因为 keep_raw_cache 尚未实现）

- [ ] **Step 3: 在 KVCacheStoreConfig 增加字段并在 set_prefill 结束时丢弃 raw**

在 `src/vitriol/kv/cache_store.py` 的 `KVCacheStoreConfig` dataclass（约 `49-129`）增加：

```python
keep_raw_cache: bool = True
```

在 `KVCacheStore.set_prefill()` 末尾（调用 `_rebuild_encoded_cache()` 后）增加：

```python
if not bool(self.cfg.keep_raw_cache):
    self._k_raw = None
    self._v_raw = None
```

- [ ] **Step 4: 修复 seq_len 推导（raw 为空时仍能工作）**

修改 `KVCacheStore.seq_len` property（当前依赖 `_k_raw`）：

建议逻辑：
1) 若 `_k_raw` 存在：返回 `_k_raw.size(-2)`（保持现状）  
2) 否则若 `_k_enc` 是 `PackedKVTensor` / `ResidualQJLPackedTensor`：从 `orig_shape[-2]` 推导  
3) 否则若 `_k_enc` 是 `torch.Tensor`：返回 `size(-2)`  
4) 其它：返回 0

- [ ] **Step 5: keep_raw_cache=False 下的 append 最小正确性**

当 raw 被丢弃后，`append()` 当前会走 `torch.cat([self._k_raw, ...])` 直接崩溃。

Phase 1 最小策略：
- 如果 `keep_raw_cache=False`，则在 `append()` 里跳过维护 raw，只维护 enc：
  - 若 enc 是 packed：走 `_encode_tensor(new)` + `_concat_packed(existing, new)`  
  - 若 enc 是 tensor：走 `torch.cat([enc, encoded_new], dim=-2)`  

> 这里允许牺牲“可重建/可回退”能力；并在 config 文档说明 keep_raw_cache=False 时不支持 eviction 重建。

建议追加一个 append 单测（可与 Step 1 合并），例如再 append 一个 token，断言 seq_len+1。

- [ ] **Step 6: 运行测试**

Run:
```bash
pytest -q tests/test_kv_store_keep_raw_cache.py -q
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/vitriol/kv/cache_store.py tests/test_kv_store_keep_raw_cache.py
git commit -m "feat(kv): add keep_raw_cache to avoid raw cache negating packed savings"
```

---

### Task 3：把 Phase 1 基准“质量-性能-存储”最小回归固化到 tests

**Files:**
- Modify/Create: `tests/test_bench_run_summary_fields.py`（扩展断言）

**目标：**
不依赖真实 HuggingFace 下载的前提下，尽可能在 unit tests 层验证：
- 结构字段稳定（已有）  
- KVCacheStore 的 `estimated_kv_bytes` 随 turbo_quant 启用而变化（已在 Task 2 单测覆盖）  

> 说明：任何依赖网络/大模型的 e2e bench 放到 docs 或 scripts，不放 CI 单测。

- [ ] **Step 1: 增强结构测试断言**

在 `tests/test_bench_run_summary_fields.py` 增加：
- `estimated_kv_bytes` 为 int
- `layer_stats` 为 dict

- [ ] **Step 2: 运行全量 tests（可先跑子集）**

Run:
```bash
pytest -q tests/test_bench_run_summary_fields.py tests/test_kv_store_keep_raw_cache.py -q
```

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_bench_run_summary_fields.py
git commit -m "test(kv): stabilize minimal kv benchmark regression checks"
```

---

## Phase 2：观测闭环（run_id + metrics + trace + dashboard）

### Task 4：新增 RunContext（run_id）并在 infer/bench/ppl/trace 注入

**Files:**
- Create: `src/vitriol/telemetry/run_context.py`
- Modify: `src/vitriol/bench/runner.py:1240-1434`（run_smoke/run_generate_preset 增加 run_id）
- Modify: `src/vitriol/cli/commands/infer.py:197-304`
- Modify: `src/vitriol/cli/commands/trace.py`（trace.v1 增加 run_id）
- Test: `tests/test_run_context.py`（新增）
- Test: `tests/test_trace_includes_run_id.py`（新增）

- [ ] **Step 1: 写 RunContext 单测（先失败）**

创建 `tests/test_run_context.py`：

```python
from vitriol.telemetry.run_context import new_run_id


def test_new_run_id_is_non_empty_and_stable_shape() -> None:
    rid = new_run_id()
    assert isinstance(rid, str)
    assert len(rid) >= 8
```

- [ ] **Step 2: 实现 run_context.py**

创建 `src/vitriol/telemetry/run_context.py`：

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import uuid


def new_run_id() -> str:
    # 格式：YYYYMMDD-HHMMSS-<8位uuid>
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:8]
    return f"{ts}-{short}"


@dataclass(frozen=True)
class RunContext:
    run_id: str
    model_id: str
    preset: str
    device: str
    dtype: str
```

- [ ] **Step 3: runner 注入 run_id**

在 `src/vitriol/bench/runner.py`：
- `run_smoke` 返回 dict（约 `1319+`）加入 `run_id: new_run_id()`
- `run_generate_preset` 返回 dict（约 `1415+`）加入 `run_id`

并确保 `infer --format json` 能拿到 run_id。

- [ ] **Step 4: trace.v1 增加 run_id**

在 `src/vitriol/cli/commands/trace.py` 的 `_build_trace_v1()` 返回结构中增加 `run_id` 字段：

```python
"run_id": run_id,
```

并在命令入口 `trace()` 里生成 `run_id = new_run_id()` 传入。

- [ ] **Step 5: trace 单测（结构测试）**

创建 `tests/test_trace_includes_run_id.py`：

```python
from vitriol.cli.commands.trace import _build_trace_v1


def test_trace_v1_includes_run_id() -> None:
    trace = _build_trace_v1(
        model_path="x",
        prompt="hi",
        max_new_tokens=1,
        prompt_token_ids=[1],
        prompt_tokens=["hi"],
        generated_token_ids=[2],
        generated_tokens=["ok"],
        events=[],
    )
    # 实现后在 _build_trace_v1 里 set run_id
    assert "run_id" in trace
```

> 注：实现时若改为参数传入 run_id，则这里改为传参并断言相等。

- [ ] **Step 6: 运行测试**

Run:
```bash
pytest -q tests/test_run_context.py tests/test_trace_includes_run_id.py -q
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/vitriol/telemetry/run_context.py src/vitriol/bench/runner.py src/vitriol/cli/commands/trace.py tests/test_run_context.py tests/test_trace_includes_run_id.py
git commit -m "feat(telemetry): add run_id and propagate to bench/infer/trace"
```

---

### Task 5：统一 KV 统计聚合（cache_hooks / kv_backend / turboquant / runtime_patch）并导出 metrics

**Files:**
- Create: `src/vitriol/telemetry/kv_stats.py`
- Modify: `src/vitriol/telemetry/metrics.py`（支持 run_id labels 或便捷 API）
- Modify: `src/vitriol/bench/runner.py`（在返回结构中增加 `stats` 聚合字段）
- Test: `tests/test_kv_stats_aggregation.py`（新增）

- [ ] **Step 1: 写聚合单测（先失败）**

创建 `tests/test_kv_stats_aggregation.py`：

```python
from vitriol.telemetry.kv_stats import merge_kv_stats


def test_merge_kv_stats_merges_dicts() -> None:
    out = merge_kv_stats({"a": 1}, {"b": 2})
    assert out["a"] == 1
    assert out["b"] == 2
```

- [ ] **Step 2: 实现 kv_stats.py**

创建 `src/vitriol/telemetry/kv_stats.py`：

```python
from __future__ import annotations

from typing import Any, Dict


def merge_kv_stats(*parts: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for p in parts:
        for k, v in (p or {}).items():
            out[k] = v
    return out
```

> Phase 2 只先提供“合并口径”；后续再细化为 Prometheus metric 名称规范。

- [ ] **Step 3: 在 runner 的返回结构补齐 stats 聚合字段**

在 `run_smoke / run_generate_preset` 中增加：

```python
"stats": {
    "cache_hooks": get_cache_hook_stats(),  # 注意：实现时需 import
    "turboquant": tuned_turboquant,
    "kv_store": tuned_memory,
},
```

并在 `infer --show-stats` 的 `_stats_text()` 里继续保留 `kv_hook_stats` 打印（现状已有）。

- [ ] **Step 4: metrics 导出（最小实现）**

为 `telemetry.metrics.MetricsCollector` 增加一个 helper，例如：

```python
def ingest_dict(self, prefix: str, data: dict, labels: dict | None = None): ...
```

实现时：
- 数值型 → gauge/counter（按约定）
- 非数值/嵌套 dict → 暂不导出或做扁平化（phase2 先做最小可用）

- [ ] **Step 5: 运行测试**

Run:
```bash
pytest -q tests/test_kv_stats_aggregation.py -q
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/vitriol/telemetry/kv_stats.py src/vitriol/telemetry/metrics.py src/vitriol/bench/runner.py tests/test_kv_stats_aggregation.py
git commit -m "feat(metrics): aggregate kv stats and provide minimal metrics export helpers"
```

---

### Task 6：dashboard 最小 run_id 展示与过滤（不追求完整产品化）

**Files:**
- Modify: `src/vitriol/viz/dashboard.py`
- Test: 可选（dashboard 属于交互件，单测价值较低）

- [ ] **Step 1: 在 dashboard state 增加 run_id 字段**

在 `DashboardDataStore` 里增加：
- `current_run_id: Optional[str] = None`
- `set_run_id(run_id: str)` 方法
- `get_state()` 返回中包含 `run_id`

- [ ] **Step 2: 在 bench/infer 路径（后续实现阶段）调用 set_run_id**

最小方案：在 CLI 启动 dashboard 时传入 run_id（如果已有脚本入口）；或只支持手动设置。

- [ ] **Step 3: Commit**

```bash
git add src/vitriol/viz/dashboard.py
git commit -m "feat(dashboard): show current run_id in dashboard state"
```

---

## Phase 3：审查 hermes-agentic-rl 是否满足基于 hermes-agent 的 RL 优化要求

### Task 7：生成 readiness audit 报告（pass/blocker/risk/建议）

**Files:**
- Create: `docs/superpowers/reports/2026-05-07-hermes-agentic-rl-readiness-audit.md`
- Inputs (read-only):
  - `output/.../hermes-agentic-rl/README.md`
  - `output/.../hermes-agentic-rl/docs/*`
  - `output/.../hermes-agentic-rl/hermes_rl/envs/*`
  - `output/.../hermes-agentic-rl/hermes_rl/mdp/*`
  - `output/.../hermes-agentic-rl/hermes_rl/algos/*`
  - `output/.../hermes-agentic-rl/hermes_rl/bridge/*`
  - `output/.../hermes-agentic-rl/hermes_rl/eval/*`
  - `output/.../hermes-agentic-rl/scripts/train.py`

- [ ] **Step 1: 收集证据点（逐文件）**

按清单逐项记录证据（路径 + 关键接口/类/函数名），并归档到报告中：
- Env 是否是 gym-style step/reset，是否能驱动真实 hermes-agent runtime 或可插拔 backend
- StateEncoder 是否包含对话/工具/记忆/scratchpad
- ActionSpace 是否支持 ToolCall 参数化、MemoryOp、Delegate、Respond/Terminate
- RewardComposer/约束（Lagrangian/RCPO）是否存在且可配置
- 在线/离线算法与 replay buffer 是否齐全
- EvalHarness / A/B harness 是否存在且可跑
- 部署导出（vLLM/LoRA 热切换）是否有明确接口与示例

- [ ] **Step 2: 输出结论（Pass/Blocker/Risk）**

报告必须包含：
- “是否满足要求”的一句话结论
- blockers（必须修复项）列表
- 风险项与建议（P0/P1/P2）
- 最小 smoke 命令（CPU 可跑）与期望输出

- [ ] **Step 3: 写入报告文件**

创建 `docs/superpowers/reports/2026-05-07-hermes-agentic-rl-readiness-audit.md`

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/reports/2026-05-07-hermes-agentic-rl-readiness-audit.md
git commit -m "docs(report): hermes-agentic-rl readiness audit for hermes-agent RL optimization"
```

---

## Plan 自检（写完计划后执行，实施前无需）

- [ ] **Spec 覆盖检查**：对照 `docs/superpowers/specs/2026-05-07-kv-mainline-benchmark-and-observability-design.md`，确保 Phase 1/2/3 都有任务落地。
- [ ] **占位符扫描**：搜索 “TODO/TBD/后续再说/适当处理” 等字样，全部替换为具体步骤或明确不做。
- [ ] **接口一致性检查**：`run_id` 字段名、`kv.compute_path/storage_path` 字段名在 runner/infer/trace/dashboard 中完全一致。

