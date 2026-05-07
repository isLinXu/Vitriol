# KV Hook Stability & Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变默认推理/量化算法行为的前提下，提升 KV-Cache hook 的兼容性与可观测性，并修复 pytest/可选依赖导致的开箱体验问题。

**Architecture:**  
1) KV hook 维持现有 `CacheHookPatcher + UniversalAttentionPatcher` 机制；新增全局、线程安全的 stats 计数器与显式“是否支持 patch”的检测。  
2) 将 KV hook stats 以“只读附加信息”的方式暴露给 CLI（`vitriol infer --show-stats`），不改变推理路径。  
3) 调整测试配置：移除默认 `pytest -n1` 的硬依赖；对可视化测试的可选依赖（plotly）做 `importorskip`。

**Tech Stack:** Python, PyTorch, HuggingFace Transformers, pytest

---

## File Map (Create/Modify)

**Modify**
- `src/vitriol/patches/cache_hooks.py`：新增 hook 兼容性探测与统计计数（命中/回退/异常）。
- `src/vitriol/cli/commands/infer.py`：在 `--show-stats` 输出中追加 hook stats（仅展示，不参与推理）。
- `pyproject.toml`：pytest 默认参数不再硬依赖 `-n1`。
- `tests/test_visualization_visualizer.py`：对 plotly 等可选依赖做跳过，避免默认环境失败。

**Create**
- `src/vitriol/kv/utils.py`：新增 `clear_vitriol_kv(handle)` 清理接口（释放挂在 cache handle 上的 vitriol 字段）。
- `tests/test_kv_hook_stats.py`：新增单测，验证 hook stats 计数器可工作且 reset 行为正确。

---

### Task 1: KV Hook stats + 兼容性检测

**Files:**
- Modify: `src/vitriol/patches/cache_hooks.py`
- Test: `tests/test_kv_hook_stats.py` (new)

- [ ] **Step 1: 写一个失败的单测（先定义期望接口）**

创建 `tests/test_kv_hook_stats.py`：

```python
import pytest


def test_cache_hook_stats_api_exists_and_resets():
    from vitriol.patches import cache_hooks

    assert hasattr(cache_hooks, "get_cache_hook_stats")
    assert hasattr(cache_hooks, "reset_cache_hook_stats")

    cache_hooks.reset_cache_hook_stats()
    stats0 = cache_hooks.get_cache_hook_stats()
    assert isinstance(stats0, dict)

    # Manually bump counters through helper (should exist)
    cache_hooks._bump_cache_hook_stat("read_attention_hit", 1)
    stats1 = cache_hooks.get_cache_hook_stats()
    assert stats1["read_attention_hit"] == stats0.get("read_attention_hit", 0) + 1

    cache_hooks.reset_cache_hook_stats()
    stats2 = cache_hooks.get_cache_hook_stats()
    assert stats2.get("read_attention_hit", 0) == 0
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
PYTHONPATH=src python -m pytest -q -c /dev/null tests/test_kv_hook_stats.py
```

Expected: FAIL（找不到 `get_cache_hook_stats/reset_cache_hook_stats/_bump_cache_hook_stat`）。

- [ ] **Step 3: 在 cache_hooks.py 中实现线程安全 stats 计数器**

在 `src/vitriol/patches/cache_hooks.py` 顶部（`logger` / `_thread_local` 附近）添加：

```python
_STATS_LOCK = threading.Lock()
_CACHE_HOOK_STATS: dict[str, int] = {}


def _bump_cache_hook_stat(name: str, delta: int = 1) -> None:
    try:
        d = int(delta)
    except Exception:
        d = 1
    if d == 0:
        return
    with _STATS_LOCK:
        _CACHE_HOOK_STATS[name] = int(_CACHE_HOOK_STATS.get(name, 0)) + d


def get_cache_hook_stats(*, reset: bool = False) -> dict[str, int]:
    with _STATS_LOCK:
        snap = dict(_CACHE_HOOK_STATS)
        if reset:
            _CACHE_HOOK_STATS.clear()
    return snap


def reset_cache_hook_stats() -> None:
    with _STATS_LOCK:
        _CACHE_HOOK_STATS.clear()
```

并在关键路径打点（只计数，不改变逻辑）：

1) `update_wrapped()`：
- 每次进入 `update_wrapped`：`_bump_cache_hook_stat("cache_update_calls")`
- `mode` 为 True 且写入 backend：`_bump_cache_hook_stat("write_kv_calls")`
- prefill (`q_len>1`) 走原 update：`_bump_cache_hook_stat("update_passthrough_prefill")`
- decode (`q_len==1`) 且 `passthrough_update=True`：`_bump_cache_hook_stat("update_passthrough_decode")`
- decode 且不透传：`_bump_cache_hook_stat("update_short_circuit_decode")`

2) `UniversalAttentionPatcher.apply()`：
- 如果 `_supported` 为 False：`_bump_cache_hook_stat("attention_hook_unsupported")`
- patch 成功：`_bump_cache_hook_stat("attention_hook_enabled")`

3) `custom_attention_forward()`：
- 命中 vitriol 路径（`cache is not None and q_len == 1 and mode==True and layer_idx is not None`）：`_bump_cache_hook_stat("read_attention_attempt")`
- `backend.read_attention(...)` 成功：`_bump_cache_hook_stat("read_attention_hit")`
- 捕获异常并 fallback：`_bump_cache_hook_stat("read_attention_fallback")`

- [ ] **Step 4: 重新跑测试**

Run:

```bash
PYTHONPATH=src python -m pytest -q -c /dev/null tests/test_kv_hook_stats.py
```

Expected: PASS

- [ ] **Step 5: 回归跑 KV 相关单测**

Run:

```bash
PYTHONPATH=src python -m pytest -q -c /dev/null tests/test_kv_optimizations.py
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/vitriol/patches/cache_hooks.py tests/test_kv_hook_stats.py
git commit -m "fix: add kv cache hook stats and capability probes"
```

---

### Task 2: 增加显式 KV 清理接口（避免长生命周期累积）

**Files:**
- Create: `src/vitriol/kv/utils.py`
- Modify: `src/vitriol/kv/__init__.py`
- Test: `tests/test_kv_hook_stats.py` (append)

- [ ] **Step 1: 扩展测试（先失败）**

在 `tests/test_kv_hook_stats.py` 追加：

```python
def test_clear_vitriol_kv_removes_handle_fields():
    from vitriol.kv.utils import clear_vitriol_kv

    class Handle:
        pass

    h = Handle()
    h._vitriol_kv_stores = {0: object()}
    h._vitriol_seq_lens = [123]
    h._vitriol_kv_store_mode = True

    clear_vitriol_kv(h)
    assert not hasattr(h, "_vitriol_kv_stores")
    assert not hasattr(h, "_vitriol_seq_lens")
    assert not hasattr(h, "_vitriol_kv_store_mode")
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
PYTHONPATH=src python -m pytest -q -c /dev/null tests/test_kv_hook_stats.py
```

Expected: FAIL（找不到模块/函数）。

- [ ] **Step 3: 实现 clear_vitriol_kv(handle)**

创建 `src/vitriol/kv/utils.py`：

```python
from __future__ import annotations

from typing import Any


def clear_vitriol_kv(handle: Any) -> None:
    """清理挂在 HuggingFace cache handle 上的 vitriol KV 字段，避免长生命周期对象累积占用。"""
    for attr in ("_vitriol_kv_stores", "_vitriol_seq_lens", "_vitriol_kv_store_mode"):
        try:
            if hasattr(handle, attr):
                delattr(handle, attr)
        except Exception:
            # best-effort: do not raise
            pass
```

并在 `src/vitriol/kv/__init__.py` 里导出：

```python
from .utils import clear_vitriol_kv

__all__ = [
    # ...
    "clear_vitriol_kv",
]
```

- [ ] **Step 4: 重新跑测试**

Run:

```bash
PYTHONPATH=src python -m pytest -q -c /dev/null tests/test_kv_hook_stats.py
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/vitriol/kv/utils.py src/vitriol/kv/__init__.py tests/test_kv_hook_stats.py
git commit -m "feat: add explicit clear_vitriol_kv cleanup helper"
```

---

### Task 3: infer --show-stats 输出追加 KV hook stats（只读）

**Files:**
- Modify: `src/vitriol/cli/commands/infer.py`
- Test: `tests/test_cli_optional_dependencies.py` (if needed) 或新增轻量单测

- [ ] **Step 1: 写一个失败的单测（可选，但推荐）**

如果项目已有 CLI 单测框架，可新增 `tests/test_cli_infer_stats.py`（否则跳过此步，直接走手工验证）：

```python
def test_infer_stats_text_includes_hook_stats():
    from vitriol.patches.cache_hooks import reset_cache_hook_stats, _bump_cache_hook_stat
    from vitriol.cli.commands.infer import _stats_text

    reset_cache_hook_stats()
    _bump_cache_hook_stat("read_attention_hit", 3)
    text = _stats_text({"ok": True, "policy_insights": {}, "tuned_memory": {}, "tuned_turboquant": {}})
    assert "kv_hook_stats" in text
    assert "read_attention_hit" in text
```

- [ ] **Step 2: 实现输出拼接**

在 `src/vitriol/cli/commands/infer.py` 的 `_stats_text()` 末尾追加：

```python
    try:
        from ...patches.cache_hooks import get_cache_hook_stats

        hook_stats = get_cache_hook_stats()
        lines.append("kv_hook_stats:")
        # stable order
        for k in sorted(hook_stats.keys()):
            lines.append(f"  {k}: {hook_stats[k]}")
    except Exception:
        lines.append("kv_hook_stats: -")
```

注意：只读展示，不做 reset（避免影响其他调用者）。如需 reset，可另加 `--reset-hook-stats` 开关（不在本低风险包范围内）。

- [ ] **Step 3: 手工验证**

Run:

```bash
vitriol infer <model_id> --prompt "hello" --show-stats --format summary
```

Expected: 输出包含 `kv_hook_stats:` 段落（哪怕计数全 0）。

- [ ] **Step 4: Commit**

```bash
git add src/vitriol/cli/commands/infer.py
git commit -m "chore: surface kv hook stats in infer --show-stats"
```

---

### Task 4: pytest 默认配置去除 -n1 硬依赖；可视化测试对 plotly 做跳过

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/test_visualization_visualizer.py`

- [ ] **Step 1: 更新 pytest addopts**

在 `pyproject.toml` 的 `[tool.pytest.ini_options]` 下，将：

```toml
addopts = "-n1 --ignore=tests/integration"
```

改为：

```toml
addopts = "--ignore=tests/integration"
```

（并在 CI 中需要并行时显式传 `-n auto`，不依赖默认值）

- [ ] **Step 2: 修复 plotly 可选依赖导致的测试失败**

在 `tests/test_visualization_visualizer.py` 顶部添加：

```python
import pytest

pytest.importorskip("plotly")
```

如果还依赖 `matplotlib/seaborn/scipy/pandas` 等可选包，也按需 `importorskip`（只对缺失包跳过，不要改测试逻辑）。

- [ ] **Step 3: 验证**

Run（使用项目默认配置，确认不会因为 -n1 直接挂）：

```bash
PYTHONPATH=src python -m pytest -q -c /dev/null tests/test_visualization_visualizer.py
```

Expected:
- 若未安装 plotly：SKIPPED（不是 ERROR）
- 若已安装 plotly：正常执行（PASS/FAIL 依据测试本身）

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml tests/test_visualization_visualizer.py
git commit -m "test: remove pytest-xdist default and skip optional viz deps"
```

---

## Plan Self-Review Checklist

- 覆盖性：本计划覆盖 KV hook stats、hook 兼容性探测、infer stats 输出、清理接口、pytest/plotly 体验修复。  
- 无占位符：每个任务包含明确文件路径、代码片段与命令。  
- 不改变行为：所有改动均为计数/展示/清理与测试配置，不改变推理结果。

