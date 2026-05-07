from __future__ import annotations

from typing import Any


def clear_vitriol_kv(handle: Any) -> None:
    """清理挂在 HuggingFace cache handle 上的 vitriol KV 字段，避免长生命周期对象累积占用。

    说明：
    - 该函数是 best-effort：遇到不可删除的属性不会抛错
    - 不会触发任何推理逻辑分支变化；仅用于显式释放缓存引用
    """
    for attr in ("_vitriol_kv_stores", "_vitriol_seq_lens", "_vitriol_kv_store_mode"):
        try:
            if hasattr(handle, attr):
                delattr(handle, attr)
        except Exception:
            # best-effort: do not raise
            pass

