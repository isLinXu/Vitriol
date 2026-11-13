from __future__ import annotations

from typing import Any


def clear_vitriol_kv(handle: Any) -> None:
    """清理挂在 HuggingFace cache handle 上的 vitriol KV 字段，避免长生命周期对象累积占用。

    说明：
    - 该函数是 best-effort：遇到不可删除的属性不会抛错
    - 同时使待消费的 attention 关联失效；后续 hook 从原生 KV 历史重新同步
    """
    # Invalidate any outstanding tensor binding without retaining the handle.
    try:
        handle._vitriol_cache_epoch = getattr(handle, "_vitriol_cache_epoch", 0) + 1
    except Exception:
        pass
    for attr in (
        "_vitriol_kv_stores", "_vitriol_seq_lens", "_vitriol_kv_store_mode",
        "_vitriol_cache_backend", "_vitriol_disabled_layers", "_vitriol_native_cache_retained",
    ):
        try:
            if hasattr(handle, attr):
                delattr(handle, attr)
        except Exception:
            # best-effort: do not raise
            pass
