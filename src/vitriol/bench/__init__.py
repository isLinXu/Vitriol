"""
vitriol.bench

Note: the bench subpackage depends on heavyweight dependencies such as torch/transformers.
To allow documentation/config/report-related logic (e.g., ppl_evaluator report rendering) to be
imported in lightweight environments, we use lazy imports here and avoid importing runner eagerly
when doing `import vitriol.bench` (runner strongly depends on torch/transformers).
"""

from __future__ import annotations

from .autokv import default_prompt_suite, prefix_match_tokens

_RUNNER_EXPORTS = {
    "RunConfig",
    "build_policy_plan",
    "compare_long_context_preset",
    "compare_short_suite",
    "compare_smoke",
    "diff_policy_plans",
    "run_generate_preset",
    "run_long_context",
    "run_long_context_preset",
    "run_short_suite",
    "run_smoke",
}

_KV_EXPORTS = {"analyze_kv_quantization"}

_TURBOQUANT_EXPORTS = {
    "run_turboquantum_synthetic",
    "compare_turboquantum_modes",
    "run_turboquantum_on_model_kv",
}

_PPL_EXPORTS = {"PPLConfig", "PPLResult", "PPLEvaluator"}


def __getattr__(name: str):  # pragma: no cover
    if name in _KV_EXPORTS:
        from . import runner_kv as _runner_kv

        return getattr(_runner_kv, name)

    if name in _TURBOQUANT_EXPORTS:
        from . import runner_turboquant as _runner_turboquant

        return getattr(_runner_turboquant, name)

    if name in _RUNNER_EXPORTS:
        from . import runner as _runner

        return getattr(_runner, name)

    if name in _PPL_EXPORTS:
        from . import ppl_evaluator as _ppl

        return getattr(_ppl, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "default_prompt_suite",
    "prefix_match_tokens",
    # runner exports (lazy)
    *_RUNNER_EXPORTS,
    *_KV_EXPORTS,
    *_TURBOQUANT_EXPORTS,
    # ppl exports (lazy)
    *_PPL_EXPORTS,
]
