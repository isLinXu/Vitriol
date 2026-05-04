# NAS and PPL Compatibility Notes

This note documents two maintenance contracts that keep Vitriol's research paths aligned with the executable CLI and benchmark code.

## PPL Evaluation

`vitriol.bench.ppl_evaluator.PPLEvaluator` compares a baseline decode against a tuned KV-compressed decode. The tuned branch must use the same preset resolution and hook application path as `vitriol bench`:

```python
from vitriol.bench.runner import _apply_vitriol_universal

_apply_vitriol_universal(
    tuned_cfg,
    v_quantize_only_first_n_layers=int(first_n),
    policy=policy,
    passthrough_update=passthrough_update,
    enable_attention_patch=enable_attention_patch,
)
```

The keyword name is part of the compatibility contract. If the runner signature changes, update the PPL caller and the regression test together. Otherwise the evaluator may catch the hook error and continue without compression, which makes PPL comparisons misleading.

Recommended checks:

```bash
python -m pytest tests/test_ppl_evaluator.py -q
python -m pytest tests/test_cli_bench.py tests/test_cli_infer.py -q
```

## NAS Search-Space Compatibility

The NAS module has multiple consumers: random search, evolutionary search, targeted optimization, and the experimental RL searcher. They share `ArchitectureGene` and `LLMSearchSpace`.

Stable methods:

| Method | Purpose |
|--------|---------|
| `ArchitectureGene.to_config()` | Emit a HuggingFace-style config dictionary. |
| `ArchitectureGene.from_config()` | Rebuild a gene after controller or RL edits to a config dictionary. |
| `LLMSearchSpace.sample()` | Standard random sampler. |
| `LLMSearchSpace.sample_random()` | Backward-compatible alias used by RL code. |
| `LLMSearchSpace.validate_gene()` | Check that a gene remains inside the configured discrete search space. |
| `LLMSearchSpace.default_config` | Discrete choices consumed by the RL encoder/action path. |

CLI support:

```bash
python -m vitriol.cli.main nas --algorithm random --iterations 20
python -m vitriol.cli.main nas --algorithm evolutionary --generations 10 --population 20
python -m vitriol.cli.main nas --algorithm targeted --target-vram 24
python -m vitriol.cli.main nas --algorithm rl --episodes 50
```

Recommended checks:

```bash
python -m pytest tests/test_nas_rl_compat.py tests/test_cli_nas_rl.py -q
python -m mypy src/vitriol/nas/search_space.py --ignore-missing-imports
```

## Release Gate

Before publishing changes around these areas, run the focused checks below. For the complete local release gate, see `docs/release-validation.md`.

```bash
python -m compileall -q src tests
python -m ruff check src/vitriol/bench/ppl_evaluator.py src/vitriol/nas/search_space.py src/vitriol/nas/controller.py src/vitriol/cli/commands/nas.py tests/test_ppl_evaluator.py tests/test_nas_rl_compat.py tests/test_cli_nas_rl.py
python -m pytest tests/test_ppl_evaluator.py tests/test_nas_rl_compat.py tests/test_cli_nas_rl.py -q
python -m pytest -q
python -m pip wheel . --no-deps -w /tmp/vitriol-wheel-check
```
