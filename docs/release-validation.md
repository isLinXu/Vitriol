# Release Validation Checklist

This checklist is the final local gate before committing or publishing Vitriol changes.

## Scope

Run the full gate when a change touches:

- CLI command registration or command options
- NAS searchers, `ArchitectureGene`, or `LLMSearchSpace`
- KV benchmark presets, PPL evaluation, or inference wrappers
- packaging metadata, README files, or public documentation

## Commands

First confirm that Python imports the checkout being validated. This matters when an older editable install is still on `sys.path`.

```bash
PYTHONPATH="$PWD/src" python - <<'PY'
import vitriol
print(vitriol.__file__)
PY
```

```bash
PYTHONPATH="$PWD/src" python -m compileall -q src tests

PYTHONPATH="$PWD/src" python -m ruff check \
  src/vitriol/core/validator.py \
  src/vitriol/bench/ppl_evaluator.py \
  src/vitriol/nas/search_space.py \
  src/vitriol/nas/controller.py \
  src/vitriol/cli/commands/nas.py \
  tests/test_family_matrix.py \
  tests/test_ppl_evaluator.py \
  tests/test_nas_rl_compat.py \
  tests/test_cli_nas_rl.py \
  tests/test_trace_cli_outputs_token_fields.py

PYTHONPATH="$PWD/src" python -m pytest \
  tests/test_family_matrix.py \
  tests/test_ppl_evaluator.py \
  tests/test_nas_rl_compat.py \
  tests/test_cli_nas_rl.py \
  tests/test_trace_cli_outputs_token_fields.py \
  -q

PYTHONPATH="$PWD/src" python -m pytest -q
python -m pip wheel . --no-deps -w /tmp/vitriol-wheel-check
```

If the configured package index is unavailable, rerun the wheel check with the current environment:

```bash
python -m pip wheel . --no-deps --no-build-isolation -w /tmp/vitriol-wheel-check
```

## Expected Results

- Compile step exits with code `0`.
- Focused `ruff` step exits with code `0`.
- Focused NAS/PPL tests pass.
- Full pytest suite passes, with only expected skips/warnings.
- Wheel builds as `vitriol-<version>-py3-none-any.whl`.

## Known Environment Warnings

The shared development environment may emit warnings unrelated to this project change, including:

- `pytest_asyncio` default fixture loop scope deprecation
- `torchao` deprecation messages for older import paths
- invalid local distribution warnings such as `~ransformers`

Treat new test failures, import failures, wheel build failures, or project-code lint errors as blockers.

## Git Hygiene

Before committing:

```bash
git status --short
git diff -- README.md README_CN.md CHANGELOG.md docs src tests
```

Do not include local artifacts such as `.DS_Store`, caches, generated output directories, or wheel/build products.
