# Vitriol Release Checklist

Use this checklist before tagging or publishing a Vitriol release.

## Environment

- Confirm the supported Python range in `pyproject.toml` matches the CI matrix.
- Start from a clean working tree except for the intended release changes.
- Remove local runtime/cache artifacts such as `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, and `.mypy_cache/`.

## Quality Gates

- Run the governance lint gate:
  `ruff check src/vitriol/api/server.py src/vitriol/cli/main.py src/vitriol/config src/vitriol/security src/vitriol/utils/hf_loading.py tests/test_api_server.py`
- Run the governance type gate:
  `mypy src/vitriol/api/server.py src/vitriol/cli/main.py src/vitriol/security/context.py --ignore-missing-imports --follow-imports=skip`
- Run the full test suite:
  `pytest tests/ --ignore=tests/integration --tb=short`
- Run the CLI smoke checks:
  `python -m vitriol --version`
  `python -m vitriol --help`
  `python -m vitriol generate --help`

## Security And Offline Semantics

- Verify `--no-trust-remote-code` is propagated through CLI, API, and generation paths.
- Verify `--offline` implies `allow_network=False` and `local_files_only=True`.
- Confirm new HuggingFace loading paths go through `vitriol.utils.hf_loading` or explicitly document why not.

## Packaging

- Build the distribution:
  `python -m build --no-isolation`
- Inspect the wheel contents for expected package data, especially files under `vitriol.viz`.
- Install the built wheel in a fresh environment and rerun CLI smoke checks.

## Documentation

- Update README command lists when adding, renaming, or removing CLI commands.
- Label experimental features clearly in README and API/WebUI docs.
- Move release screenshots and audit artifacts under a documented `docs/` location or external release artifacts.
