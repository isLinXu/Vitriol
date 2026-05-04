#!/usr/bin/env bash
set -euo pipefail

# This script installs Vitriol's dev dependencies locally/in CI, preferring the CPU-only PyTorch
# to avoid accidentally pulling large CUDA dependencies on Linux.

python -m pip install -U pip

if [[ "$(uname -s)" == "Linux" ]]; then
  PIP_INDEX_URL=https://download.pytorch.org/whl/cpu \
  PIP_EXTRA_INDEX_URL=https://pypi.org/simple \
  python -m pip install -e ".[dev]"
else
  python -m pip install -e ".[dev]"
fi

echo "✅ Installation complete. You can run: pytest -v"
