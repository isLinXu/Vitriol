# Contributing to Vitriol

Thank you for your interest in contributing to **Vitriol**! This document provides guidelines and instructions for contributing to the project.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)

---

## Code of Conduct

Please follow the repository's [Code of Conduct](./CODE_OF_CONDUCT.md). Be respectful, constructive, and professional in all interactions.

## Getting Started

### Prerequisites

- Python >= 3.8 (3.11+ recommended)
- Git
- A code editor (VS Code, PyCharm, etc.)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/isLinXu/Vitriol.git
cd Vitriol

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install in development mode (includes dev tools)
pip install -e ".[dev]"

# Run tests to verify your setup
pytest
```

## Development Setup

### Optional Feature Groups

Install additional dependencies for specific features:

```bash
# Visualization support (rich, matplotlib, plotly, etc.)
pip install -e ".[viz]"

# WebUI (Gradio)
pip install -e ".[webui]"

# REST API (FastAPI)
pip install -e ".[api]"

# All optional features at once
pip install -e ".[dev,viz,webui,api]"
```

### Dev Tools

The project uses the following development tools:

| Tool | Purpose | Config |
|------|---------|--------|
| [Ruff](https://docs.astral.sh/ruff/) | Linting & formatting | `ruff check src/` / `ruff format src/` |
| [pytest](https://docs.pytest.org/) | Testing | `pytest` |
| pytest-cov | Coverage | `pytest --cov=vitriol` |
| mypy | Type checking | `mypy src/vitriol/` |
| pre-commit | Git hooks | `pre-commit run --all-files` |

## Project Structure

```
Vitriol/
├── src/vitriol/          # Main source package
│   ├── core/             # Core engine: config parsing, generation, export
│   ├── strategies/       # 12 weight generation strategies
│   ├── arch_viz/         # Architecture analysis & visualization
│   ├── nas/              # Neural Architecture Search (3 algorithms)
│   ├── evolution/        # Evolution tree, comparison, simulator
│   ├── kv/               # KV Cache compression system
│   ├── metrics/          # CIS scoring framework
│   ├── patches/          # Model-specific patches (Qwen, DeepSeek, etc.)
│   ├── adapters/         # Model family adapters (LLaMA, Qwen, DeepSeek)
│   ├── cli/              # CLI commands (17 commands)
│   ├── api/              # FastAPI experimental API
│   ├── webui/            # Gradio web interface
│   ├── viz/              # HTML visualization templates
│   └── bench/            # Benchmarking utilities
├── tests/                # Test suite (21 test files)
├── docs/                 # Documentation + GitHub Pages assets
├── scripts/              # Utility scripts
├── .github/workflows/    # CI pipelines (CI, Hub-Smoke, Pages)
└── pyproject.toml        # Project configuration
```

## How to Contribute

### Ways to Contribute

1. **Report bugs** — Open a detailed issue with reproduction steps
2. **Suggest features** — Discuss ideas via Issues before implementing
3. **Submit code** — Fix bugs, add features, improve docs
4. **Improve documentation** — Fix typos, clarify explanations, add examples
5. **Add model support** — New model adapters or patches
6. **Add strategies** — Novel weight generation/compression strategies
7. **Add analyzers** — New architecture analysis modules

### Areas That Need Help

- :star: **New model adapters** (e.g., Mistral, Gemma, Phi, Yi)
- :star: **More NAS algorithms** (e.g., Bayesian optimization, RL-based)
- :star: **WebUI enhancements** (interactive architecture editor)
- :star: **Test coverage improvements** (especially for strategies/)
- :star: **Documentation translations** (Japanese, Korean, etc.)

## Coding Standards

### Style Guide

We use **Ruff** for both linting and formatting:

```bash
# Check for issues
ruff check src/

# Auto-fix fixable issues
ruff check --fix src/

# Format code
ruff format src/
```

Key rules:
- Maximum line length: 120 characters
- Use type hints on all public function signatures
- Follow PEP 8 naming conventions (`snake_case` for variables/functions, `PascalCase` for classes)
- Docstrings use Google style for complex functions; one-liners are fine for simple ones

### Python Version Compatibility

- Minimum supported: **Python 3.8**
- Primary testing target: **Python 3.11**
- Avoid syntax features newer than 3.8 (walrus operator is OK, pattern matching is not)

### Code Quality Checklist

Before submitting:

- [ ] Code passes `ruff check src/vitriol`
- [ ] Code passes `ruff format --check src/vitriol`
- [ ] All new functions have type hints
- [ ] Public functions and classes have docstrings
- [ ] No hardcoded paths or secrets
- [ ] Tests pass locally with `pytest`

## Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

| Type | Description | Example |
|------|-------------|---------|
| `feat:` | New feature | `feat: add Mistral model adapter` |
| `fix:` | Bug fix | `fix: resolve ultra strategy OOM on large models` |
| `docs:` | Documentation only | `docs: update CLI usage examples` |
| `style:` | Formatting | `style: apply ruff formatting to core/` |
| `refactor:` | Code refactoring | `refactor: simplify config parser logic` |
| `test:` | Adding/updating tests | `test: add coverage for learned strategy` |
| `chore:` | Maintenance | `chore: update dev dependencies` |
| `perf:` | Performance improvement | `perf: cache fingerprint computation` |

### Commit Message Format

```
<type>(<scope>): <subject>

<body> (optional)
```

Examples:
```
feat(strategies): add ternary weight generation strategy

Implements quantization to {-1, 0, +1} for extreme compression.
Max compression ratio: ~32x.
```

```
fix(cli): resolve viz command crash on missing meta-config

Fall back to config.json when meta-config.json is not present.
```

## Pull Request Process

### Before Opening a PR

1. **Check existing issues/PRs** — Avoid duplicate work
2. **Discuss major changes** — Open an issue first for significant features or architectural changes
3. **Branch from `main`** — Create a feature branch:
   ```bash
   git checkout -b feat/your-feature-name
   ```

### PR Requirements

- [ ] Title follows conventional commit format
- [ ] Linked to an issue (if applicable): `Fixes #123` or `Closes #123`
- [ ] All CI checks pass (see `.github/workflows/ci.yml`)
- [ ] Tests added/updated for new functionality
- [ ] Documentation updated for user-facing changes
- [ ] Self-review completed — describe what changes you made and why

### Review Process

1. Maintainers will review within a few days
2. Address review comments by pushing new commits to your branch
3. Once approved, maintainers will merge using squash merge

### PR Template

When creating a pull request, please fill in the template:

```markdown
## Description
Brief description of the change.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How was this tested?

## Screenshots (if applicable)
Add screenshots for UI/WebUI changes.

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-reviewed my own code
- [ ] Commented complex code sections
- [ ] Changes have been tested locally
- [ ] Updated relevant documentation
```

## Reporting Bugs

### Good Bug Reports Include:

1. **Exact Vitriol version**: `vitriol --version`
2. **Python version**: `python --version`
3. **OS**: macOS / Linux / Windows + distribution
4. **Command that triggered the issue**: Full command with arguments
5. **Expected behavior**: What you expected to happen
6. **Actual behavior**: Error message or unexpected output (include full traceback)
7. **Steps to reproduce**: Minimal reproduction if possible

## Security Issues

If you believe you have found a security issue, please do not file a public bug report first.

Instead, follow the private reporting guidance in [SECURITY.md](./SECURITY.md).

### Example Bug Report

```markdown
**Vitriol version**: 0.2.0
**Python**: 3.11.5
**OS**: Ubuntu 22.04

**Command**:
```
vitriol generate meta-llama/Llama-3.1-70B --strategy sparse
```

**Error**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory.
...
```

**Expected**: Weights generated successfully
**Actual**: OOM error during sparse strategy execution
**Notes**: Works fine with `--strategy compact`. GPU has 24GB VRAM.
```

---

## Questions?

If you have questions about contributing, feel free to open a **Discussion** or an issue with the `question` label.

Happy hacking! :rocket:
