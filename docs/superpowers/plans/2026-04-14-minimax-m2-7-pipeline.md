# MiniMax-M2.7 Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a one-click pipeline that regenerates `MiniMaxAI/MiniMax-M2.7`, validates local read/load, and rebuilds architecture visualization artifacts for `output/minimax_m2_7_ultra`.

**Architecture:** Add one Python orchestrator module that owns argument parsing, subprocess execution, and local validation, then expose it through a tiny shell wrapper in `scripts/`. Keep validation explicit for `config/tokenizer/model load`, and keep interactive `viz` serving optional so the default run terminates cleanly.

**Tech Stack:** Python, Bash, Transformers, existing Vitriol CLI commands

---

### Task 1: Lock the pipeline contract with tests

**Files:**
- Test: `tests/test_minimax_pipeline.py`

- [ ] Add a unit test that asserts the default plan contains `generate`, `validate-load`, and the three static `arch-viz` steps.
- [ ] Add a unit test that asserts `--serve-viz` appends the final interactive `viz` step with `--port` and optional `--no-open`.

### Task 2: Implement the Python pipeline orchestrator

**Files:**
- Create: `src/vitriol/tools/minimax_pipeline.py`

- [ ] Add `PipelineOptions` and `PipelineStep` dataclasses to define the pipeline contract.
- [ ] Implement plan builders for `generate`, static `arch-viz`, and optional `viz`.
- [ ] Implement local validation for `AutoConfig`, `AutoTokenizer`, and `AutoModelForCausalLM`, with optional inference and JSON report output.
- [ ] Add a CLI `main()` that executes the plan in order and prints concise progress.

### Task 3: Expose a one-click shell entry

**Files:**
- Create: `scripts/run_minimax_m2_7_ultra_pipeline.sh`

- [ ] Add a shell wrapper that resolves the repo root, exports `PYTHONPATH=src`, and execs the Python orchestrator.

### Task 4: Verify and diagnose

**Files:**
- Test: `tests/test_minimax_pipeline.py`

- [ ] Run the focused pytest file and confirm the new plan contract passes.
- [ ] Run the wrapper with `--help` to verify the one-click entry is discoverable.
- [ ] Run diagnostics on touched Python files and fix any introduced issues.
