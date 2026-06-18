# Vitriol Check — Composite Action

Run the Vitriol **Structure-First** golden path in any GitHub workflow.

## Usage (same repository)

```yaml
- uses: ./.github/actions/vitriol-check
  with:
    model_id: Qwen/Qwen2.5-0.5B
    output_dir: vitriol-report
    fast: "true"
    offline: "true"
```

## Usage (external model repository)

Pin to a release tag when available:

```yaml
- uses: isLinXu/Vitriol/.github/actions/vitriol-check@v0.3.1
  with:
    model_id: org/my-model
    output_dir: vitriol-report
```

## Inputs

| Input | Default | Description |
|-------|---------|-------------|
| `model_id` | (required) | HuggingFace ID or local config path |
| `output_dir` | `vitriol-report` | Report output directory |
| `strategy` | `compact` | Weight generation strategy |
| `fast` | `true` | Skip inference + weight hash |
| `trust_remote_code` | `false` | Opt-in remote code execution |
| `offline` | `true` | Disable network downloads |
| `vitriol_ref` | `` | Git ref for `pip install git+...` |
| `python_version` | `3.11` | Python version |

## Outputs

| Output | Description |
|--------|-------------|
| `report_dir` | Path to report bundle |
| `success` | `true` when check passed |

## Artifacts

Upload the report in a follow-up step:

```yaml
- uses: actions/upload-artifact@v4
  with:
    name: vitriol-report
    path: vitriol-report/
```
