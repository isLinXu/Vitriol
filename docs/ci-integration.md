# Vitriol CI 集成指南

本文说明如何在 CI/CD 流水线中嵌入 Vitriol 的 **Structure-First** 检查。

## 推荐命令

```bash
vitriol --offline --no-trust-remote-code check <MODEL_ID_OR_LOCAL_PATH> \
  -o vitriol-report \
  --fast \
  --strategy compact
```

| 选项 | CI 中的作用 |
|------|-------------|
| `--offline` | 禁止 HuggingFace 网络访问（使用缓存或本地 config） |
| `--no-trust-remote-code` | 安全默认：不执行远程 Python 代码 |
| `--fast` | 跳过 inference 与 weight hash，缩短 CI 时间 |
| `--strategy compact` | 体积适中、加载验证稳定 |

## 门禁脚本

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?model id or local path}"
OUT="${2:-vitriol-report}"

vitriol --offline check "$MODEL" -o "$OUT" --fast

python - <<'PY'
import json, sys
from pathlib import Path

report_path = Path("vitriol-report/check-report.json")
if not report_path.exists():
    sys.exit("missing check-report.json")

payload = json.loads(report_path.read_text(encoding="utf-8"))
if not payload.get("success"):
    sys.exit(f"vitriol check failed: {payload}")
print("vitriol check passed")
PY
```

## GitHub Actions

### Composite Action（推荐）

可复用 Action 位于 `.github/actions/vitriol-check/`：

```yaml
- uses: isLinXu/Vitriol/.github/actions/vitriol-check@v0.3.1
  with:
    model_id: org/my-model
    output_dir: vitriol-report
    fast: "true"
    offline: "true"
```

本仓库内开发时使用 `./.github/actions/vitriol-check`。完整 input/output 说明见 [action README](../.github/actions/vitriol-check/README.md)。

### 示例 Workflow

见 `.github/workflows/vitriol-check-example.yml`。触发方式：

- `workflow_dispatch`：手动输入 `model_id`
- 可在 `pull_request` 中根据 changed files 调用

## Artifacts

建议上传：

- `vitriol-report/index.html` — 人工审查
- `vitriol-report/check-report.json` — 机器判据
- `vitriol-report/weights/vitriol-manifest.json` — 下游 manifest 消费

## 与 release-validation 的关系

发布 Vitriol 本身时，仍应运行 `docs/release-validation.md` 中的完整门禁。  
消费 Vitriol 的模型仓库 CI，只需 `vitriol check` + JSON 判据即可。
