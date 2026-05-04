#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

MANIFEST_PATH="${DEMO_TARGETS_FILE:-${REPO_ROOT}/docs/manifests/demo_targets.txt}"
EXPORT_PATH="${DEMO_EXPORT_SCRIPT:-${REPO_ROOT}/output/run_all_demos.generated.sh}"
REPORT_PATH="${DEMO_REPORT_PATH:-${REPO_ROOT}/output/demo_precheck_report.md}"
ARCH_PORT_BASE="${DEMO_ARCH_PORT_BASE:-8765}"
WEIGHT_PORT_BASE="${DEMO_WEIGHT_PORT_BASE:-8781}"
LOG_DIR="output/demo_logs"

EXTRA_ARGS=()
DRY_RUN=false
WRITE_REPORT=false
LAUNCH_GROUP=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --report)
      WRITE_REPORT=true
      shift
      ;;
    --report-path)
      WRITE_REPORT=true
      REPORT_PATH="$2"
      shift 2
      ;;
    --port-offset)
      ARCH_PORT_BASE="$((ARCH_PORT_BASE + $2))"
      WEIGHT_PORT_BASE="$((WEIGHT_PORT_BASE + $2))"
      shift 2
      ;;
    --launch-group)
      LAUNCH_GROUP="$2"
      group_root="${REPO_ROOT}/output/demo_groups/${LAUNCH_GROUP}"
      EXPORT_PATH="${group_root}/run_all_demos.generated.sh"
      REPORT_PATH="${group_root}/demo_precheck_report.md"
      LOG_DIR="output/demo_groups/${LAUNCH_GROUP}/logs"
      WRITE_REPORT=true
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

COMMON_ARGS=(
  --models-file "${MANIFEST_PATH}"
  --precheck-config
  --precheck-viz
  --arch-port "${ARCH_PORT_BASE}"
  --weight-port "${WEIGHT_PORT_BASE}"
  --static-arch-viz
  --log-dir "${LOG_DIR}"
  --no-open
)

if [[ "${WRITE_REPORT}" == "true" ]]; then
  COMMON_ARGS+=(--export-markdown-report "${REPORT_PATH}")
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  exec "${REPO_ROOT}/scripts/run_model_demo.sh" "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}" --dry-run
fi

"${REPO_ROOT}/scripts/run_model_demo.sh" \
  "${COMMON_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  --export-script "${EXPORT_PATH}"

exec "${EXPORT_PATH}"
