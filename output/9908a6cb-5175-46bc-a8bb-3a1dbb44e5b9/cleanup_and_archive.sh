#!/usr/bin/env bash
# ============================================================================
# 大朝议 III · 仓库级清理与文档归档脚本（可逆）
#   · 将根目录 11 份历史优化文档归档到 docs/archive/
#   · 将 test_coords_*.js 临时文件归档到 docs/archive/scratch/
#   · 删除提交产物（.coverage / htmlcov / .DS_Store / .pytest_cache）
#   · 生成 CHANGELOG.md / CONTRIBUTING.md 占位
#   · 保留 README.md / Makefile / GITHUB_FIRST_PUSH_CHECKLIST.md
#
# 使用：bash cleanup_and_archive.sh        # 预览
#       bash cleanup_and_archive.sh apply  # 真正执行
# ============================================================================

set -euo pipefail
PROJECT="/Users/gatilin/PycharmProjects/dachaoyi3"
MODE="${1:-dryrun}"

run() {
  if [ "$MODE" = "apply" ]; then
    echo "  [RUN] $*"
    eval "$@"
  else
    echo "  [DRY] $*"
  fi
}

echo "==> Project: $PROJECT"
echo "==> Mode:    $MODE   (use 'apply' to actually execute)"
cd "$PROJECT"

# 1. 归档历史优化文档
echo ""
echo "== [1/5] 归档历史优化文档到 docs/archive/ =="
run "mkdir -p docs/archive"
for f in \
  OPTIMIZATION_ANALYSIS.md \
  OPTIMIZATION_ANALYSIS_V2.md \
  OPTIMIZATION_ANALYSIS_V3.md \
  OPTIMIZATION_ANALYSIS_V4.md \
  OPTIMIZATION_ANALYSIS_V5.md \
  OPTIMIZATION_ANALYSIS_V6.md \
  OPTIMIZATION_ANALYSIS_V7.md \
  OPTIMIZATION_ANALYSIS_V8.md \
  OPTIMIZATION_IMPLEMENTATION_REPORT.md \
  PROJECT_DEEP_ANALYSIS.md \
  QUICK_FIX_GUIDE.md \
  SYSTEM_ASSESSMENT.md \
  AGENT_OPTIMIZATION_PLAN.md
do
  if [ -f "$f" ]; then
    run "mv '$f' 'docs/archive/$f'"
  fi
done

# 2. 归档前端临时坐标脚本
echo ""
echo "== [2/5] 归档 test_coords_*.js 到 docs/archive/scratch/ =="
run "mkdir -p docs/archive/scratch"
for f in test_coords.js test_coords_2.js test_coords_3.js \
         test_coords_4.js test_coords_5.js test_coords_6.js; do
  if [ -f "$f" ]; then
    run "mv '$f' 'docs/archive/scratch/$f'"
  fi
done

# 3. 清理可重现产物（这些都在 .gitignore 里，但物理占用磁盘）
echo ""
echo "== [3/5] 清理可重现产物 =="
run "find . -name '.DS_Store' -not -path './backend/venv/*' -not -path './frontend/node_modules/*' -delete"
run "rm -rf .coverage htmlcov .pytest_cache .benchmarks"
run "rm -rf backend/__pycache__ backend/*/__pycache__"

# 4. 生成精简后的文档索引 + 占位
echo ""
echo "== [4/5] 生成 docs/archive/INDEX.md + CHANGELOG/CONTRIBUTING 占位 =="
if [ "$MODE" = "apply" ]; then
cat > docs/archive/INDEX.md <<'EOF'
# 历史优化文档归档

本目录存放 2026-03 ~ 2026-05 的迭代过程文档，仅作历史参考。
最新的状态与路线请看：

- 项目入口：`../../README.md`
- 下一阶段路线：`../DACHAOYI3_DEEP_REVIEW.md`（如已移入）或 output 目录

## 时间线
| 日期 | 文档 | 主题 |
|---|---|---|
| 2026-03-21 | OPTIMIZATION_ANALYSIS.md ~ V3 | 初版安全与性能全面审计 |
| 2026-03-26 | OPTIMIZATION_ANALYSIS_V4 ~ V8 | 可观测性、游戏趣味性、Agent 思维面板 |
| 2026-04-30 | GITHUB_FIRST_PUSH_CHECKLIST.md | 首次推 GitHub 清单（仍在根目录保留） |
| 2026-05-04 | PROJECT_DEEP_ANALYSIS.md / IMPLEMENTATION_REPORT | 综合评价 + LLM/向量/持久化抽象落地 |
| 2026-05-04 | AGENT_OPTIMIZATION_PLAN.md | Hermes/OpenClaw 借鉴方案（规划，未全部实现）|
EOF
  echo "  [RUN] wrote docs/archive/INDEX.md"

  if [ ! -f CHANGELOG.md ]; then
cat > CHANGELOG.md <<'EOF'
# Changelog

本项目采用 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 约定。

## [Unreleased]
### Planned
- 将 `llm_provider` 接入 `core_agents.py` / `dept_agents.py`
- 将 `vector_store` 接入 `knowledge.py`
- 切换 `persistence_v2`
- 拆分 `backend/main.py` 为 `backend/api/*`
- 拆分 `frontend/src/hooks/useImperialWS.ts`

## [3.0.0-MVP] - 2026-04-30
### Added
- Agent 思维面板、随机事件、节日事件、声望系统
- LLM Provider 抽象、Chroma 向量库、持久化 V2（已建，主路径待接入）
EOF
    echo "  [RUN] wrote CHANGELOG.md"
  fi

  if [ ! -f CONTRIBUTING.md ]; then
cat > CONTRIBUTING.md <<'EOF'
# 贡献指南

## 目录约定
- `backend/` — FastAPI + LangGraph 后端
- `frontend/` — Next.js 14 前端
- `docs/archive/` — 历史优化文档（只读）
- `tests/` — 后端 pytest；前端 Jest 在 `frontend/src/__tests__`

## 本地开发
见 `README.md`。

## 提交规范
- 遵循 Conventional Commits：`feat: ...` / `fix: ...` / `chore: ...` / `refactor: ...`
- PR 前需：后端 `pytest -q` + 前端 `npm run lint && npm test -- --runInBand`

## 重构红线
- `backend/main.py` 不应再增加新路由——新路由去 `backend/api/<域名>.py`
- 前端单文件组件 > 300 行应拆分
EOF
    echo "  [RUN] wrote CONTRIBUTING.md"
  fi
fi

# 5. 总结
echo ""
echo "== [5/5] 完成 =="
if [ "$MODE" = "apply" ]; then
  echo "✓ 清理完成。请检查 docs/archive/ 后提交 git。"
else
  echo "ℹ 以上为预览。执行：bash $0 apply  以真正操作。"
fi
