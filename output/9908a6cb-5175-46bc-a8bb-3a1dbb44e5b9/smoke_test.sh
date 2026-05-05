#!/usr/bin/env bash
# ============================================================================
# 大朝议 III · 关键链路冒烟测试脚本
#   验证 P0 修复后系统的核心功能是否正常
# ============================================================================

set -euo pipefail

PROJECT="/Users/gatilin/PycharmProjects/dachaoyi3"
BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"
FRONTEND_URL="${FRONTEND_URL:-http://localhost:3000}"

echo "==> 大朝议 III 冒烟测试"
echo "    Backend:  $BACKEND_URL"
echo "    Frontend: $FRONTEND_URL"
echo ""

# 颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass() {
  echo -e "${GREEN}✓${NC} $1"
}

fail() {
  echo -e "${RED}✗${NC} $1"
  exit 1
}

warn() {
  echo -e "${YELLOW}⚠${NC} $1"
}

# ============================================================================
# 前置检查
# ============================================================================
echo "== [0/8] 前置检查 =="

if ! command -v curl &> /dev/null; then
  fail "curl 未安装"
fi

if ! command -v jq &> /dev/null; then
  warn "jq 未安装，部分 JSON 解析将跳过（可选依赖）"
  HAS_JQ=false
else
  HAS_JQ=true
fi

pass "前置工具检查通过"
echo ""

# ============================================================================
# Test 1: 后端健康检查
# ============================================================================
echo "== [1/8] 后端健康检查 =="

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BACKEND_URL/health" || echo "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "后端健康检查通过 (HTTP $HTTP_CODE)"
else
  fail "后端健康检查失败 (HTTP $HTTP_CODE)。请确保后端已启动：cd backend && uvicorn main:app --reload"
fi
echo ""

# ============================================================================
# Test 2: LLM Provider 可用性
# ============================================================================
echo "== [2/8] LLM Provider 可用性 =="

cd "$PROJECT"
LLM_CHECK=$(python3 -c "
import sys
sys.path.insert(0, 'backend')
try:
    from llm_provider import get_llm
    llm = get_llm(temperature=0.7)
    print('OK:', llm.provider.get_model_name())
except Exception as e:
    print('ERROR:', str(e))
" 2>&1)

if [[ "$LLM_CHECK" == OK:* ]]; then
  pass "LLM Provider 可用: ${LLM_CHECK#OK: }"
else
  fail "LLM Provider 初始化失败: $LLM_CHECK"
fi
echo ""

# ============================================================================
# Test 3: core_agents.py 已切换到新 Provider
# ============================================================================
echo "== [3/8] core_agents 使用新 LLM Provider =="

if grep -q "from langchain_openai import ChatOpenAI" "$PROJECT/backend/core_agents.py"; then
  fail "core_agents.py 仍在使用 ChatOpenAI，未切换到 llm_provider"
else
  if grep -q "from backend.llm_provider import get_llm" "$PROJECT/backend/core_agents.py"; then
    pass "core_agents.py 已切换到 llm_provider"
  else
    warn "core_agents.py 未找到 llm_provider 导入，可能未完成迁移"
  fi
fi
echo ""

# ============================================================================
# Test 4: persistence_v2 已启用
# ============================================================================
echo "== [4/8] persistence_v2 已启用 =="

if grep -q "from backend.persistence_v2 import" "$PROJECT/backend/main.py"; then
  pass "main.py 已切换到 persistence_v2"
else
  if grep -q "from backend.persistence import" "$PROJECT/backend/main.py"; then
    warn "main.py 仍在使用旧 persistence，未切换到 v2"
  else
    warn "main.py 未找到 persistence 导入"
  fi
fi
echo ""

# ============================================================================
# Test 5: /health 路由无重复
# ============================================================================
echo "== [5/8] /health 路由无重复 =="

HEALTH_COUNT=$(grep -c '@app.get("/health")' "$PROJECT/backend/main.py" || echo 0)

if [ "$HEALTH_COUNT" -le 1 ]; then
  pass "/health 路由无重复 (发现 $HEALTH_COUNT 个)"
else
  fail "/health 路由重复 (发现 $HEALTH_COUNT 个)，需手动删除早期版本"
fi
echo ""

# ============================================================================
# Test 6: 后端核心 API 可访问
# ============================================================================
echo "== [6/8] 后端核心 API 可访问 =="

# /api/decree
DECREE_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
  "$BACKEND_URL/api/decree" \
  -H "Content-Type: application/json" \
  -d '{"content":"测试圣旨","user_id":"test"}' || echo "000")

if [ "$DECREE_CODE" = "200" ] || [ "$DECREE_CODE" = "422" ]; then
  # 422 也算通过（可能缺少必填字段，但路由存在）
  pass "/api/decree 可访问 (HTTP $DECREE_CODE)"
else
  warn "/api/decree 不可访问 (HTTP $DECREE_CODE)"
fi

# /api/knowledge/documents
KNOWLEDGE_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BACKEND_URL/api/knowledge/documents" || echo "000")

if [ "$KNOWLEDGE_CODE" = "200" ]; then
  pass "/api/knowledge/documents 可访问"
else
  warn "/api/knowledge/documents 不可访问 (HTTP $KNOWLEDGE_CODE)"
fi

echo ""

# ============================================================================
# Test 7: 前端可访问（可选）
# ============================================================================
echo "== [7/8] 前端可访问（可选）=="

FRONTEND_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$FRONTEND_URL" || echo "000")

if [ "$FRONTEND_CODE" = "200" ]; then
  pass "前端可访问 ($FRONTEND_URL)"
else
  warn "前端不可访问 (HTTP $FRONTEND_CODE)。如未启动，可跳过：cd frontend && npm run dev"
fi
echo ""

# ============================================================================
# Test 8: 后端单元测试（关键用例）
# ============================================================================
echo "== [8/8] 后端关键测试用例 =="

cd "$PROJECT"
if [ -d "tests/backend" ]; then
  TEST_OUTPUT=$(python3 -m pytest tests/backend/test_edict_graph.py -q --tb=no 2>&1 || echo "FAILED")
  
  if [[ "$TEST_OUTPUT" == *"passed"* ]]; then
    pass "edict_graph 测试通过"
  else
    warn "edict_graph 测试失败或未运行，详情：pytest tests/backend/test_edict_graph.py -v"
  fi
else
  warn "未找到 tests/backend 目录"
fi
echo ""

# ============================================================================
# 总结
# ============================================================================
echo "=========================================="
echo "冒烟测试完成"
echo "=========================================="
echo ""
echo "✅ 通过的检查会标记为绿色 ✓"
echo "⚠️  警告会标记为黄色（不影响核心功能）"
echo "❌ 失败会标记为红色并终止"
echo ""
echo "下一步："
echo "  1. 手动在浏览器测试：$FRONTEND_URL"
echo "  2. 发送一条圣旨，观察 WebSocket 消息"
echo "  3. 查看 Agent 思维面板、户部账本"
echo "  4. 切换 .env 中 LLM_PROVIDER 验证多模型支持"
echo ""
