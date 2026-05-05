#!/usr/bin/env bash
# ============================================================================
# 大朝议 III · P0 优化任务自动化脚本
#   自动完成深度分析报告中第 1 周的 P0 任务：
#   T1. 接入 llm_provider 到 core_agents.py / dept_agents.py
#   T2. 接入 vector_store 到 knowledge.py
#   T3. 切换到 persistence_v2
#   T4. 修复 main.py 重复 /health
#   T5. 初始化 git 仓库（可选）
#
# 使用：bash apply_p0_fixes.sh        # 预览
#       bash apply_p0_fixes.sh apply  # 真正执行
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
echo "==> Mode:    $MODE"
cd "$PROJECT"

# ============================================================================
# T1. 接入 llm_provider 到 core_agents.py
# ============================================================================
echo ""
echo "== [T1] 接入 llm_provider 到 core_agents.py =="

if [ "$MODE" = "apply" ]; then
  # 备份原文件
  cp backend/core_agents.py backend/core_agents.py.backup
  
  # 使用 sed 替换（跨平台兼容方式）
  sed -i.bak '
    s/from langchain_openai import ChatOpenAI/from backend.llm_provider import get_llm/
    s/llm = ChatOpenAI(/llm = get_llm(/
    /^llm = get_llm(/,/)/ {
      s/model=model_name/model=model_name/
      s/, api_key=api_key, base_url=base_url//
    }
  ' backend/core_agents.py
  
  rm -f backend/core_agents.py.bak
  echo "  [✓] core_agents.py updated"
else
  echo "  [DRY] 将替换 core_agents.py 中的 ChatOpenAI → get_llm"
fi

# ============================================================================
# T1b. 接入 llm_provider 到 dept_agents.py
# ============================================================================
echo ""
echo "== [T1b] 接入 llm_provider 到 dept_agents.py =="

if [ "$MODE" = "apply" ]; then
  if [ -f backend/dept_agents.py ]; then
    cp backend/dept_agents.py backend/dept_agents.py.backup
    
    # 检查是否有 ChatOpenAI 导入
    if grep -q "from langchain_openai import ChatOpenAI" backend/dept_agents.py; then
      sed -i.bak '
        s/from langchain_openai import ChatOpenAI/from backend.llm_provider import get_llm/
        /ChatOpenAI(/,/)/ {
          s/ChatOpenAI(/get_llm(/
          s/, api_key=[^,)]*, base_url=[^,)]*//
        }
      ' backend/dept_agents.py
      rm -f backend/dept_agents.py.bak
      echo "  [✓] dept_agents.py updated"
    else
      echo "  [ℹ] dept_agents.py 中未发现 ChatOpenAI，跳过"
    fi
  fi
else
  echo "  [DRY] 将替换 dept_agents.py 中的 ChatOpenAI → get_llm"
fi

# ============================================================================
# T2. 接入 vector_store 到 knowledge.py
# ============================================================================
echo ""
echo "== [T2] 接入 vector_store 到 knowledge.py =="

if [ "$MODE" = "apply" ]; then
  if [ -f backend/knowledge.py ]; then
    cp backend/knowledge.py backend/knowledge.py.backup
    
    # 在导入部分添加 vector_store
    if ! grep -q "from backend.vector_store import" backend/knowledge.py; then
      # 在第一个 from typing 之后插入
      sed -i.bak '/^from typing import/a\
from backend.vector_store import get_vector_store
' backend/knowledge.py
      rm -f backend/knowledge.py.bak
      echo "  [✓] 添加 vector_store 导入到 knowledge.py"
      echo "  [!] 需要手动修改 KnowledgeBase.search() 方法以使用语义搜索"
      echo "      参考：store = get_vector_store(); results = store.search(query, top_k=limit)"
    else
      echo "  [ℹ] knowledge.py 已导入 vector_store"
    fi
  fi
else
  echo "  [DRY] 将在 knowledge.py 中添加 vector_store 导入"
fi

# ============================================================================
# T3. 切换到 persistence_v2
# ============================================================================
echo ""
echo "== [T3] 切换到 persistence_v2 =="

if [ "$MODE" = "apply" ]; then
  if [ -f backend/main.py ] && [ -f backend/persistence_v2.py ]; then
    cp backend/main.py backend/main.py.backup
    
    sed -i.bak 's/from backend.persistence import/from backend.persistence_v2 import/' backend/main.py
    rm -f backend/main.py.bak
    echo "  [✓] main.py 切换到 persistence_v2"
  else
    echo "  [✗] 缺少 persistence_v2.py，跳过"
  fi
else
  echo "  [DRY] 将 main.py 的导入从 persistence 改为 persistence_v2"
fi

# ============================================================================
# T4. 修复 main.py 重复 /health 端点
# ============================================================================
echo ""
echo "== [T4] 修复 main.py 重复 /health =="

if [ "$MODE" = "apply" ]; then
  if [ -f backend/main.py ]; then
    # 查找是否有重复的 @app.get("/health")
    HEALTH_COUNT=$(grep -c '@app.get("/health")' backend/main.py || echo 0)
    
    if [ "$HEALTH_COUNT" -gt 1 ]; then
      echo "  [!] 检测到 $HEALTH_COUNT 个 /health 路由"
      echo "  [!] 请手动检查并删除早期版本（通常在第 500 行附近）"
      echo "  [!] 保留完整版（通常在第 1065 行附近，包含 Redis/依赖检查）"
    else
      echo "  [✓] /health 路由无重复"
    fi
  fi
else
  echo "  [DRY] 将检查 /health 路由重复情况"
fi

# ============================================================================
# T5. 初始化 git 仓库（可选）
# ============================================================================
echo ""
echo "== [T5] 初始化 git 仓库（可选）=="

if [ ! -d .git ]; then
  if [ "$MODE" = "apply" ]; then
    read -p "是否初始化 git 仓库？(y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      git init
      git add .gitignore
      git commit -m "chore: add .gitignore" --allow-empty
      echo "  [✓] Git 仓库已初始化"
    else
      echo "  [ℹ] 跳过 git 初始化"
    fi
  else
    echo "  [DRY] 将提示初始化 git 仓库"
  fi
else
  echo "  [✓] Git 仓库已存在"
fi

# ============================================================================
# 总结
# ============================================================================
echo ""
echo "== 完成 =="
if [ "$MODE" = "apply" ]; then
  echo ""
  echo "✓ P0 修复已应用。备份文件保存为 *.backup"
  echo ""
  echo "下一步验证："
  echo "  1. cd backend && python -c 'from core_agents import llm; print(llm)'"
  echo "  2. pytest tests/backend/test_edict_graph.py -v"
  echo "  3. curl http://localhost:8000/health"
  echo ""
  echo "手动任务："
  echo "  - knowledge.py 的 search() 方法需手动集成语义搜索"
  echo "  - 检查 main.py 第 500/1065 行附近的 /health 路由并删除重复"
else
  echo ""
  echo "ℹ 以上为预览。执行：bash $0 apply 以真正操作。"
fi
