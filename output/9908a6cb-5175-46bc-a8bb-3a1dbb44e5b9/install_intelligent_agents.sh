#!/bin/bash
# 大朝议 III - Agent 智能增强系统安装脚本

set -e

PROJECT_ROOT="/Users/gatilin/PycharmProjects/dachaoyi3"
BACKEND_DIR="$PROJECT_ROOT/backend"
OUTPUT_DIR="/Users/gatilin/PycharmProjects/Vitriol/output/9908a6cb-5175-46bc-a8bb-3a1dbb44e5b9"

echo "=================================================="
echo "  大朝议 III · Agent 智能增强系统"
echo "  安装脚本"
echo "=================================================="
echo

# ==================== 1. 环境检查 ====================
echo "[1/6] 检查环境..."

if [ ! -d "$PROJECT_ROOT" ]; then
    echo "❌ 错误：项目目录不存在 $PROJECT_ROOT"
    exit 1
fi

if [ ! -f "$BACKEND_DIR/edict_graph.py" ]; then
    echo "❌ 错误：找不到 edict_graph.py"
    exit 1
fi

echo "✅ 环境检查通过"

# ==================== 2. 备份现有文件 ====================
echo
echo "[2/6] 备份现有文件..."

BACKUP_DIR="$PROJECT_ROOT/backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

cp "$BACKEND_DIR/edict_graph.py" "$BACKUP_DIR/" 2>/dev/null || true
cp "$BACKEND_DIR/core_agents.py" "$BACKUP_DIR/" 2>/dev/null || true

echo "✅ 备份完成：$BACKUP_DIR"

# ==================== 3. 复制新模块 ====================
echo
echo "[3/6] 安装智能增强模块..."

cp "$OUTPUT_DIR/agent_memory.py" "$BACKEND_DIR/"
cp "$OUTPUT_DIR/agent_evolution.py" "$BACKEND_DIR/"
cp "$OUTPUT_DIR/agent_intelligence.py" "$BACKEND_DIR/"

echo "✅ 已安装：agent_memory.py, agent_evolution.py, agent_intelligence.py"

# ==================== 4. 创建数据目录 ====================
echo
echo "[4/6] 创建数据目录..."

mkdir -p "$PROJECT_ROOT/data/memories/mid_term/default"
mkdir -p "$PROJECT_ROOT/data/memories/long_term/default/preferences"
mkdir -p "$PROJECT_ROOT/data/memories/long_term/default/facts"
mkdir -p "$PROJECT_ROOT/data/memories/long_term/default/skills"
mkdir -p "$PROJECT_ROOT/data/memories/long_term/default/insights"
mkdir -p "$PROJECT_ROOT/backend/skills/auto_generated"

echo "✅ 数据目录已创建"

# ==================== 5. 安装依赖 ====================
echo
echo "[5/6] 检查 Python 依赖..."

cd "$PROJECT_ROOT"

# 检查是否有虚拟环境
if [ -d "venv" ] || [ -d ".venv" ]; then
    source venv/bin/activate 2>/dev/null || source .venv/bin/activate 2>/dev/null || true
fi

# 检查必要的包
python3 -c "import langchain_core" 2>/dev/null || {
    echo "⚠️  警告：未检测到 langchain_core，请确保已安装"
}

echo "✅ 依赖检查完成"

# ==================== 6. 生成集成示例 ====================
echo
echo "[6/6] 生成集成示例..."

cat > "$BACKEND_DIR/example_intelligent_usage.py" << 'EOF'
"""
大朝议 III - Agent 智能增强使用示例
"""
import asyncio
from backend.agent_intelligence import init_intelligent_agents
from backend.llm_provider import get_llm
from backend.vector_store import VectorStoreManager
from backend.skills.skill_registry import SkillRegistry


async def demo_intelligent_agents():
    """演示智能增强功能"""
    print("=== 大朝议 III · 智能增强演示 ===\n")
    
    # 1. 初始化
    print("[1] 初始化智能组件...")
    llm = get_llm()
    vector_store = VectorStoreManager()
    skill_registry = SkillRegistry()
    
    agents = init_intelligent_agents(llm, vector_store, skill_registry)
    memory = agents["memory_manager"]
    zhongshu = agents["zhongshu"]
    
    # 2. 第一轮对话
    print("\n[2] 第一轮对话：建立初始上下文")
    memory.new_session("demo_session_001")
    
    decree_1 = "我的项目在 /home/user/myproject，使用 FastAPI 框架"
    print(f"   皇帝：{decree_1}")
    
    state = {"decree": decree_1, "logs": [], "thoughts": []}
    result_1 = await zhongshu.draft(decree_1, state)
    
    memory.add_turn(decree_1, str(result_1))
    print(f"   中书省：{result_1.get('reasoning', '')}\n")
    
    # 保存到长期记忆
    memory.long_term.remember(
        category=memory.long_term.MemoryCategory.FACT,
        title="项目路径和框架",
        content=f"用户项目在 /home/user/myproject，使用 FastAPI"
    )
    
    # 3. 第二轮对话（测试记忆召回）
    print("[3] 第二轮对话：测试记忆召回")
    decree_2 = "帮我优化项目的性能"
    print(f"   皇帝：{decree_2}")
    
    state_2 = {"decree": decree_2, "logs": [], "thoughts": []}
    result_2 = await zhongshu.draft(decree_2, state_2)
    
    memory.add_turn(decree_2, str(result_2))
    print(f"   中书省：{result_2.get('reasoning', '')}")
    
    if state_2.get("thoughts"):
        for thought in state_2["thoughts"]:
            if "记忆召回" in thought.get("action", ""):
                print(f"   💡 记忆生效：{thought.get('thought', '')}\n")
                break
    
    # 4. 结束会话
    print("[4] 会话结束，触发反思...")
    await memory.end_session({
        "execution_result": "优化建议已生成",
        "error_msg": None,
        "tokens_used": {"zhongshu": 150},
        "duration": 8.5
    })
    
    # 5. 查看进化报告
    print("\n[5] 进化报告")
    report = agents["evolution_manager"].get_evolution_report()
    print(f"   - Prompt 版本数：{len(report.get('prompt_versions', {}))}")
    print(f"   - 自动生成技能：{report.get('generated_skills', 0)}")
    print(f"   - 策略类型数：{report.get('strategy_types', 0)}")
    
    print("\n=== 演示完成 ===")


if __name__ == "__main__":
    asyncio.run(demo_intelligent_agents())
EOF

echo "✅ 示例文件已生成：$BACKEND_DIR/example_intelligent_usage.py"

# ==================== 完成 ====================
echo
echo "=================================================="
echo "  ✅ 安装完成！"
echo "=================================================="
echo
echo "📦 已安装的模块："
echo "   - backend/agent_memory.py        (三层记忆系统)"
echo "   - backend/agent_evolution.py     (自进化引擎)"
echo "   - backend/agent_intelligence.py  (集成方案)"
echo "   - backend/example_intelligent_usage.py  (使用示例)"
echo
echo "📂 数据目录："
echo "   - data/memories/                 (记忆存储)"
echo "   - backend/skills/auto_generated/ (自动生成技能)"
echo
echo "🚀 下一步："
echo "   1. 查看集成文档：$OUTPUT_DIR/AGENT_INTELLIGENCE_INTEGRATION_GUIDE.md"
echo "   2. 运行演示：python backend/example_intelligent_usage.py"
echo "   3. 手动集成到 main.py 和 edict_graph.py"
echo
echo "💾 备份位置：$BACKUP_DIR"
echo

exit 0
