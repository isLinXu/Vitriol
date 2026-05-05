"""
大朝议 III · Agent 智能增强 - 集成方案
File: backend/agent_intelligence.py

将记忆、反思、学习、进化能力集成到现有 Agent 系统
"""
from typing import Dict, Any, Optional
from backend.agent_memory import (
    MemoryManager,
    ErrorKnowledgeBase,
    ToolUsagePatternLearner
)
from backend.agent_evolution import EvolutionManager
from backend.core_agents import ZhongshuAgent, MenshuAgent, ShangshuAgent
from backend.edict_graph import add_thought


# ==================== 1. 增强版核心 Agent ====================

class IntelligentZhongshuAgent(ZhongshuAgent):
    """
    增强版中书省 Agent
    集成记忆召回、Prompt 自优化
    """
    def __init__(self, memory_manager: MemoryManager, evolution_manager: EvolutionManager):
        super().__init__()
        self.memory = memory_manager
        self.evolution = evolution_manager
        self.agent_id = "zhongshu"
        
        # 注册初始 Prompt
        if not self.evolution.prompt_optimizer.get_latest_prompt(self.agent_id):
            self.evolution.prompt_optimizer.register_prompt(
                self.agent_id,
                self.prompt.template
            )
    
    async def draft(self, decree: str, state: Optional[Dict] = None) -> Dict[str, Any]:
        """
        增强版拟旨：集成记忆召回
        """
        import time
        start_time = time.time()
        
        # 1. 召回相关上下文
        context = self.memory.recall_context(decree)
        
        if context and context != "（无相关记忆）":
            # 将上下文注入到 decree
            enhanced_decree = f"""【相关记忆】
{context}

【当前任务】
{decree}
"""
            if state:
                add_thought(
                    state, self.agent_id, "中书省",
                    f"已调阅相关案卷，发现{len(context)}字历史记录",
                    "记忆召回",
                    "准备参考历史经验"
                )
        else:
            enhanced_decree = decree
        
        # 2. 执行原拟旨逻辑
        try:
            result = await super().draft(enhanced_decree)
            duration = time.time() - start_time
            
            # 3. 记录成功
            self.evolution.prompt_optimizer.record_performance(
                self.agent_id,
                success=True,
                duration=duration
            )
            
            # 4. 检查是否需要进化
            await self.evolution.check_and_evolve(self.agent_id)
            
            return result
        
        except Exception as e:
            duration = time.time() - start_time
            
            # 记录失败
            self.evolution.prompt_optimizer.record_performance(
                self.agent_id,
                success=False,
                duration=duration,
                error_msg=str(e),
                context={"decree": decree[:100]}
            )
            
            # 立即尝试优化
            if self.evolution.prompt_optimizer.should_optimize(self.agent_id, min_samples=5):
                print(f"[Intelligence] {self.agent_id} 连续失败，触发紧急优化...")
                optimized = await self.evolution.prompt_optimizer.optimize_prompt(
                    self.agent_id,
                    self.chain.last  # LLM instance
                )
                if optimized:
                    # 重新构建 Chain
                    from langchain_core.prompts import ChatPromptTemplate
                    self.prompt = ChatPromptTemplate.from_template(optimized)
                    self.chain = self.prompt | self.chain.last | self.parser
            
            raise


class IntelligentMenshuAgent(MenshuAgent):
    """
    增强版门下省 Agent
    集成错误知识库查询
    """
    def __init__(self, error_kb: ErrorKnowledgeBase):
        super().__init__()
        self.error_kb = error_kb
        self.agent_id = "menshu"
    
    async def review(self, task_json: Dict[str, Any], state: Optional[Dict] = None) -> Dict[str, Any]:
        """
        增强版审核：检查历史错误模式
        """
        # 1. 执行原审核逻辑
        report = await super().review(task_json)
        
        # 2. 查找类似任务的历史错误
        similar_errors = self.error_kb.find_similar_tasks(task_json)
        
        if similar_errors:
            # 发现历史风险
            report["findings"].append(
                f"⚠️ 历史记录：类似任务曾出现 {len(similar_errors)} 次错误"
            )
            
            # 追加补救建议
            remedies = [e["remedy"] for e in similar_errors if e.get("remedy")]
            if remedies:
                report["recommendation"] += f"\n\n【历史经验】\n" + "\n".join(f"- {r}" for r in remedies[:3])
            
            # 提高风险等级
            if report.get("risk_level", "").lower() == "low":
                report["risk_level"] = "medium"
            
            if state:
                add_thought(
                    state, self.agent_id, "门下省",
                    f"查阅案卷：此类任务历史上曾失手 {len(similar_errors)} 次，建议慎重",
                    "从错误知识库召回",
                    f"发现 {len(similar_errors)} 个相关先例"
                )
        
        return report


class IntelligentShangshuAgent(ShangshuAgent):
    """
    增强版尚书省 Agent
    集成策略优化器
    """
    def __init__(self, evolution_manager: EvolutionManager):
        super().__init__()
        self.evolution = evolution_manager
        self.agent_id = "shangshu"
    
    async def route(self, task_json: Dict[str, Any], state: Optional[Dict] = None) -> Dict[str, Any]:
        """
        增强版路由：参考历史最佳策略
        """
        # 1. 查询历史最佳部门
        task_type = task_json.get("action", "general")
        best_dept = self.evolution.suggest_best_strategy(f"routing_{task_type}")
        
        if best_dept and state:
            add_thought(
                state, self.agent_id, "尚书省",
                f"查阅往例：类似任务历史上【{best_dept}】执行成功率最高",
                "从策略优化器召回",
                "准备参考历史决策"
            )
        
        # 2. 执行原路由逻辑
        routing = await super().route(task_json)
        
        # 3. 如果历史最佳部门与当前决策不同，记录为替代方案
        if best_dept and routing.get("department") != best_dept:
            routing["alternative"] = best_dept
            routing["explanation"] += f"\n（历史最优：{best_dept}）"
        
        return routing


# ==================== 2. 增强版执行节点 ====================

async def intelligent_execution_node(state: Dict[str, Any], 
                                    tool_learner: ToolUsagePatternLearner,
                                    evolution_manager: EvolutionManager):
    """
    增强版执行节点
    集成工具推荐、学习记录
    """
    from backend.edict_graph import execution_node, normalize_department
    from backend.dept_agents import DEPT_AGENTS
    import time
    
    task = state.get("task_json", {})
    
    # 1. 查询工具推荐
    task_desc = task.get("action", "")
    suggested_tools = tool_learner.suggest_tools(task_desc)
    
    if suggested_tools:
        add_thought(
            state, "execution", "六部",
            f"查阅成功案卷：类似任务常用工具 {', '.join(suggested_tools[:3])}",
            "从工具学习器召回",
            "准备参考历史经验"
        )
    
    # 2. 执行任务
    start_time = time.time()
    dept_id = normalize_department(state["routing_info"].get("department", "hubu"))
    
    try:
        result = await DEPT_AGENTS[dept_id].execute(task)
        duration = time.time() - start_time
        
        # 3. 记录成功执行
        tools_used = result.get("tools_used", [])
        tool_learner.record_execution(
            task_desc=task_desc,
            tools_used=tools_used,
            success=True,
            duration=duration
        )
        
        # 4. 跟踪工具序列（用于技能生成）
        if tools_used:
            evolution_manager.track_tool_sequence(tools_used)
        
        # 5. 记录部门决策成功
        evolution_manager.record_decision(
            decision_type=f"routing_{task.get('action', 'general')}",
            decision=dept_id,
            success=True,
            context={"task": task}
        )
        
        state["execution_result"] = result.get("message", "")
        
    except Exception as e:
        duration = time.time() - start_time
        
        # 记录失败
        tool_learner.record_execution(
            task_desc=task_desc,
            tools_used=[],
            success=False,
            duration=duration
        )
        
        evolution_manager.record_decision(
            decision_type=f"routing_{task.get('action', 'general')}",
            decision=dept_id,
            success=False,
            context={"task": task, "error": str(e)}
        )
        
        state["error_msg"] = str(e)
    
    return state


# ==================== 3. 增强版监察节点 ====================

async def intelligent_monitor_node(state: Dict[str, Any],
                                   error_kb: ErrorKnowledgeBase,
                                   memory_manager: MemoryManager):
    """
    增强版监察节点
    记录错误模式、触发反思
    """
    from backend.edict_graph import monitor_node
    from backend.core_agents import llm
    
    # 1. 执行原监察逻辑
    state = await monitor_node(state)
    
    # 2. 如果有错误，记录到错误知识库
    error_msg = state.get("error_msg")
    if error_msg:
        # 尝试获取补救方案
        remedy = state.get("impeachment_report", {}).get("remedy", "")
        
        error_kb.add_pattern(
            error_msg=error_msg,
            remedy=remedy,
            context={
                "task": state.get("task_json", {}),
                "department": state.get("routing_info", {}).get("department")
            }
        )
    
    # 3. 会话结束，触发反思
    if memory_manager.llm:
        final_state = {
            "execution_result": state.get("execution_result"),
            "error_msg": error_msg,
            "tokens_used": state.get("tokens_used", {}),
            "duration": state.get("duration", 0)
        }
        
        await memory_manager.end_session(final_state)
    
    return state


# ==================== 4. 全局初始化 ====================

def init_intelligent_agents(llm, vector_store=None, skill_registry=None):
    """
    初始化所有智能增强组件
    """
    # 1. 初始化记忆管理器
    memory_manager = MemoryManager(
        user_id="emperor_001",
        vector_store=vector_store,
        llm=llm
    )
    
    # 2. 初始化进化管理器
    evolution_manager = EvolutionManager(
        llm=llm,
        skill_registry=skill_registry
    )
    
    # 3. 创建增强版 Agent
    zhongshu = IntelligentZhongshuAgent(memory_manager, evolution_manager)
    menshu = IntelligentMenshuAgent(memory_manager.error_kb)
    shangshu = IntelligentShangshuAgent(evolution_manager)
    
    return {
        "memory_manager": memory_manager,
        "evolution_manager": evolution_manager,
        "zhongshu": zhongshu,
        "menshu": menshu,
        "shangshu": shangshu
    }


# ==================== 5. 使用示例 ====================

async def example_integration():
    """集成示例"""
    from backend.llm_provider import get_llm
    from backend.vector_store import VectorStoreManager
    from backend.skills.skill_registry import SkillRegistry
    
    # 初始化依赖
    llm = get_llm()
    vector_store = VectorStoreManager()
    skill_registry = SkillRegistry()
    
    # 初始化智能 Agent
    intelligent_agents = init_intelligent_agents(llm, vector_store, skill_registry)
    
    memory_manager = intelligent_agents["memory_manager"]
    zhongshu = intelligent_agents["zhongshu"]
    
    # 模拟一轮对话
    memory_manager.new_session("task_test_001")
    
    decree = "帮我分析这个项目的架构"
    state = {"decree": decree, "logs": [], "thoughts": []}
    
    # 调用增强版拟旨
    task_json = await zhongshu.draft(decree, state)
    print("拟旨结果:", task_json)
    
    # 添加对话到记忆
    memory_manager.add_turn(
        user_msg=decree,
        assistant_msg=str(task_json)
    )
    
    # 会话结束
    final_state = {
        "execution_result": "分析完成",
        "error_msg": None,
        "tokens_used": {"zhongshu": 100},
        "duration": 5.2
    }
    await memory_manager.end_session(final_state)


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_integration())
