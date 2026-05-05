"""
大朝议 III · Agent 自进化系统 - 完整实现代码
File: backend/agent_evolution.py

Prompt 自调优 + 动态技能生成 + 策略优化
"""
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


# ==================== 1. Prompt 优化器 ====================

@dataclass
class PromptVersion:
    """Prompt 版本"""
    version: int
    prompt: str
    success_count: int = 0
    fail_count: int = 0
    avg_duration: float = 0.0
    created_at: float = field(default_factory=time.time)
    examples: List[Dict] = field(default_factory=list)  # 失败案例


class PromptOptimizer:
    """
    Prompt 自调优器
    根据执行成功率自动优化系统提示词
    """
    def __init__(self):
        self.storage_file = Path("data/memories/prompt_versions.json")
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        self.prompt_versions: Dict[str, List[PromptVersion]] = self._load()
    
    def _load(self) -> Dict[str, List[PromptVersion]]:
        if self.storage_file.exists():
            try:
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 转换为 PromptVersion 对象
                    result = {}
                    for agent_id, versions in data.items():
                        result[agent_id] = [
                            PromptVersion(**v) for v in versions
                        ]
                    return result
            except Exception:
                return {}
        return {}
    
    def _save(self):
        # 转换为可序列化的字典
        data = {}
        for agent_id, versions in self.prompt_versions.items():
            data[agent_id] = [
                {
                    "version": v.version,
                    "prompt": v.prompt,
                    "success_count": v.success_count,
                    "fail_count": v.fail_count,
                    "avg_duration": v.avg_duration,
                    "created_at": v.created_at,
                    "examples": v.examples
                }
                for v in versions
            ]
        
        with open(self.storage_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def register_prompt(self, agent_id: str, prompt: str, version: int = 1):
        """注册一个 Prompt 版本"""
        if agent_id not in self.prompt_versions:
            self.prompt_versions[agent_id] = []
        
        self.prompt_versions[agent_id].append(
            PromptVersion(version=version, prompt=prompt)
        )
        self._save()
    
    def get_latest_prompt(self, agent_id: str) -> str:
        """获取最新 Prompt"""
        if agent_id not in self.prompt_versions or not self.prompt_versions[agent_id]:
            return ""
        return self.prompt_versions[agent_id][-1].prompt
    
    def record_performance(self, agent_id: str, success: bool, duration: float, 
                          error_msg: str = "", context: Dict = None):
        """记录性能"""
        if agent_id not in self.prompt_versions or not self.prompt_versions[agent_id]:
            return
        
        current = self.prompt_versions[agent_id][-1]
        
        if success:
            current.success_count += 1
        else:
            current.fail_count += 1
            # 记录失败案例
            if len(current.examples) < 10:  # 最多保留 10 个
                current.examples.append({
                    "error_msg": error_msg,
                    "context": context or {},
                    "timestamp": time.time()
                })
        
        # 更新平均耗时
        total = current.success_count + current.fail_count
        current.avg_duration = (
            current.avg_duration * (total - 1) + duration
        ) / total
        
        self._save()
    
    def should_optimize(self, agent_id: str, min_samples: int = 10) -> bool:
        """
        判断是否需要优化 Prompt
        条件：失败率 > 20% 或 平均耗时 > 10秒
        """
        if agent_id not in self.prompt_versions or not self.prompt_versions[agent_id]:
            return False
        
        current = self.prompt_versions[agent_id][-1]
        total = current.success_count + current.fail_count
        
        if total < min_samples:
            return False
        
        fail_rate = current.fail_count / total
        return fail_rate > 0.2 or current.avg_duration > 10
    
    async def optimize_prompt(self, agent_id: str, llm) -> Optional[str]:
        """使用 LLM 优化 Prompt"""
        if agent_id not in self.prompt_versions or not self.prompt_versions[agent_id]:
            return None
        
        current = self.prompt_versions[agent_id][-1]
        error_samples = current.examples[-5:]  # 最近 5 个失败案例
        
        if not error_samples:
            return None
        
        from langchain_core.messages import SystemMessage, HumanMessage
        
        optimization_prompt = f"""你是 Prompt 工程专家。以下是一个 Agent 的系统提示词及其失败案例：

【当前 Prompt】
{current.prompt}

【失败案例】（最近 5 个）
{json.dumps(error_samples, ensure_ascii=False, indent=2)}

【统计】
- 成功次数: {current.success_count}
- 失败次数: {current.fail_count}
- 失败率: {current.fail_count / (current.success_count + current.fail_count):.2%}
- 平均耗时: {current.avg_duration:.2f}秒

【任务】
分析失败原因，优化 Prompt 以：
1. 减少错误率（目标 < 10%）
2. 提升响应速度
3. 保持输出格式稳定
4. 增强错误处理能力

输出优化后的 Prompt（纯文本，不要包裹代码块）。
"""
        try:
            response = await llm.ainvoke([
                SystemMessage(content="你是 Prompt 工程专家"),
                HumanMessage(content=optimization_prompt)
            ])
            
            optimized_prompt = response.content.strip()
            
            # 注册新版本
            new_version = len(self.prompt_versions[agent_id]) + 1
            self.register_prompt(agent_id, optimized_prompt, new_version)
            
            print(f"[Evolution] {agent_id} Prompt 优化完成，新版本: v{new_version}")
            return optimized_prompt
        
        except Exception as e:
            print(f"[Evolution] Prompt 优化失败: {e}")
            return None
    
    def get_performance_report(self, agent_id: str) -> Dict[str, Any]:
        """获取性能报告"""
        if agent_id not in self.prompt_versions:
            return {}
        
        versions = self.prompt_versions[agent_id]
        report = {
            "agent_id": agent_id,
            "total_versions": len(versions),
            "versions": []
        }
        
        for v in versions:
            total = v.success_count + v.fail_count
            report["versions"].append({
                "version": v.version,
                "success_count": v.success_count,
                "fail_count": v.fail_count,
                "success_rate": v.success_count / total if total > 0 else 0,
                "avg_duration": v.avg_duration,
                "created_at": v.created_at
            })
        
        return report


# ==================== 2. 动态技能生成器 ====================

class SkillGenerator:
    """
    动态技能生成器
    当检测到重复操作模式时，自动生成新技能
    """
    def __init__(self, skill_registry=None):
        self.skill_registry = skill_registry
        self.pattern_tracker: Dict[str, int] = {}
        self.skills_dir = Path("backend/skills/auto_generated")
        self.skills_dir.mkdir(parents=True, exist_ok=True)
    
    def track_pattern(self, action_sequence: List[str]):
        """跟踪操作序列"""
        if len(action_sequence) < 2:
            return  # 单步操作不生成技能
        
        pattern_key = " -> ".join(action_sequence)
        self.pattern_tracker[pattern_key] = self.pattern_tracker.get(pattern_key, 0) + 1
        
        # 出现 3 次以上，生成技能
        if self.pattern_tracker[pattern_key] == 3:
            print(f"[Evolution] 检测到重复模式（3次）：{pattern_key}")
            print(f"[Evolution] 将在下次触发时生成技能")
        elif self.pattern_tracker[pattern_key] > 3:
            # 避免重复生成
            return
    
    async def generate_skill(self, pattern_key: str, action_sequence: List[str], llm):
        """生成新技能"""
        from langchain_core.messages import SystemMessage, HumanMessage
        
        prompt = f"""你是技能工程师。以下操作序列重复出现了多次，请为其生成一个可复用的技能。

【操作序列】
{' -> '.join(action_sequence)}

【任务】
生成一个 YAML 技能定义，包含：
```yaml
name: 技能名称（英文，下划线分隔）
description: 功能描述
trigger_keywords:
  - 关键词1
  - 关键词2
steps:
  - step: 步骤1
    tool: 工具名
    params: {{}}
  - step: 步骤2
    tool: 工具名
    params: {{}}
```

只输出 YAML 内容，不要包裹代码块。
"""
        try:
            response = await llm.ainvoke([
                SystemMessage(content="你是技能工程师"),
                HumanMessage(content=prompt)
            ])
            
            skill_yaml = response.content.strip()
            
            # 保存技能
            skill_name = pattern_key.replace(' -> ', '_').replace(' ', '_')[:50]
            skill_file = self.skills_dir / f"{skill_name}.yaml"
            
            skill_file.write_text(skill_yaml, encoding='utf-8')
            
            # 注册到技能系统
            if self.skill_registry:
                self.skill_registry.load_skill(skill_file)
            
            print(f"[Evolution] 自动生成新技能：{skill_file.name}")
            return skill_file
        
        except Exception as e:
            print(f"[Evolution] 技能生成失败: {e}")
            return None


# ==================== 3. 策略优化器 ====================

class StrategyOptimizer:
    """
    策略优化器
    根据历史数据优化决策策略
    """
    def __init__(self):
        self.strategy_file = Path("data/memories/strategies.json")
        self.strategy_file.parent.mkdir(parents=True, exist_ok=True)
        self.strategies: Dict[str, Dict] = self._load()
    
    def _load(self) -> Dict:
        if self.strategy_file.exists():
            try:
                with open(self.strategy_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save(self):
        with open(self.strategy_file, 'w', encoding='utf-8') as f:
            json.dump(self.strategies, f, ensure_ascii=False, indent=2)
    
    def record_decision(self, decision_type: str, decision: str, 
                       success: bool, context: Dict = None):
        """记录决策结果"""
        if decision_type not in self.strategies:
            self.strategies[decision_type] = {}
        
        if decision not in self.strategies[decision_type]:
            self.strategies[decision_type][decision] = {
                "success_count": 0,
                "fail_count": 0,
                "contexts": []
            }
        
        strategy = self.strategies[decision_type][decision]
        if success:
            strategy["success_count"] += 1
        else:
            strategy["fail_count"] += 1
        
        # 记录上下文（最多 5 个）
        if len(strategy["contexts"]) < 5:
            strategy["contexts"].append({
                "success": success,
                "context": context or {},
                "timestamp": time.time()
            })
        
        self._save()
    
    def get_best_strategy(self, decision_type: str) -> Optional[str]:
        """获取最佳策略"""
        if decision_type not in self.strategies:
            return None
        
        strategies = self.strategies[decision_type]
        if not strategies:
            return None
        
        # 计算成功率
        best_strategy = None
        best_rate = 0
        
        for decision, data in strategies.items():
            total = data["success_count"] + data["fail_count"]
            if total < 3:  # 样本太少
                continue
            
            success_rate = data["success_count"] / total
            if success_rate > best_rate:
                best_rate = success_rate
                best_strategy = decision
        
        return best_strategy
    
    def get_strategy_report(self, decision_type: str) -> Dict:
        """获取策略报告"""
        if decision_type not in self.strategies:
            return {}
        
        report = {
            "decision_type": decision_type,
            "strategies": []
        }
        
        for decision, data in self.strategies[decision_type].items():
            total = data["success_count"] + data["fail_count"]
            report["strategies"].append({
                "decision": decision,
                "success_count": data["success_count"],
                "fail_count": data["fail_count"],
                "success_rate": data["success_count"] / total if total > 0 else 0,
                "sample_size": total
            })
        
        # 按成功率排序
        report["strategies"].sort(key=lambda x: x["success_rate"], reverse=True)
        
        return report


# ==================== 4. 进化管理器 ====================

class EvolutionManager:
    """
    进化管理器
    协调所有自进化组件
    """
    def __init__(self, llm=None, skill_registry=None):
        self.llm = llm
        self.prompt_optimizer = PromptOptimizer()
        self.skill_generator = SkillGenerator(skill_registry)
        self.strategy_optimizer = StrategyOptimizer()
    
    async def check_and_evolve(self, agent_id: str):
        """检查并执行进化"""
        if not self.llm:
            return
        
        # 1. 检查 Prompt 是否需要优化
        if self.prompt_optimizer.should_optimize(agent_id):
            print(f"[Evolution] 触发 {agent_id} 的 Prompt 优化...")
            await self.prompt_optimizer.optimize_prompt(agent_id, self.llm)
    
    def track_tool_sequence(self, tools_used: List[str]):
        """跟踪工具使用序列"""
        self.skill_generator.track_pattern(tools_used)
    
    async def generate_skill_if_needed(self, pattern_key: str, action_sequence: List[str]):
        """如果需要，生成新技能"""
        if self.skill_generator.pattern_tracker.get(pattern_key, 0) >= 3:
            await self.skill_generator.generate_skill(pattern_key, action_sequence, self.llm)
    
    def record_decision(self, decision_type: str, decision: str, 
                       success: bool, context: Dict = None):
        """记录决策"""
        self.strategy_optimizer.record_decision(decision_type, decision, success, context)
    
    def suggest_best_strategy(self, decision_type: str) -> Optional[str]:
        """推荐最佳策略"""
        return self.strategy_optimizer.get_best_strategy(decision_type)
    
    def get_evolution_report(self) -> Dict:
        """获取进化报告"""
        return {
            "prompt_versions": {
                agent_id: self.prompt_optimizer.get_performance_report(agent_id)
                for agent_id in self.prompt_optimizer.prompt_versions.keys()
            },
            "generated_skills": len(list(self.skill_generator.skills_dir.glob("*.yaml"))),
            "strategy_types": len(self.strategy_optimizer.strategies)
        }


# ==================== 5. 使用示例 ====================

async def example_usage():
    """使用示例"""
    from backend.llm_provider import get_llm
    
    llm = get_llm()
    evolution_manager = EvolutionManager(llm=llm)
    
    # 1. 注册初始 Prompt
    initial_prompt = """你是大朝议系统中的【中书省拟旨官】。
你的职责是将皇帝（用户）的口头指令转化为结构化的任务 JSON。
"""
    evolution_manager.prompt_optimizer.register_prompt("zhongshu", initial_prompt)
    
    # 2. 模拟执行与性能记录
    for i in range(15):
        success = i % 5 != 0  # 每 5 次失败 1 次
        duration = 3.5 if success else 8.0
        error_msg = "" if success else "LLM parsing failed"
        
        evolution_manager.prompt_optimizer.record_performance(
            agent_id="zhongshu",
            success=success,
            duration=duration,
            error_msg=error_msg,
            context={"iteration": i}
        )
    
    # 3. 检查是否需要优化
    if evolution_manager.prompt_optimizer.should_optimize("zhongshu"):
        print("触发 Prompt 优化...")
        optimized = await evolution_manager.prompt_optimizer.optimize_prompt("zhongshu", llm)
        print(f"优化后 Prompt 预览: {optimized[:100]}...")
    
    # 4. 跟踪工具序列
    evolution_manager.track_tool_sequence(["grep_tool", "ast_parser", "file_writer"])
    evolution_manager.track_tool_sequence(["grep_tool", "ast_parser", "file_writer"])
    evolution_manager.track_tool_sequence(["grep_tool", "ast_parser", "file_writer"])
    
    # 第 3 次后应该提示可以生成技能
    
    # 5. 记录决策
    evolution_manager.record_decision(
        decision_type="department_routing",
        decision="hubu",
        success=True,
        context={"task_type": "query"}
    )
    
    # 6. 获取进化报告
    report = evolution_manager.get_evolution_report()
    print("\n进化报告:")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
