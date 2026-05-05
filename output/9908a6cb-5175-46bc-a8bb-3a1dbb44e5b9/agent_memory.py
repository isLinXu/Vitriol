"""
大朝议 III · Agent 智能增强系统 - 完整实现代码
File: backend/agent_memory.py

三层记忆系统 + 反思引擎 + 行为学习 + 自进化
"""
import json
import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


# ==================== 数据模型 ====================

@dataclass
class Message:
    """消息对象"""
    role: str  # user / assistant / system
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class Handoff:
    """交接单（会话摘要）"""
    session_id: str
    goal: str
    constraints: List[str]
    progress: str
    decisions: List[str]
    files: List[str]
    next_steps: str
    tools_used: List[str]
    timestamp: float = field(default_factory=time.time)


class MemoryCategory(Enum):
    """长期记忆分类"""
    PREFERENCE = "preferences"    # 用户偏好
    FACT = "facts"               # 事实性知识
    SKILL = "skills"             # 自创建技能
    INSIGHT = "insights"         # 跨会话洞察
    ERROR_PATTERN = "error_patterns"  # 错误模式


# ==================== 1. 短期记忆 ====================

class ShortTermMemory:
    """
    短期记忆：当前会话的对话历史
    限制：最近 20 轮对话
    """
    def __init__(self, session_id: str, max_turns: int = 20):
        self.session_id = session_id
        self.max_turns = max_turns
        self.messages: List[Message] = []
        self.key_facts: List[str] = []
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """添加消息"""
        self.messages.append(Message(
            role=role,
            content=content,
            metadata=metadata or {}
        ))
        
        # 限制窗口大小
        if len(self.messages) > self.max_turns * 2:
            self.messages = self.messages[-(self.max_turns * 2):]
    
    def format_messages(self) -> str:
        """格式化消息为文本"""
        lines = []
        for msg in self.messages:
            prefix = "皇帝" if msg.role == "user" else "臣工"
            lines.append(f"{prefix}: {msg.content}")
        return "\n".join(lines)
    
    def get_recent_messages(self, n: int = 5) -> List[Message]:
        """获取最近 N 条消息"""
        return self.messages[-n:]
    
    async def extract_key_facts(self, llm):
        """使用 LLM 提取关键事实"""
        if len(self.messages) < 3:
            return []
        
        from langchain_core.messages import SystemMessage, HumanMessage
        from langchain_core.output_parsers import JsonOutputParser
        
        prompt = f"""从以下对话中提取值得长期记住的关键信息：

{self.format_messages()}

只提取：
- 用户明确表达的偏好（"我喜欢..."、"下次记得..."、"不要..."）
- 重要的约束条件（"必须..."、"永远不要..."）
- 项目关键信息（路径、配置、依赖）

以 JSON 数组格式输出，每条 < 50 字。示例：
["用户偏好使用 TypeScript", "项目路径在 /home/user/project"]
"""
        try:
            response = await llm.ainvoke([
                SystemMessage(content="你是信息提取专家"),
                HumanMessage(content=prompt)
            ])
            parser = JsonOutputParser()
            facts = parser.parse(response.content)
            self.key_facts.extend(facts if isinstance(facts, list) else [])
            return self.key_facts
        except Exception as e:
            print(f"[Memory] 提取关键事实失败: {e}")
            return []


# ==================== 2. 中期记忆 ====================

class MidTermMemory:
    """
    中期记忆：跨会话摘要（7-30 天）
    存储：data/memories/mid_term/{user_id}/{date}.md
    """
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.memory_dir = Path(f"data/memories/mid_term/{user_id}")
        self.memory_dir.mkdir(parents=True, exist_ok=True)
    
    async def condense_session(self, session: ShortTermMemory, llm):
        """将会话浓缩为摘要"""
        handoff = await self._generate_handoff(session, llm)
        if not handoff:
            return
        
        today_file = self.memory_dir / f"{datetime.now().date()}.md"
        with open(today_file, 'a', encoding='utf-8') as f:
            f.write(f"\n## 会话 {session.session_id[:8]}\n")
            f.write(f"**时间**: {datetime.now().strftime('%H:%M')}\n")
            f.write(f"**目标**: {handoff.goal}\n")
            f.write(f"**进展**: {handoff.progress}\n")
            if handoff.decisions:
                f.write(f"**决策**: {', '.join(handoff.decisions)}\n")
            if handoff.files:
                f.write(f"**文件**: {', '.join(handoff.files)}\n")
            f.write(f"**遗留**: {handoff.next_steps}\n\n")
        
        return handoff
    
    async def _generate_handoff(self, session: ShortTermMemory, llm) -> Optional[Handoff]:
        """生成结构化交接单"""
        from langchain_core.messages import SystemMessage, HumanMessage
        from langchain_core.output_parsers import JsonOutputParser
        
        prompt = f"""你是任务交接专家。根据以下会话历史生成结构化交接单：

{session.format_messages()}

严格按以下 JSON 格式输出：
{{
  "goal": "原始目标（1 句话）",
  "constraints": ["约束条件"],
  "progress": "已完成步骤（按时间顺序）",
  "decisions": ["关键决策"],
  "files": ["涉及的文件路径"],
  "next_steps": "下一步应该做什么",
  "tools_used": ["使用过的工具"]
}}
"""
        try:
            response = await llm.ainvoke([
                SystemMessage(content="你是任务交接专家"),
                HumanMessage(content=prompt)
            ])
            parser = JsonOutputParser()
            data = parser.parse(response.content)
            
            return Handoff(
                session_id=session.session_id,
                goal=data.get("goal", ""),
                constraints=data.get("constraints", []),
                progress=data.get("progress", ""),
                decisions=data.get("decisions", []),
                files=data.get("files", []),
                next_steps=data.get("next_steps", ""),
                tools_used=data.get("tools_used", [])
            )
        except Exception as e:
            print(f"[Memory] 生成交接单失败: {e}")
            return None
    
    def get_recent_context(self, days: int = 7) -> str:
        """获取最近 N 天的上下文摘要"""
        summaries = []
        for day in range(days):
            date = (datetime.now() - timedelta(days=day)).date()
            file = self.memory_dir / f"{date}.md"
            if file.exists():
                content = file.read_text(encoding='utf-8')
                summaries.append(f"## {date}\n{content}")
        
        return "\n---\n".join(summaries) if summaries else "（无近期记录）"
    
    def cleanup_old_memories(self, retention_days: int = 30):
        """清理超过保留期的记忆"""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        for file in self.memory_dir.glob("*.md"):
            try:
                date_str = file.stem  # e.g., "2026-05-05"
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                if file_date < cutoff_date:
                    file.unlink()
                    print(f"[Memory] 清理过期记忆: {file.name}")
            except Exception:
                continue


# ==================== 3. 长期记忆 ====================

class LongTermMemory:
    """
    长期记忆：永久知识 + 向量检索
    存储：
      - 纯文本：data/memories/long_term/{user_id}/*.md
      - 向量索引：ChromaDB
    """
    def __init__(self, user_id: str = "default", vector_store=None):
        self.user_id = user_id
        self.memory_dir = Path(f"data/memories/long_term/{user_id}")
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store = vector_store
        
        # 创建子目录
        for category in MemoryCategory:
            (self.memory_dir / category.value).mkdir(exist_ok=True)
    
    def remember(self, category: MemoryCategory, title: str, content: str):
        """记住一条长期信息"""
        safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in title)
        file = self.memory_dir / category.value / f"{safe_title}.md"
        
        # 写入纯文本
        with open(file, 'w', encoding='utf-8') as f:
            f.write(f"# {title}\n\n")
            f.write(f"{content}\n\n")
            f.write(f"---\n*创建时间: {datetime.now()}*\n")
        
        # 写入向量数据库
        if self.vector_store:
            try:
                self.vector_store.add_document(
                    content=f"{title}\n{content}",
                    metadata={
                        "category": category.value,
                        "title": title,
                        "user_id": self.user_id,
                        "file_path": str(file)
                    }
                )
            except Exception as e:
                print(f"[Memory] 向量存储失败: {e}")
    
    def recall(self, query: str, top_k: int = 5, category: Optional[MemoryCategory] = None) -> List[Dict]:
        """根据查询召回相关记忆"""
        if not self.vector_store:
            # 回退到文件搜索
            return self._fallback_search(query, top_k, category)
        
        try:
            filter_dict = {"user_id": self.user_id}
            if category:
                filter_dict["category"] = category.value
            
            results = self.vector_store.search(
                query=query,
                filter=filter_dict,
                top_k=top_k
            )
            return results
        except Exception as e:
            print(f"[Memory] 向量召回失败: {e}")
            return self._fallback_search(query, top_k, category)
    
    def _fallback_search(self, query: str, top_k: int, category: Optional[MemoryCategory]) -> List[Dict]:
        """简单关键词搜索（回退方案）"""
        results = []
        query_lower = query.lower()
        
        search_dirs = [self.memory_dir / category.value] if category else \
                      [self.memory_dir / c.value for c in MemoryCategory]
        
        for dir in search_dirs:
            if not dir.exists():
                continue
            
            for file in dir.glob("*.md"):
                content = file.read_text(encoding='utf-8')
                if query_lower in content.lower():
                    results.append({
                        "title": file.stem,
                        "content": content[:200],
                        "file_path": str(file),
                        "category": dir.name
                    })
        
        return results[:top_k]
    
    def forget(self, title: str, category: MemoryCategory):
        """删除记忆"""
        safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in title)
        file = self.memory_dir / category.value / f"{safe_title}.md"
        if file.exists():
            file.unlink()


# ==================== 4. 反思引擎 ====================

class ReflectionEngine:
    """
    自我反思引擎
    在每次任务结束后自动运行，分析成败得失
    """
    def __init__(self, memory: LongTermMemory):
        self.memory = memory
    
    async def reflect(self, session: ShortTermMemory, final_state: Dict[str, Any], llm):
        """对一次完整任务进行反思"""
        analysis = await self._analyze_session(session, final_state, llm)
        
        if not analysis:
            return None
        
        # 记录洞察到长期记忆
        if analysis.get("insights"):
            for insight in analysis["insights"]:
                self.memory.remember(
                    category=MemoryCategory.INSIGHT,
                    title=insight["title"],
                    content=insight["content"]
                )
        
        # 记录用户偏好
        if analysis.get("user_preferences"):
            for pref in analysis["user_preferences"]:
                self.memory.remember(
                    category=MemoryCategory.PREFERENCE,
                    title=pref["key"],
                    content=pref["value"]
                )
        
        return analysis
    
    async def _analyze_session(self, session: ShortTermMemory, final_state: Dict, llm) -> Optional[Dict]:
        """深度分析会话"""
        from langchain_core.messages import SystemMessage, HumanMessage
        from langchain_core.output_parsers import JsonOutputParser
        
        is_success = final_state.get('execution_result') and not final_state.get('error_msg')
        
        prompt = f"""你是朝堂智囊，负责事后分析每次任务的得失。

【任务记录】
{session.format_messages()}

【最终状态】
- 是否成功：{is_success}
- 错误信息：{final_state.get('error_msg', '无')}
- 使用Token：{final_state.get('tokens_used', {{}})}
- 耗时：{final_state.get('duration', 0)}秒

【分析维度】
1. **成功因素**：哪些决策是正确的？为什么？
2. **失败原因**：哪里出错了？根因是什么？
3. **改进空间**：下次如何做得更好？
4. **可复用模式**：这次经验有哪些通用价值？
5. **用户洞察**：从对话中学到了用户的哪些偏好/习惯？

严格按以下 JSON 格式输出：
{{
  "success": true/false,
  "insights": [{{"title": "洞察标题", "content": "详细描述"}}],
  "error_patterns": [{{"pattern": "错误模式", "remedy": "补救方案"}}],
  "success_patterns": [{{"pattern": "成功模式", "context": "适用场景"}}],
  "user_preferences": [{{"key": "偏好类型", "value": "具体内容"}}]
}}
"""
        try:
            response = await llm.ainvoke([
                SystemMessage(content="你是朝堂智囊，擅长事后分析"),
                HumanMessage(content=prompt)
            ])
            parser = JsonOutputParser()
            return parser.parse(response.content)
        except Exception as e:
            print(f"[Reflection] 分析失败: {e}")
            return None


# ==================== 5. 错误知识库 ====================

class ErrorKnowledgeBase:
    """
    错误模式知识库
    记录历史错误及补救方案
    """
    def __init__(self):
        self.kb_file = Path("data/memories/error_patterns.json")
        self.kb_file.parent.mkdir(parents=True, exist_ok=True)
        self.patterns: Dict[str, dict] = self._load()
    
    def _load(self) -> Dict:
        if self.kb_file.exists():
            try:
                with open(self.kb_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save(self):
        with open(self.kb_file, 'w', encoding='utf-8') as f:
            json.dump(self.patterns, f, ensure_ascii=False, indent=2)
    
    def add_pattern(self, error_msg: str, remedy: str, context: Dict = None):
        """记录新的错误模式"""
        pattern_id = self._hash_error(error_msg)
        
        if pattern_id in self.patterns:
            self.patterns[pattern_id]["count"] += 1
            self.patterns[pattern_id]["examples"].append({
                "error_msg": error_msg,
                "context": context or {},
                "timestamp": time.time()
            })
        else:
            self.patterns[pattern_id] = {
                "pattern": self._generalize_error(error_msg),
                "remedy": remedy,
                "count": 1,
                "examples": [{
                    "error_msg": error_msg,
                    "context": context or {},
                    "timestamp": time.time()
                }],
                "created_at": time.time()
            }
        
        self._save()
    
    def search_remedy(self, error_msg: str) -> Optional[str]:
        """搜索补救方案"""
        pattern_id = self._hash_error(error_msg)
        if pattern_id in self.patterns:
            return self.patterns[pattern_id]["remedy"]
        return None
    
    def find_similar_tasks(self, task_json: Dict) -> List[Dict]:
        """查找类似任务的历史错误"""
        # 简化实现：基于 action 匹配
        action = task_json.get("action", "").lower()
        similar = []
        
        for pattern_id, data in self.patterns.items():
            for example in data["examples"]:
                if action in example.get("context", {}).get("action", "").lower():
                    similar.append({
                        "pattern": data["pattern"],
                        "remedy": data["remedy"],
                        "count": data["count"]
                    })
        
        return similar[:3]  # 返回最多 3 个
    
    @staticmethod
    def _hash_error(error_msg: str) -> str:
        """对错误信息生成哈希"""
        # 简化错误信息后生成哈希
        simplified = error_msg.lower()
        # 移除数字、路径等变量
        import re
        simplified = re.sub(r'\d+', '<NUM>', simplified)
        simplified = re.sub(r'/[\w/]+', '<PATH>', simplified)
        return hashlib.md5(simplified.encode()).hexdigest()[:16]
    
    @staticmethod
    def _generalize_error(error_msg: str) -> str:
        """泛化错误信息"""
        import re
        generalized = error_msg
        generalized = re.sub(r'\d+', '<NUM>', generalized)
        generalized = re.sub(r'/[\w/.]+', '<PATH>', generalized)
        generalized = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '<UUID>', generalized)
        return generalized


# ==================== 6. 工具使用模式学习器 ====================

class ToolUsagePatternLearner:
    """
    工具使用模式学习器
    自动发现高效的工具组合
    """
    def __init__(self):
        self.patterns_file = Path("data/memories/tool_patterns.json")
        self.patterns_file.parent.mkdir(parents=True, exist_ok=True)
        self.patterns: List[Dict] = self._load()
    
    def _load(self) -> List[Dict]:
        if self.patterns_file.exists():
            try:
                with open(self.patterns_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return []
        return []
    
    def _save(self):
        with open(self.patterns_file, 'w', encoding='utf-8') as f:
            json.dump(self.patterns, f, ensure_ascii=False, indent=2)
    
    def record_execution(self, task_desc: str, tools_used: List[str], 
                        success: bool, duration: float):
        """记录一次执行"""
        self.patterns.append({
            "task_type": self._classify_task(task_desc),
            "tools": tools_used,
            "success": success,
            "duration": duration,
            "timestamp": time.time()
        })
        
        # 限制大小
        if len(self.patterns) > 1000:
            self.patterns = self.patterns[-500:]
        
        self._save()
    
    def suggest_tools(self, task_desc: str) -> List[str]:
        """根据任务描述推荐工具"""
        task_type = self._classify_task(task_desc)
        
        # 查找相似任务的成功案例
        successful = [
            p for p in self.patterns
            if p["task_type"] == task_type and p["success"]
        ]
        
        if not successful:
            return []
        
        # 统计工具频率
        tool_freq = {}
        for p in successful:
            for tool in p["tools"]:
                tool_freq[tool] = tool_freq.get(tool, 0) + 1
        
        # 返回高频工具
        return sorted(tool_freq.keys(), key=lambda t: tool_freq[t], reverse=True)[:5]
    
    @staticmethod
    def _classify_task(task_desc: str) -> str:
        """简单任务分类"""
        desc_lower = task_desc.lower()
        if "file" in desc_lower or "文件" in desc_lower:
            return "file_operation"
        elif "search" in desc_lower or "搜索" in desc_lower:
            return "search"
        elif "execute" in desc_lower or "执行" in desc_lower:
            return "execution"
        elif "query" in desc_lower or "查询" in desc_lower:
            return "query"
        else:
            return "general"


# ==================== 7. 统一记忆管理器 ====================

class MemoryManager:
    """
    统一记忆管理器
    协调三层记忆系统
    """
    def __init__(self, user_id: str = "default", vector_store=None, llm=None):
        self.user_id = user_id
        self.llm = llm
        
        self.short_term = ShortTermMemory(session_id=f"session_{int(time.time())}")
        self.mid_term = MidTermMemory(user_id=user_id)
        self.long_term = LongTermMemory(user_id=user_id, vector_store=vector_store)
        
        self.reflection_engine = ReflectionEngine(self.long_term)
        self.error_kb = ErrorKnowledgeBase()
        self.tool_learner = ToolUsagePatternLearner()
    
    def new_session(self, session_id: str):
        """开始新会话"""
        self.short_term = ShortTermMemory(session_id=session_id)
    
    def add_turn(self, user_msg: str, assistant_msg: str, metadata: Dict = None):
        """添加一轮对话"""
        self.short_term.add_message("user", user_msg, metadata)
        self.short_term.add_message("assistant", assistant_msg, metadata)
    
    async def end_session(self, final_state: Dict):
        """结束会话并浓缩记忆"""
        if not self.llm:
            return
        
        # 1. 提取关键事实
        await self.short_term.extract_key_facts(self.llm)
        
        # 2. 浓缩为中期记忆
        await self.mid_term.condense_session(self.short_term, self.llm)
        
        # 3. 反思分析
        await self.reflection_engine.reflect(self.short_term, final_state, self.llm)
    
    def recall_context(self, query: str) -> str:
        """召回相关上下文"""
        # 1. 长期记忆召回
        long_term_memories = self.long_term.recall(query, top_k=3)
        
        # 2. 中期记忆召回
        mid_term_context = self.mid_term.get_recent_context(days=7)
        
        # 3. 组合返回
        context_parts = []
        
        if long_term_memories:
            context_parts.append("【长期记忆】")
            for mem in long_term_memories:
                context_parts.append(f"- {mem.get('title', '')}: {mem.get('content', '')[:100]}")
        
        if mid_term_context and mid_term_context != "（无近期记录）":
            context_parts.append("\n【近期上下文】")
            context_parts.append(mid_term_context[:500])
        
        return "\n".join(context_parts) if context_parts else "（无相关记忆）"


# ==================== 8. 使用示例 ====================

async def example_usage():
    """使用示例"""
    from backend.llm_provider import get_llm
    
    llm = get_llm()
    memory_manager = MemoryManager(user_id="emperor_001", llm=llm)
    
    # 1. 新会话
    memory_manager.new_session("task_abc123")
    
    # 2. 添加对话
    memory_manager.add_turn(
        user_msg="帮我分析这个项目的架构",
        assistant_msg="好的陛下，正在分析..."
    )
    
    # 3. 召回上下文
    context = memory_manager.recall_context("项目架构")
    print("召回上下文:", context)
    
    # 4. 结束会话
    final_state = {
        "execution_result": "分析完成",
        "error_msg": None,
        "tokens_used": {"zhongshu": 100, "hubu": 50},
        "duration": 15.3
    }
    await memory_manager.end_session(final_state)
    
    # 5. 记录错误模式
    memory_manager.error_kb.add_pattern(
        error_msg="FileNotFoundError: /tmp/data.json",
        remedy="检查文件路径是否存在",
        context={"action": "read_file", "path": "/tmp/data.json"}
    )
    
    # 6. 工具学习
    memory_manager.tool_learner.record_execution(
        task_desc="搜索代码",
        tools_used=["grep_tool", "ast_parser"],
        success=True,
        duration=2.5
    )
    
    suggested_tools = memory_manager.tool_learner.suggest_tools("搜索代码")
    print("推荐工具:", suggested_tools)


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
