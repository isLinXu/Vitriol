# 大朝议 III · Agent 智能增强方案
## 记忆、行为学习与自进化系统设计

> **设计日期**：2026-05-05  
> **针对项目**：`/Users/gatilin/PycharmProjects/dachaoyi3`  
> **设计目标**：赋予 Agent 长期记忆、行为学习、自我优化能力，实现真正的智能体自进化

---

## 0. 设计理念与架构概览

### 核心理念

**从"无状态工具调用"到"有记忆的智能体"**

当前系统的 Agent 本质上是**无状态的 LLM 包装器**：每次对话后记忆清空，不会从过去的错误中学习，不会随使用变得更智能。这是传统 RAG/Agent 系统的通病。

本方案参考 **Hermes-Agent** 和 **OpenClaw** 的设计思想，结合大朝议的朝堂隐喻，构建一套完整的 **Agent 认知架构**：

```
┌─────────────────────────────────────────────────────────────┐
│                      Agent 认知架构                          │
├─────────────────────────────────────────────────────────────┤
│  Layer 5: Self-Evolution （自我进化）                       │
│   - 策略优化器：根据执行反馈调整决策策略                     │
│   - 工具学习器：自动发现新工具组合模式                       │
│   - Prompt自调优：根据成功率优化系统提示词                   │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: Reflection （自我反思）                           │
│   - 事后分析：每次任务结束后分析成败原因                     │
│   - 错误模式识别：归纳常见错误并记录避坑指南                 │
│   - 跨会话洞察：发现用户偏好、习惯、隐性需求                 │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Long-Term Memory （三层记忆系统）                 │
│   - 短期记忆：当前会话（20 轮对话）                         │
│   - 中期记忆：近期摘要（7-30 天，按主题聚类）                │
│   - 长期记忆：永久知识（用户偏好/事实/技能）+ 向量检索       │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Contextual Reasoning （上下文推理）               │
│   - 意图延续：识别"继续上次"并自动恢复上下文                 │
│   - 交接单生成：长对话自动压缩为结构化交接单                 │
│   - 依赖分析：识别任务间依赖关系                             │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Tool-Calling Agent （现有能力）                   │
│   - 工具调用、LangGraph 编排、审批流程                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. 三层记忆系统设计

### 1.1 短期记忆（Session Memory）

**职责**：存储当前会话的完整对话历史。

**存储结构**：
```python
class ShortTermMemory:
    """
    短期记忆：存储当前会话的对话历史
    限制：最近 20 轮对话
    """
    def __init__(self, session_id: str, max_turns: int = 20):
        self.session_id = session_id
        self.max_turns = max_turns
        self.messages: List[Message] = []  # 完整消息链
        self.key_facts: List[str] = []     # 关键事实（LLM 提取）
    
    def add_message(self, role: str, content: str, metadata: dict = None):
        """添加消息到短期记忆"""
        self.messages.append(Message(role=role, content=content, metadata=metadata))
        
        # 限制窗口大小
        if len(self.messages) > self.max_turns * 2:  # user + assistant = 2
            self.messages = self.messages[-(self.max_turns * 2):]
    
    def extract_key_facts(self, llm):
        """使用 LLM 提取当前对话中的关键事实"""
        # 调用 LLM：提取本轮对话中值得长期记住的信息
        prompt = f"""
从以下对话中提取值得长期记住的关键信息：
{self.format_messages()}

只提取：
- 用户明确表达的偏好（"我喜欢..."、"下次记得..."）
- 重要的约束条件（"不要..."、"必须..."）
- 项目关键信息（路径、配置、依赖）

以 JSON 数组格式输出，每条 < 50 字。
"""
        return llm.invoke(prompt)
```

**生命周期**：
- 创建：用户发起会话时初始化
- 更新：每轮对话后追加消息
- 销毁：会话结束时浓缩为中期记忆

---

### 1.2 中期记忆（Mid-Term Memory）

**职责**：跨会话摘要，保留最近 7-30 天的关键上下文。

**存储结构**：
```python
class MidTermMemory:
    """
    中期记忆：跨会话摘要（7-30天）
    存储：data/memories/mid_term/{user_id}/{date}.md
    """
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.memory_dir = Path(f"data/memories/mid_term/{user_id}")
        self.memory_dir.mkdir(parents=True, exist_ok=True)
    
    def condense_session(self, session: ShortTermMemory):
        """
        将一个会话浓缩为摘要并追加到今日记忆
        使用 LLM 生成结构化交接单
        """
        handoff = self._generate_handoff(session)
        
        today_file = self.memory_dir / f"{datetime.now().date()}.md"
        with open(today_file, 'a', encoding='utf-8') as f:
            f.write(f"\n## 会话 {session.session_id[:8]}\n")
            f.write(f"**时间**: {datetime.now().strftime('%H:%M')}\n")
            f.write(f"**目标**: {handoff['goal']}\n")
            f.write(f"**进展**: {handoff['progress']}\n")
            f.write(f"**决策**: {', '.join(handoff['decisions'])}\n")
            f.write(f"**遗留**: {handoff['next_steps']}\n\n")
    
    def _generate_handoff(self, session: ShortTermMemory) -> dict:
        """生成交接单（Structured Handoff）"""
        prompt = f"""
你是任务交接专家。根据以下会话历史生成结构化交接单：

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
        return llm.invoke(prompt, output_parser=JsonOutputParser())
    
    def get_recent_context(self, days: int = 7) -> str:
        """获取最近 N 天的上下文摘要"""
        summaries = []
        for day in range(days):
            date = (datetime.now() - timedelta(days=day)).date()
            file = self.memory_dir / f"{date}.md"
            if file.exists():
                summaries.append(file.read_text(encoding='utf-8'))
        return "\n---\n".join(summaries)
```

**生命周期**：
- 创建：会话结束时从短期记忆浓缩
- 更新：每日追加当天所有会话摘要
- 清理：30 天后自动归档或删除

---

### 1.3 长期记忆（Long-Term Memory）

**职责**：永久存储用户偏好、项目知识、技能库，支持语义检索。

**存储结构**：
```python
class LongTermMemory:
    """
    长期记忆：永久知识 + 向量检索
    存储：
      - 纯文本：data/memories/long_term/{user_id}/*.md
      - 向量索引：ChromaDB
    """
    def __init__(self, user_id: str, vector_store):
        self.user_id = user_id
        self.memory_dir = Path(f"data/memories/long_term/{user_id}")
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store = vector_store
        
        # 子分类目录
        (self.memory_dir / "preferences").mkdir(exist_ok=True)   # 用户偏好
        (self.memory_dir / "facts").mkdir(exist_ok=True)         # 事实性知识
        (self.memory_dir / "skills").mkdir(exist_ok=True)        # 自创建的技能
        (self.memory_dir / "insights").mkdir(exist_ok=True)      # 跨会话洞察
    
    def remember(self, category: str, title: str, content: str):
        """
        记住一条长期信息
        AI 主动决定什么值得长期记住
        """
        file = self.memory_dir / category / f"{title}.md"
        
        # 写入纯文本文件
        with open(file, 'w', encoding='utf-8') as f:
            f.write(f"# {title}\n\n")
            f.write(f"{content}\n\n")
            f.write(f"---\n*创建时间: {datetime.now()}*\n")
        
        # 同时写入向量数据库
        self.vector_store.add_document(
            content=content,
            metadata={"category": category, "title": title, "user_id": self.user_id}
        )
    
    def recall(self, query: str, top_k: int = 5) -> List[dict]:
        """
        根据查询召回相关记忆
        使用向量检索 + 重排序
        """
        # 1. 向量检索
        candidates = self.vector_store.search(
            query=query,
            filter={"user_id": self.user_id},
            top_k=top_k * 2  # 多召回一些候选
        )
        
        # 2. LLM 重排序（根据相关性）
        reranked = self._rerank(query, candidates)
        
        return reranked[:top_k]
    
    def _rerank(self, query: str, candidates: List[dict]) -> List[dict]:
        """使用 LLM 对召回结果重排序"""
        prompt = f"""
查询：{query}

候选记忆：
{json.dumps(candidates, ensure_ascii=False, indent=2)}

根据查询意图，对候选记忆按相关性排序。输出 JSON 数组，只包含 ID。
"""
        ranked_ids = llm.invoke(prompt, output_parser=JsonOutputParser())
        # 按排序重新组织
        id_map = {c["id"]: c for c in candidates}
        return [id_map[id] for id in ranked_ids if id in id_map]
```

**自动化记忆策略**：
- Agent 在会话结束时判断：哪些信息值得长期记住
- 用户明确表达的偏好（"记住这个"、"下次这样"）→ 立即记入长期记忆
- 重复出现 3 次以上的操作模式 → 自动识别为技能并记录

---

## 2. 自我反思机制（Reflection Layer）

### 2.1 事后分析（Post-Task Analysis）

每个任务结束后，自动进行深度反思：

```python
class ReflectionEngine:
    """
    自我反思引擎
    在每次任务结束后自动运行，分析成败得失
    """
    def __init__(self, memory: LongTermMemory):
        self.memory = memory
    
    async def reflect(self, session: ShortTermMemory, final_state: dict):
        """
        对一次完整任务进行反思
        """
        analysis = await self._analyze_session(session, final_state)
        
        # 记录洞察到长期记忆
        if analysis.get("insights"):
            for insight in analysis["insights"]:
                self.memory.remember(
                    category="insights",
                    title=insight["title"],
                    content=insight["content"]
                )
        
        # 记录错误模式
        if analysis.get("error_patterns"):
            self._update_error_kb(analysis["error_patterns"])
        
        # 记录成功模式
        if analysis.get("success_patterns"):
            self._update_success_kb(analysis["success_patterns"])
        
        return analysis
    
    async def _analyze_session(self, session, final_state) -> dict:
        """
        深度分析会话
        """
        prompt = f"""
你是朝堂智囊，负责事后分析每次任务的得失。

【任务记录】
{session.format_messages()}

【最终状态】
- 是否成功：{final_state.get('execution_result') and not final_state.get('error_msg')}
- 错误信息：{final_state.get('error_msg', '无')}
- 使用工具：{final_state.get('tokens_used', {})}
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
  "insights": [
    {{"title": "洞察标题", "content": "详细描述"}}
  ],
  "error_patterns": [
    {{"pattern": "错误模式", "remedy": "补救方案"}}
  ],
  "success_patterns": [
    {{"pattern": "成功模式", "context": "适用场景"}}
  ],
  "user_preferences": [
    {{"key": "偏好类型", "value": "具体内容"}}
  ]
}}
"""
        return await llm.ainvoke(prompt, output_parser=JsonOutputParser())
```

**反思触发时机**：
1. **主动反思**：每次任务结束（`monitor_node` 之后）
2. **错误反思**：遇到错误时立即分析根因
3. **周期反思**：每日/每周汇总，发现长期趋势

---

### 2.2 错误模式知识库

```python
class ErrorKnowledgeBase:
    """
    错误模式知识库
    记录历史错误及补救方案，避免重复犯错
    """
    def __init__(self):
        self.kb_file = Path("data/memories/error_patterns.json")
        self.patterns: Dict[str, dict] = self._load()
    
    def add_pattern(self, error_msg: str, remedy: str, context: dict):
        """记录新的错误模式"""
        pattern_id = self._hash_error(error_msg)
        
        if pattern_id in self.patterns:
            # 已存在，增加计数和示例
            self.patterns[pattern_id]["count"] += 1
            self.patterns[pattern_id]["examples"].append({
                "error_msg": error_msg,
                "context": context,
                "timestamp": time.time()
            })
        else:
            # 新模式
            self.patterns[pattern_id] = {
                "pattern": self._generalize_error(error_msg),
                "remedy": remedy,
                "count": 1,
                "examples": [{"error_msg": error_msg, "context": context}],
                "created_at": time.time()
            }
        
        self._save()
    
    def search_remedy(self, error_msg: str) -> Optional[str]:
        """根据错误信息搜索补救方案"""
        pattern_id = self._hash_error(error_msg)
        if pattern_id in self.patterns:
            return self.patterns[pattern_id]["remedy"]
        
        # 语义相似搜索
        candidates = [(pid, p) for pid, p in self.patterns.items()]
        # ... 使用 embedding 相似度 ...
        return None
    
    def _generalize_error(self, error_msg: str) -> str:
        """
        将具体错误泛化为模式
        例如："FileNotFoundError: /path/to/file" → "FileNotFoundError: <PATH>"
        """
        # 使用 LLM 提取错误模式
        prompt = f"""
将以下具体错误信息泛化为通用模式（用 <VAR> 替换变量）：
{error_msg}

示例：
  "KeyError: 'user_id'" → "KeyError: <KEY>"
  "连接超时: 192.168.1.1:8080" → "连接超时: <IP>:<PORT>"
"""
        return llm.invoke(prompt)
```

**集成到 Agent 决策**：
```python
async def menshu_node_with_memory(state: AgentState):
    """
    门下省审核（带记忆增强）
    """
    task_json = state.get("task_json", {})
    
    # 1. 调用原审核逻辑
    report = await menshu.review(task_json)
    
    # 2. 检查历史错误模式
    error_kb = ErrorKnowledgeBase()
    similar_errors = error_kb.find_similar_tasks(task_json)
    
    if similar_errors:
        # 发现类似任务历史上出过错
        report["findings"].append(
            f"⚠️ 历史记录：类似任务曾出现 {len(similar_errors)} 次错误"
        )
        report["recommendation"] += f"\n建议参考：{similar_errors[0]['remedy']}"
        
        # 记录到思维过程
        add_thought(
            state, "menshu", "门下省",
            f"查阅案卷：此类任务历史上曾失手，建议采取预防措施",
            "从错误知识库召回类似案例",
            f"发现 {len(similar_errors)} 个相关先例"
        )
    
    state["safety_report"] = report
    return state
```

---

## 3. 行为学习与策略优化

### 3.1 工具使用模式学习

```python
class ToolUsagePatternLearner:
    """
    工具使用模式学习器
    自动发现高效的工具组合
    """
    def __init__(self):
        self.patterns: List[ToolPattern] = []
        self.success_db = Path("data/memories/tool_patterns.json")
    
    def record_execution(self, task_desc: str, tools_used: List[str], 
                        success: bool, duration: float):
        """记录一次执行"""
        pattern = {
            "task_type": self._classify_task(task_desc),
            "tools": tools_used,
            "success": success,
            "duration": duration,
            "timestamp": time.time()
        }
        # 存储并聚类
        self._add_to_cluster(pattern)
    
    def suggest_tools(self, task_desc: str) -> List[str]:
        """
        根据任务描述推荐工具序列
        基于历史成功案例
        """
        task_type = self._classify_task(task_desc)
        
        # 查找相似任务的成功案例
        successful_patterns = [
            p for p in self.patterns
            if p["task_type"] == task_type and p["success"]
        ]
        
        if not successful_patterns:
            return []
        
        # 统计工具出现频率
        tool_freq = {}
        for p in successful_patterns:
            for tool in p["tools"]:
                tool_freq[tool] = tool_freq.get(tool, 0) + 1
        
        # 返回高频工具
        return sorted(tool_freq.keys(), key=lambda t: tool_freq[t], reverse=True)
```

**集成到 Agent 执行**：
```python
async def execution_node_with_learning(state: AgentState):
    """
    执行节点（带学习增强）
    """
    task = state.get("task_json", {})
    
    # 1. 查询工具推荐
    learner = ToolUsagePatternLearner()
    suggested_tools = learner.suggest_tools(task.get("action", ""))
    
    if suggested_tools:
        add_thought(
            state, "hubu", "户部",
            f"查阅成功案卷：类似任务历史上常用工具 {suggested_tools[:3]}",
            "从工具模式库召回推荐",
            "准备参考历史经验"
        )
    
    # 2. 执行任务（原逻辑）
    start_time = time.time()
    dept_id = normalize_department(state["routing_info"].get("department"))
    dept_agent = DEPT_AGENTS.get(dept_id)
    
    try:
        result = await dept_agent.execute(task)
        duration = time.time() - start_time
        
        # 3. 记录执行结果到学习器
        tools_used = result.get("tools_used", [])
        learner.record_execution(
            task_desc=task.get("action"),
            tools_used=tools_used,
            success=True,
            duration=duration
        )
        
        state["execution_result"] = result.get("message")
    except Exception as e:
        duration = time.time() - start_time
        learner.record_execution(
            task_desc=task.get("action"),
            tools_used=[],
            success=False,
            duration=duration
        )
        raise
    
    return state
```

---

## 4. 自进化循环（Self-Evolution）

### 4.1 Prompt 自调优

```python
class PromptOptimizer:
    """
    Prompt 自调优器
    根据执行成功率自动优化系统提示词
    """
    def __init__(self):
        self.prompt_versions: Dict[str, List[dict]] = {}
        self.performance_log = Path("data/memories/prompt_performance.json")
    
    def register_prompt(self, agent_id: str, prompt: str, version: int = 1):
        """注册一个 Prompt 版本"""
        if agent_id not in self.prompt_versions:
            self.prompt_versions[agent_id] = []
        
        self.prompt_versions[agent_id].append({
            "version": version,
            "prompt": prompt,
            "success_count": 0,
            "fail_count": 0,
            "avg_duration": 0,
            "created_at": time.time()
        })
    
    def record_performance(self, agent_id: str, success: bool, duration: float):
        """记录性能"""
        current_version = self.prompt_versions[agent_id][-1]
        if success:
            current_version["success_count"] += 1
        else:
            current_version["fail_count"] += 1
        
        # 更新平均耗时
        total = current_version["success_count"] + current_version["fail_count"]
        current_version["avg_duration"] = (
            current_version["avg_duration"] * (total - 1) + duration
        ) / total
    
    def should_optimize(self, agent_id: str) -> bool:
        """
        判断是否需要优化 Prompt
        条件：失败率 > 20% 或 平均耗时 > 10秒
        """
        current = self.prompt_versions[agent_id][-1]
        total = current["success_count"] + current["fail_count"]
        
        if total < 10:
            return False  # 样本太少
        
        fail_rate = current["fail_count"] / total
        return fail_rate > 0.2 or current["avg_duration"] > 10
    
    async def optimize_prompt(self, agent_id: str, error_samples: List[dict]):
        """
        使用 LLM 优化 Prompt
        """
        current_prompt = self.prompt_versions[agent_id][-1]["prompt"]
        
        optimization_prompt = f"""
你是 Prompt 工程专家。以下是一个 Agent 的系统提示词及其失败案例：

【当前 Prompt】
{current_prompt}

【失败案例】（最近 5 个）
{json.dumps(error_samples[:5], ensure_ascii=False, indent=2)}

【任务】
分析失败原因，优化 Prompt 以：
1. 减少错误率
2. 提升响应速度
3. 保持输出格式稳定

输出优化后的 Prompt（纯文本，不要包裹代码块）。
"""
        optimized_prompt = await llm.ainvoke(optimization_prompt)
        
        # 注册新版本
        new_version = len(self.prompt_versions[agent_id]) + 1
        self.register_prompt(agent_id, optimized_prompt.content, new_version)
        
        return optimized_prompt.content
```

**集成到 Agent 初始化**：
```python
class ZhongshuAgentWithEvolution(ZhongshuAgent):
    def __init__(self):
        self.prompt_optimizer = PromptOptimizer()
        
        # 注册初始 Prompt
        initial_prompt = "你现在是大朝议系统中的【中书省拟旨官】..."
        self.prompt_optimizer.register_prompt("zhongshu", initial_prompt)
        
        # 加载最新 Prompt
        latest_prompt = self.prompt_optimizer.get_latest_prompt("zhongshu")
        self.prompt = ChatPromptTemplate.from_template(latest_prompt)
        self.chain = self.prompt | llm | self.parser
    
    async def draft(self, decree: str) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            result = await self.chain.ainvoke({
                "decree": decree,
                "format_instructions": self.parser.get_format_instructions()
            })
            duration = time.time() - start_time
            
            # 记录成功
            self.prompt_optimizer.record_performance("zhongshu", True, duration)
            return result
        
        except Exception as e:
            duration = time.time() - start_time
            
            # 记录失败
            self.prompt_optimizer.record_performance("zhongshu", False, duration)
            
            # 检查是否需要优化
            if self.prompt_optimizer.should_optimize("zhongshu"):
                print("[进化] 检测到中书省拟旨频繁失败，启动 Prompt 优化...")
                optimized = await self.prompt_optimizer.optimize_prompt(
                    "zhongshu",
                    error_samples=[{"decree": decree, "error": str(e)}]
                )
                print(f"[进化] 已生成优化版本 {len(self.prompt_optimizer.prompt_versions['zhongshu'])}")
                
                # 重新加载 Chain
                self.prompt = ChatPromptTemplate.from_template(optimized)
                self.chain = self.prompt | llm | self.parser
            
            raise
```

---

### 4.2 动态技能生成

```python
class SkillGenerator:
    """
    动态技能生成器
    当检测到重复操作模式时，自动生成新技能
    """
    def __init__(self, skill_registry):
        self.skill_registry = skill_registry
        self.pattern_tracker: Dict[str, int] = {}  # 模式 -> 出现次数
    
    def track_pattern(self, action_sequence: List[str]):
        """跟踪操作序列"""
        pattern_key = " -> ".join(action_sequence)
        self.pattern_tracker[pattern_key] = self.pattern_tracker.get(pattern_key, 0) + 1
        
        # 出现 3 次以上，生成技能
        if self.pattern_tracker[pattern_key] >= 3:
            self._generate_skill(pattern_key, action_sequence)
    
    async def _generate_skill(self, pattern_key: str, action_sequence: List[str]):
        """生成新技能"""
        prompt = f"""
你是技能工程师。以下操作序列重复出现了 3 次以上，请为其生成一个可复用的技能。

【操作序列】
{' -> '.join(action_sequence)}

【任务】
生成一个 YAML 技能定义，包含：
- name: 技能名称
- description: 功能描述
- trigger_keywords: 触发关键词（用户说什么时自动匹配）
- steps: 执行步骤（工具调用序列）

输出 YAML 格式。
"""
        skill_yaml = await llm.ainvoke(prompt)
        
        # 保存到技能目录
        skill_file = Path(f"backend/skills/auto_generated/{pattern_key.replace(' -> ', '_')}.yaml")
        skill_file.parent.mkdir(parents=True, exist_ok=True)
        skill_file.write_text(skill_yaml.content, encoding='utf-8')
        
        # 注册到技能系统
        self.skill_registry.load_skill(skill_file)
        
        print(f"[进化] 自动生成新技能：{pattern_key}")
```

---

## 5. 实施路线图

### Phase 1: 三层记忆系统（2 周）
1. 实现 `ShortTermMemory` / `MidTermMemory` / `LongTermMemory`
2. 集成到 `edict_graph.py` 的状态管理
3. 在 `monitor_node` 结束时触发记忆浓缩
4. 测试记忆召回与上下文恢复

### Phase 2: 反思机制（1 周）
1. 实现 `ReflectionEngine`
2. 实现 `ErrorKnowledgeBase`
3. 在 `menshu_node` / `execution_node` 集成错误模式检索
4. 测试事后分析与洞察提取

### Phase 3: 行为学习（1 周）
1. 实现 `ToolUsagePatternLearner`
2. 在 `execution_node` 记录工具使用
3. 在任务分配时推荐工具
4. 测试学习效果

### Phase 4: 自进化（2 周）
1. 实现 `PromptOptimizer`
2. 实现 `SkillGenerator`
3. 集成到核心 Agent 类
4. 运行 1 周观察自动优化效果

---

## 6. 预期效果

| 指标 | 当前 | 增强后 |
|---|---|---|
| 跨会话上下文保持 | ❌ | ✅ (自动召回中期/长期记忆) |
| 重复错误率 | 无统计 | 预计降低 **50%+** (错误知识库) |
| 工具选择准确率 | 依赖 LLM | 提升 **30%+** (学习器推荐) |
| Prompt 稳定性 | 固定 | 自动优化，失败率 < 10% |
| 技能库扩展 | 手动 | 自动发现并生成新技能 |
| 用户个性化 | 无 | 记住偏好，越用越懂用户 |

---

## 7. 与现有系统集成点

| 现有模块 | 集成方式 |
|---|---|
| `edict_graph.py` | 在 `taizi_node` 注入记忆召回；`monitor_node` 后触发反思 |
| `persistence_v2.py` | 扩展为记忆存储后端，添加记忆表 |
| `vector_store.py` | 用于长期记忆的语义检索 |
| `dept_agents.py` | 集成工具学习器与 Prompt 优化器 |
| `skills/` | 添加自动生成技能目录 |

---

*本方案完整实现代码见下一份交付物*
