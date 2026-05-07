# hermes-agentic-rl Readiness Audit（面向基于 hermes-agent 的 RL 强化学习优化）

日期：2026-05-07  
审查对象：`output/f36c32d5-d7f6-4dc5-ab6e-19ecb1ebdf47/hermes-agentic-rl/`  

## 1. 结论（One-line）

**当前版本可以作为“Hermes-Agent 语义的 RL 训练框架/脚手架”使用，但尚未满足“直接基于 NousResearch/hermes-agent 的真实 LLM agent 做端到端 RL 优化”的工程要求**；主要阻塞点是：**真实 runtime 的工具调用/记忆/对话生成接口仅做了 duck-typing 适配、训练 policy 仍是自研小网络而非 hermes-agent 使用的 LLM（HF/vLLM 后端与训练核心尚未打通）**。

> 简化理解：它更像“把 Hermes-Agent 抽象成 MDP 后，训练一个结构化策略网络”的框架；离“对 hermes-agent 的 LLM 做 RL 微调（PPO/GRPO/RCPO + LoRA）”还差关键连接件。

---

## 2. 准入清单（Pass / Blocker / Risk）

| 维度 | 结论 | 说明 |
|---|---|---|
| Hermes-Agent 桥接真实性 | **Risk / 部分 Pass** | 有 `hermes_rl/bridge/hermes_adapter.py`（duck-typing），但没有对 NousResearch/hermes-agent 的**明确依赖与接口契约测试**；tool_registry 的结构假设较强。 |
| MDP 定义完整性（S/A/T/R） | **Pass** | 状态/动作空间建模完整：`envs/hermes_env.py`、`mdp/state_encoder.py`、`mdp/action_space.py`。 |
| observation/state encoder 覆盖度 | **Pass（但偏“轻量模拟”）** | 覆盖 dialog/tools/memory/scratchpad；但 `tokenizer_lite.py` 为 hash tokenizer，非真实 LLM tokenizer。 |
| action space 表达力 | **Risk** | `HermesAction` 支持 tool/memory/delegate/respond；但 `param_tokens` 目前更像“离散 token 序列”，环境侧 decode 成简化字符串/伪参数，未形成真实 JSON 参数生成/校验闭环。 |
| 奖励与约束（含 Lagrangian） | **Pass（框架层面）** | `rewards/composer.py` 支持分层奖励；README/docs 声称有 Lagrangian/RCPO（需进一步接口核验）。 |
| 在线 RL（PPO/GRPO/IMPALA） | **Pass（框架存在）** | `algos/ppo.py`、distributed 模块存在；但 PPO 优化对象为 `HermesActorCritic`（自研策略网络），非 hermes-agent LLM。 |
| 离线训练（BC/AWR/DPO） | **Pass（框架存在）** | `offline/dpo.py` 等模块齐全，接口针对“因子化动作 logprob”。 |
| 评测 Harness / A-B | **Pass（基础版）** | `eval/harness.py` 存在；是否覆盖真实任务与工具效果仍需补充 suite。 |
| 部署/导出（HF/vLLM/LoRA） | **Risk** | `backends/hf_backend.py` / `vllm_backend.py` / `peft/lora.py` 存在，但与训练主策略网络未打通；部署文档偏“目标态”。 |
| 可复现性（config/seed/ckpt/manifest） | **Pass（基础）** | `scripts/train.py` + configs 体系存在；真实端到端（含 runtime 依赖）仍需补齐 lockfile/版本约束。 |

---

## 3. 关键证据（按模块）

### 3.1 环境与 MDP

- 环境：`hermes_rl/envs/hermes_env.py`  
  - 有 `HermesAgentEnv`（mock sandbox）与 `RealHermesAgentEnv`（使用 adapter 从 runtime 构建 sandbox/记忆）。  
- 动作空间：`hermes_rl/mdp/action_space.py`  
  - `HermesAction` 含 `action_type/tool_id/param_tokens/memory_op/memory_slot/is_terminal`。  
- 状态编码：`hermes_rl/mdp/state_encoder.py`  
  - `HierarchicalMemoryEncoder` 将 dialog/tools/memory/scratchpad 多路编码并 gated fuse。  
- tokenization：`hermes_rl/utils/tokenizer_lite.py`  
  - hash-based tokenizer（适合无网 smoke/CI），不等价于真实 LLM tokenizer。  

### 3.2 训练策略网络与算法

- 策略网络：`hermes_rl/mdp/policy_network.py`  
  - `HermesActorCritic`：encoder + 多 head（action/tool/param/mem_op/mem_slot）+ value head。  
  - **param_logits 由线性层一次性产生 `[B, max_param_len, vocab]`**，并非调用 HF/vLLM 的 generate/logits。  
- 在线 RL：`hermes_rl/algos/ppo.py`  
  - PPO/GAE/clip/KL 等框架齐全，但优化对象为上述 actor-critic。  
- 离线：`hermes_rl/offline/dpo.py`  
  - DPO 直接用动作级 logprob 训练。  

### 3.3 Hermes-Agent 真实桥接

- 适配器：`hermes_rl/bridge/hermes_adapter.py`  
  - 通过 duck-typing 假设 runtime 暴露 `tool_registry/memory`；并从 tool_registry 值构建 `ToolSandbox` 可调用对象。  
  - **缺少“针对 NousResearch/hermes-agent 的接口契约测试/示例 runtime shim”**：如果 hermes-agent 的 `tool_registry` 存的不是 `callable(**kwargs)`，则会无法直接工作或语义偏差。  

### 3.4 LLM 后端与部署（当前更像“目标态能力”）

- HF 后端：`hermes_rl/backends/hf_backend.py`（训练可 forward）  
- vLLM 后端：`hermes_rl/backends/vllm_backend.py`（仅 generate + sync weights）  
- LoRA：`hermes_rl/peft/lora.py`（自研注入/保存/加载/merge）  

这些能力**在代码层存在**，但目前训练主循环没有把“LLM backend logits/生成”作为策略的一部分来优化（仍是结构化策略网络）。

---

## 4. Blockers（必须修复项，P0）

### B1：缺少 hermes-agent 的“接口契约层 + 真实对接样例”

**问题**：`HermesAgentAdapter` 采用 duck-typing，无法保证与 NousResearch/hermes-agent 的 runtime 接口一致；tool_registry 的值类型/调用约定很可能不匹配。  

**建议**：
1. 增加一个 `RuntimeProtocol`（`typing.Protocol`）明确必须实现的方法与数据结构（tool schema、memory API、run_tool/generate_reply 等）。  
2. 提供 `HermesAgentRuntimeShim`：把真实 hermes-agent 的对象适配成 protocol（集中处理 tool 调用、权限、trace、memory side effects）。  
3. 增加契约测试：用最小 hermes-agent runtime（或 mock 但遵循真实 schema）跑通 `RealHermesAgentEnv.step()`。

### B2：训练对象不是 hermes-agent 的 LLM（无法称为“基于 hermes-agent 的 RL 微调”）

**问题**：当前 PPO/GRPO/DPO 优化的是 `HermesActorCritic`（自研小网络），而 hermes-agent 实际的决策/回复通常由 LLM 生成。  

**建议（最小可行路线）**：
1. 定义“LLM policy 形式”：例如把 Hermes-Agent 的输出规范化为结构化 action 的 JSON（tool call / respond / memory op），并以真实 tokenizer token 作为 action。  
2. 用 `HuggingFaceBackend` 提供 logits 与反向传播，配合 `peft/lora.py` 只训练 LoRA adapter。  
3. rollout 采样用 `VLLMBackend.generate`（高吞吐），并实现 `sync_weights_from_hf` 的稳定版本以保证 actor/learner 同步。  

### B3：param_tokens 的参数生成/校验闭环不完整

**问题**：环境侧将 `param_tokens` decode 成简化字符串/伪参数，不能保证生成的参数能通过真实 tool schema（JSON）校验。  

**建议**：
1. 引入 tool schema（JSONSchema / pydantic model）到 env：`tool_registry` 需要暴露 `name/args_schema`。  
2. 把 param generation 改为“结构化 JSON 生成 + schema validate + 自动修复（可选）”。  
3. reward 中引入 “schema-valid bonus / invalid penalty”，让 RL 有梯度方向。

---

## 5. 风险项与优化建议（P1/P2）

### P1：tokenizer_lite 与真实 tokenizer 差距导致 sim2real gap
- 现状：hash tokenizer 便于无网测试，但训练出的策略很难迁移到真实 LLM。  
- 建议：保持 lite 作为 smoke；新增 hf tokenizer path，并在 configs 中强制区分 `lite_smoke` vs `hf_train`。

### P1：tool_registry 语义缺失（工具描述、权限、side-effect）
- 建议：tool_registry 中统一包含：`name/description/args_schema/callable/permission_tags`，并在 state encoder 中编码 description（而非只编码 name 列表）。

### P2：评测 suite 的“真实性”
- 现状：harness 存在，但 suite 是否覆盖真实 Hermes Agent 任务（多轮、跨会话记忆、工具链）不明。  
- 建议：把你们现有 Hermes agent 的任务集（或 trace replay）接入 EvalHarness，形成 A/B 可复现数据。

---

## 6. 最小可复现实验（smoke）

> 以下命令来自该项目 README 的“CPU 也可跑”路径；用于验证训练闭环（不代表真实 hermes-agent 接入已完成）。

```bash
# 在 hermes-agentic-rl 目录内
python scripts/train.py --config configs/ppo_smoke.yaml
```

期望看到：
- 能启动 rollout，log 中出现 `ppo/loss`、`reward/total`、`success_rate` 等关键指标；
- 能在 `runs/` 或 `output/` 下生成日志/ckpt（按配置而定）。

---

## 7. 是否“已符合要求”？（最终判定）

如果你的要求是：**“先用一个可运行的 RL 框架，在 Hermes-Agent 语义的 MDP 上做策略优化（可先 mock，再逐步真实化）”**  
→ **可以用（Pass with Risks）**：框架/算法/奖励/分布式/离线训练的骨架基本齐全。

如果你的要求是：**“直接对 NousResearch/hermes-agent 的真实 LLM 决策与工具调用行为做 RL 微调（LoRA + PPO/RCPO），并可部署到线上”**  
→ **当前不满足（Blockers）**：需要补齐 **runtime 契约层**、**LLM backend 与训练闭环的真正接入**、以及 **tool 参数生成的 schema 校验闭环**。

