# ExoBrain 架构修正：从"零权重空壳"到"异构认知对齐"

> 修正日期：2026-04-20
> 版本：v0.4.0-alpha

---

## 一、原始设定的核心缺陷

### 1.1 零权重空壳的数学不可行性

**问题**：Ultra Strategy 导出的"零权重空壳模型"在数学上无法完成有效的 Attention 检索。

```python
# 零权重下的 Shell 模型前向传播：
query = shell_model(input_ids)  # = zeros + LayerNorm(zeros) = zeros
attention_scores = query @ external_key.T  # = 0 @ anything = 0
attention_weights = softmax(0)  # = uniform distribution
output = attention_weights @ external_value  # = uniform average (无意义)
```

**结论**：零权重壳产生的 Query 是零向量或噪声，与外部 KV 的注意力机制产生的是均匀分布，毫无信息量。

### 1.2 原始 ExoBrain 文档中的错误描述

```
"Ultra strategy exports a 'shell model' — complete architecture but
zero (or near-zero) weights."
```

**问题**：这种描述暗示"零权重 = 可行"，但实际上缺少了关键的**认知对齐 (Cognitive Alignment)** 组件。

---

## 二、修正后的架构

### 2.1 核心概念：异构推理 (Heterogeneous Reasoning)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ExoBrain v0.4 — 异构认知对齐架构                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│    壳模型 (0.1B 真实权重)           外脑 (7B 教师 KV 缓存)            │
│    ┌─────────────────┐           ┌──────────────────────┐          │
│    │  Embedding      │           │  Teacher Model KV     │          │
│    │  Layer (有权重)  │           │  ┌────┐ ┌────┐ ...   │          │
│    ├─────────────────┤           │  │L0  │ │L1  │        │          │
│    │  Layer 0  ~~~~~│~~attention~~│→ │KV  │ │KV  │        │          │
│    │  Layer 1  ~~~~~│~~attention~~│→ │    │ │    │        │          │
│    │  ...            │           │  └────┘ └────┘        │          │
│    │  Layer N  ~~~~~│~~attention~~│→                      │          │
│    ├─────────────────┤           └──────────────────────┘          │
│    │  🔑 ShellProjection │                                          │
│    │  (薄层可学习投影)   │                                          │
│    ├─────────────────┤                                             │
│    │  LM Head (有权重) │                                            │
│    └─────────────────┘                                             │
│              │                                                      │
│              ▼                                                      │
│    ┌──────────────────────────────────────────────┐                 │
│    │  Query 生成能力测试 (Demo 核心)                │                 │
│    │  "Shell 的 Query 能否精准击中外部 KV?"         │                 │
│    └──────────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 三大修正

| 原始设定 | 修正建议 | 理由 |
|---------|---------|------|
| 零权重空壳 | **轻量底座 (0.1B 真实权重)** | 必须具备基础的"提问能力"（生成有意义的 Query） |
| 全量 KV 注入 | **关键层 (Key-layers) 注入** | 减少 IO 压力，只在中间逻辑层进行知识干预 |
| 简单的截断对齐 | **线性投影对齐 (Learned Projection)** | 必须有一层极薄的可学习参数，将壳模型的 hidden_dim 投影到脑模型的空间 |

---

## 三、ShellProjection 模块设计

### 3.1 功能

将壳模型的隐藏空间投影到外脑的隐藏空间，使 Query 能够语义对齐。

```python
class ShellProjection(torch.nn.Module):
    """
    薄层特征对齐投影器。

    将壳模型的 hidden_dim → 外脑的 hidden_dim（双向）
    仅包含 1-2 层极薄线性层，参数量 << 壳模型本身。
    """

    def __init__(
        self,
        shell_hidden_dim: int,   # 壳模型隐藏维度
        brain_hidden_dim: int,    # 外脑隐藏维度
        mode: str = "linear",    # "linear" | "mlp" | "linear + layernorm"
        dropout: float = 0.1,
    ):
        ...
```

### 3.2 训练目标

学习目标是让 Shell 的 Query 能够：
1. **语义相似**：与外脑 KV 中的相关键产生高注意力分数
2. **精确检索**：top-k 检索结果与真实需求高度相关
3. **可逆性**：投影后可还原，保留壳模型自身的推理能力

---

## 四、关键层 (Key-Layer) 注入策略

### 4.1 不是所有层都需要外脑

```
层分布示意 (24层模型):
┌────────────────────────────────────────────────────┐
│ L0-L2:  表层编码（词法、语法）— 壳模型自有能力        │
│ L3-L8:  中层语义（概念、实体）— 🔑 关键层，注入外脑   │
│ L9-L14: 高层推理（逻辑、常识）— 🔑 关键层，注入外脑   │
│ L15-L20: 深层归纳（抽象、泛化）— 部分注入外脑        │
│ L21-L23: 输出映射（解码、生成）— 壳模型自有能力       │
└────────────────────────────────────────────────────┘
```

### 4.2 配置接口

```python
@dataclass
class ExoBrainConfig:
    # 原始：active_layers = [] 表示所有层
    # 修正：明确指定关键层
    key_layers: List[int] = field(default_factory=lambda: [3, 4, 5, 6, 7, 8,
                                                           9, 10, 11, 12, 13, 14])

    # 每层注入的 KV 数量
    kv_injection_top_k: int = 5

    # 注入强度（用于 residual/gated 模式）
    injection_strength: float = 1.0
```

---

## 五、落地场景评估

### 5.1 极具竞争力的场景

| 场景 | 描述 | ExoBrain 优势 |
|------|------|--------------|
| **即时知识更新** | 今天的重大新闻，无需微调，直接注入 KV | 秒级更新，全模型生效 |
| **垂直领域"脑插槽"** | 医疗/法律/代码专用 KV 模块，像游戏卡带一样切换 | 零微调成本，即插即用 |
| **超长文本推理** | 海量文档预处理成 KV 索引，模型推理到哪，ExoBrain 异步喂送 | IO 压力分散，显存利用率高 |

### 5.2 当前版本的 Demo 重点

**核心验证目标**：
> "壳模型的 Query 能否精准击中外部 KV？"

这不依赖于"零权重"，而是依赖于：
1. Shell 模型本身有真实的语言理解能力（能生成有意义的 Query）
2. ShellProjection 实现语义空间对齐
3. Attention 机制正确工作

---

## 六、下一步行动

### Phase 1: 修正代码架构 ✅
- [x] 修正 `exobrain.py` 文档（零权重 → 异构认知对齐）
- [x] 实现 `ShellProjection` 模块
- [x] 添加 `key_layers` 配置支持
- [x] 更新 `exobrain_inference.py` 文档

### Phase 2: Demo 聚焦 Query-KV Hit ✅
- [x] 编写 `exobrain_query_hit_demo.py`
- [x] 可视化 Query 与 KV 的注意力分布矩阵 (ASCII heatmap + similarity matrix)
- [x] 计算 Hit@K 指标

### Phase 3: 训练 Feature Alignment ✅
- [x] 实现对齐训练流程 (`exobrain_alignment_train.py`)
- [x] Cosine Alignment Loss (Hit@3/5: 0.250 → 0.750)
- [x] **Contrastive Alignment (InfoNCE) 修复** (2026-04-20)
  - 修复 self-contrastive 对角线问题（fill_diagonal_ → mask with -inf）
  - 保持梯度流（只 detach target，不 detach query）
  - Hit@3: 0.250 → 0.750, Hit@5: 0.250 → 1.000
- [x] 在小数据集上验证投影效果

### Phase 4: 真实模型集成 ✅ (2026-04-20)
- [x] 创建 `exobrain_real_model_test.py`
- [x] 使用 Qwen2.5-0.5B tokenizer 验证架构
- [x] ShellProjection 参数量：803.7K（896 → 896 维度）
- [x] **类别检索准确率：100%**（4/4 测试通过）
- [x] 端到端 ExoBrainInferencePipeline 结构验证
- [x] Simulated inference 运行成功

---

## 七、总结

**原始"零权重"方案在数学上不可行**——没有权重的模型无法生成有意义的 Query。

**修正后的"异构认知对齐"方案**：
- 壳模型：0.1B 真实权重的轻量模型（保留基础语言能力）
- ShellProjection：极薄对齐层（~0.001B 参数）
- 外脑：7B 模型 KV 缓存

**这才是真正意义上的"借脑生子"**——不是空壳嫁接，而是认知接口对接。
