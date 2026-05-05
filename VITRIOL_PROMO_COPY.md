# 🎉 Vitriol 推广文案合集

> 三版本：**小红书长图文** / **推特单推** / **推特线程** / **微博短文**  
> 项目主页：https://islinxu.github.io/Vitriol/  
> GitHub：https://github.com/isLinXu/Vitriol

---

## 📕 版本一：小红书长图文（适合配 9 图）

### 标题候选（A/B 测试用）

- 🔥 **我做了个 LLM 架构 3D 可视化神器！一键看懂 DeepSeek/Qwen/Gemma 内部结构**
- 🔥 **用 KB 级配置文件探索 TB 级大模型！这个开源项目太绝了**
- 🔥 **吃透 11 种主流大模型架构！3D 可视化 + 哈希指纹 + 在线对比**

### 正文

```
家人们谁懂啊！做 LLM 相关工作一直苦于看不清模型内部结构 🙃

直到我做了个开源工具 👉 Vitriol 🧪

✨ 能干嘛：
一个网页就能 3D 可视化 11 种主流大模型！
— LLaMA 3.x / Mistral / Mixtral
— Qwen 2.5 / Qwen 3.5 / Qwen 3.6（混合注意力 + 多模态）
— Gemma-2 / Phi-3 / GLM-4
— DeepSeek-V4-Pro（862B！）
— 小米 MiMo V2.5

🎯 超带感的 5 个功能：

1️⃣ HuggingFace 一键加载
输入仓库名就能看 → 参数从 HF API 实时读取，精确到万位

2️⃣ 11 种专用渲染器
GQA / SWA / MLA / MoE / GeGLU / Fused QKV 全都精准呈现
不是一刀切的"通用 Transformer"！

3️⃣ 推理演示动画 🌊
点击▶️按钮，看着粒子流从 Token Embedding 一路流过各层
KV cache 大小实时计算，MoE 激活专家比例可视化

4️⃣ 双模 3D 并排对比
左右两个 3D 视图，切换同步相机
一眼看懂 LLaMA 和 Mixtral 的区别

5️⃣ 模型哈希指纹 🔐
浏览器内 SHA-256 秒算
生成模型唯一"身份证"，写论文复现神器！

🎁 还有：
· Benchmark 排行榜（MMLU/GSM8K/HumanEval/MATH/HellaSwag）
· Markdown/CSV 一键导出
· 中英双语 + 移动端友好
· 大模型 LOD 优化（64 层跑 60fps！）

📊 技术栈：
Python 57K 行 + 134 个测试 + Three.js/D3.js

💻 地址（戳我戳我）：
https://islinxu.github.io/Vitriol/

#大模型 #LLM #深度学习 #开源项目 #AI工具
#程序员日常 #机器学习 #模型压缩 #Transformer
#DeepSeek #Qwen #HuggingFace
```

### 配图建议（9 宫格）

1. 封面：viewer.html DeepSeek-V4 3D 截图 + 标题字样
2. Qwen3.6-27B 混合注意力结构图
3. Mixtral 8x7B 的 MoE 专家结构
4. 推理演示粒子动画 GIF
5. 双模 3D 并排对比截图
6. Benchmark 排行榜表格
7. 哈希指纹 Modal 截图
8. 架构对比矩阵表格
9. GitHub 仓库 star 数截图

---

## 🐦 版本二：推特单推（280 字符内）

### 英文版

```
🧪 Just launched Vitriol — an open-source 3D visualizer for LLM architectures.

✨ 11 models: LLaMA · Mistral · Mixtral · Qwen 3.6 · Gemma 2 · Phi-3 · GLM-4 · DeepSeek V4 (862B!)
🌊 Inference animation w/ particle flow
🔐 Online SHA-256 fingerprint
⚖️ Side-by-side 3D compare

Try it → islinxu.github.io/Vitriol
```

### 中文版

```
🧪 开源了一个 LLM 架构 3D 可视化工具：Vitriol

✨ 支持 11 种主流大模型
🌊 推理演示 + 粒子动画  
🔐 在线哈希指纹（SHA-256）
⚖️ 双模 3D 并排对比
📊 Benchmark 排行榜 

HuggingFace 任意模型一键加载

🔗 islinxu.github.io/Vitriol
⭐ github.com/isLinXu/Vitriol
```

---

## 🧵 版本三：推特线程（9 推）

### Tweet 1/9 - 开场

```
🧪 Open-sourced Vitriol today — a 3D visualizer that lets you explore LLM internals right in your browser.

It covers 11 major architectures including the 862B DeepSeek-V4-Pro.

No setup. Paste a HuggingFace repo name → see the model. 

Thread 🧵👇

https://islinxu.github.io/Vitriol/
```

### Tweet 2/9 - 问题

```
The problem:

Every time a new LLM drops (MLA, Hybrid Attention, SWA+MoE...), I had to re-read the paper, diff config.json, draw whiteboard arrows just to understand what changed.

There's no unified way to visually compare model architectures.

Until now.
```

### Tweet 3/9 - 11种架构

```
Vitriol renders 11 architectures with *dedicated* layer builders:

🔹 LLaMA 3.x — GQA + SwiGLU
🔹 Mistral — GQA + SWA
🔹 Mixtral — SWA + 8-expert MoE
🔹 Qwen 3.6 — Hybrid Linear/Full Attn + Vision
🔹 Gemma-2 — alternating SWA + GeGLU + softcap
🔹 Phi-3 — Fused QKV
🔹 GLM-4 — GLMBlock
🔹 DeepSeek-V4 — MLA + 256-expert MoE
🔹 MiMo — 384-expert fine-grained MoE
```

### Tweet 4/9 - 关键技术1

```
Unlike generic "Transformer block" renderers, each build*Layer() function reflects real architecture details:

• Mistral SWA window is color-coded (pink)
• Qwen3.6 Linear Attn has separate K/V head dims
• Gemma-2 shows Pre AND Post norm around both Attn and FFN
• MLA's q_lora_rank path
```

### Tweet 5/9 - 推理演示

```
🌊 Inference Demo feature

Click ▶️ → particles flow from Token Embedding through every layer.

Per-stage metrics displayed live:
• KV cache size (MB/layer)
• MoE expert activation ratio
• Stage name + icon per layer type

Makes "how does inference actually happen" visceral.
```

### Tweet 6/9 - 哈希指纹

```
🔐 Online Model Fingerprint

Browser SHA-256 (Web Crypto API) computes:
• architecture_sha256
• config_sha256
• parameters_sha256
• files_manifest_sha256

Then a 32-char "Vitriol Fingerprint" = unique model ID.

Great for paper reproducibility 📝
```

### Tweet 7/9 - 双模对比

```
⚖️ Side-by-side 3D compare

Two iframes, one camera controller.

Click "Sync View" → both 3D models rotate together.

Ever wanted to compare LLaMA 3 vs Qwen 3 visually? Now you can.

+ A static feature matrix with exportable Markdown/CSV.
```

### Tweet 8/9 - 性能

```
⚡ Performance work:

• LOD rendering for 64+ layer models (HEAD+MIDDLE+TAIL)
• Reloadless model switching (history.replaceState + clearScene)
• Proper Three.js resource disposal
• Mobile responsive (768px / 480px)
• i18n (?lang=en)

Runs at 60fps on DeepSeek-V4-Pro (862B).
```

### Tweet 9/9 - CTA

```
🚀 Try it now:

🌐 https://islinxu.github.io/Vitriol/
⭐ https://github.com/isLinXu/Vitriol

Tech stack: Python (57k LOC, 55% test density) + Three.js + D3.js.
MIT License.

If you work with LLMs — quantization, NAS, architecture research — this might save you hours.

RT appreciated 🙏
```

---

## 📢 版本四：微博短文（≤280 字）

```
🧪 开源了一个工具：Vitriol

能在浏览器里 3D 可视化 11 种主流大模型的架构：
LLaMA / Mistral / Mixtral / Qwen 2.5/3.5/3.6 / Gemma-2 / Phi-3 / GLM-4 / DeepSeek-V4(862B) / MiMo

✨ 特色：
· 点一下就能看，HF 任意模型
· 推理演示 + 粒子动画 🌊
· 在线 SHA-256 模型指纹 🔐
· 双模并排 3D 对比 ⚖️
· Benchmark 排行榜 📊

🔗 https://islinxu.github.io/Vitriol

#LLM #大模型 #开源
```

---

## 💼 版本五：LinkedIn / 掘金技术帖

### 标题
**Vitriol: Explore TB-scale LLMs with KB-scale configs — An Open-Source 3D Visualizer for 11 LLM Architectures**

### 开篇

```
作为一名长期从事 LLM 量化、压缩与架构研究的工程师，我深受一个痛点困扰：

每次新模型发布（DeepSeek MLA、Qwen 混合注意力、Mixtral MoE...），
都需要花大量时间阅读源码才能理解架构细节。

于是，我做了 Vitriol —— 一个开源的 3D 模型架构可视化平台。
```

### 核心价值

```
🎯 三大核心能力：

1. **精确到模块级的 3D 可视化**
   - 不是通用 Transformer 的 "黑盒"
   - 每种架构专属的 build*Layer 渲染器
   - 11 种主流架构全覆盖

2. **浏览器内在线哈希指纹**
   - Web Crypto API 原生 SHA-256
   - 5 层哈希摘要（架构/配置/参数/文件清单/commit）
   - 生成 32 字符 "Vitriol Fingerprint" 作为模型唯一 ID
   - 写论文复现、模型审计利器

3. **推理过程动画演示**
   - 粒子流动显示层间数据传递
   - KV cache 大小实时计算
   - MoE 激活专家比例可视化
```

### 技术指标

```
📊 工程质量：
• 57,908 行 Python 源码
• 32,260 行测试代码（55.7% 密度）
• 178 个模块文件
• 134 个测试文件
• 19 个 CLI 命令
• 245 个核心测试通过率 100%

🌐 前端：
• Three.js + D3.js
• 7 个 HTML 页面
• 中英双语 i18n
• 移动端响应式（768px / 480px）
• 大模型 LOD 优化（64层 60fps）
```

### 尾部 CTA

```
🚀 立即体验：
• 项目主页：https://islinxu.github.io/Vitriol/
• GitHub：https://github.com/isLinXu/Vitriol
• 架构对比：https://islinxu.github.io/Vitriol/arch-compare.html

📖 使用指南：
https://github.com/isLinXu/Vitriol/blob/main/docs/VIEWER_GUIDE.md

欢迎 ⭐ Star 支持，Issue 与 PR 都欢迎！

#LLM #模型压缩 #架构可视化 #开源
```

---

## 🎬 视频/GIF 脚本建议（15 秒短视频）

### 结构（每秒一个画面）

- 0-2s：打开 viewer.html，Qwen3.6-27B 加载
- 2-4s：镜头旋转展示多层 3D 结构
- 4-6s：点击 ▶️ Inference，粒子流动起来
- 6-8s：切换到 DeepSeek-V4（862B，带 LOD 提示）
- 8-10s：点击 🔐 Hash，显示 SHA-256 指纹
- 10-12s：切换到双模对比页，左右同步
- 12-15s：Logo + GitHub 链接 + "⭐ Star 一下吧"

### 配乐建议
电子/科技风 BGM，BPM 120-130

### 字幕关键词
"11 models · 3D · Browser only · Open source"

---

## 📝 落地节奏建议

| 平台 | 发布时间 | 内容版本 | 配图 |
|-----|---------|---------|------|
| 小红书 | 周末 18:00 | 版本一 | 9 张截图 |
| 推特英文 | 周二 09:00 EST | 版本三（9推线程） | 每推配 1 GIF |
| 推特中文 | 同步英文 | 版本二 | 1 张封面 |
| 微博 | 工作日 12:30 | 版本四 | 3 张截图 |
| LinkedIn | 工作日 08:00 | 版本五 | 1 张架构图 |
| 掘金/思否 | 周四 14:00 | 版本五改编 | 完整图文 |

---

## 🏷️ 万能 Tag 池

### 英文
`#LLM #OpenSource #MachineLearning #DeepLearning #HuggingFace #Transformer #ModelCompression #3DVisualization #DataScience #AIResearch #Quantization #NAS #PyTorch #ThreeJS`

### 中文
`#大模型 #开源 #深度学习 #机器学习 #LLM #AI工具 #程序员 #可视化 #模型压缩 #HuggingFace #DeepSeek #Qwen #架构设计 #Transformer`

---

*文案由 Vitriol 项目组织整理 · 可根据实际发布平台微调*
