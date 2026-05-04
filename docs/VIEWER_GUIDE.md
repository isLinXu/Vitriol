# 🎨 Vitriol 3D Viewer 使用指南

一份完整的 Vitriol 模型可视化工具使用说明，涵盖单视图、架构对比、双模 3D 对比三个视图的全部功能。

## 📋 目录

- [页面总览](#页面总览)
- [核心功能](#核心功能)
- [URL 参数](#url-参数)
- [支持的模型架构](#支持的模型架构)
- [快捷键](#快捷键)
- [高级用法](#高级用法)
- [常见问题](#常见问题)

---

## 📄 页面总览

| 页面 | 路径 | 用途 |
|------|------|------|
| 🏠 主页 | `index.html` | 项目介绍与导航中枢 |
| 🎨 3D Viewer | `viewer.html` | 单模型精确 3D 可视化 |
| 🧬 架构对比 | `arch-compare.html` | 9 种模型特性矩阵（支持 MD/CSV 导出） |
| ⚖️ 双模 3D 对比 | `compare-3d.html` | 两个模型并排 3D 对比 |
| 🌳 进化树 | `evolution-tree.html` | Vitriol 模型进化历史 |
| 📅 时间线 | `innovation-timeline.html` | 创新节点时间线 |

---

## ⚡ 核心功能

### 1. 精确架构可视化

根据模型 config.json 的 `model_type` 自动选择专用渲染器：

| 模型类型 | 渲染函数 | 精确特性 |
|---------|---------|---------|
| LLaMA 3.x | `buildDenseLayer` | GQA + SwiGLU |
| Mistral / Mixtral | `buildMistralLayer` | GQA + SWA (+ MoE) |
| Qwen 3.5 / 3.6 | `buildQwen35Layer` | Linear + Full Attn 混合 + 多模态 |
| Gemma-2 | `buildGemma2Layer` | 交替 SWA/Global + Pre/Post Norm + GeGLU |
| Phi-3 | `buildPhi3Layer` | Fused QKV + gate_up + su_scaled_rope |
| GLM-4 | `buildGLM4Layer` | GLMBlock: Pre/Post Norm + h_to_4h |
| DeepSeek-V4 | `buildDeepSeekV4Layer` | MLA + MoE + Hash Attn |
| MiMo V2.5 | `buildMiMoLayer` | GQA + fine-grained MoE |
| 其他 | `buildDenseLayer` | 通用 Transformer 回退 |

### 2. 参数数量精确获取

**优先级**：
1. HuggingFace API `safetensors.parameters`（最精确，实际权重统计）
2. 从 config.json 计算（MLA/MoE 考虑）
3. 默认值

### 3. 推理演示动画

点击底部 `▶️ Inference` 按钮，查看每层的推理过程：

- 🎨 按层类型着色：embedding=蓝，attn=青，MoE=琥珀，output=红
- 📊 实时显示 KV cache 大小：`💾 KV=X MB/layer @ 2K ctx`
- 📈 进度百分比 + 模块数
- ⚡ MoE 层显示激活专家比例
- 🔤 Embedding 层显示向量维度
- 🎯 Output 层显示 Softmax → Next Token

### 4. 架构徽章自动识别

统计面板会自动显示架构徽章：

- 🔵 **LLaMA GQA** - LLaMA 系列
- 🔵 **Hybrid Attn** - Qwen 3.x 混合注意力
- 🟣 **GQA + SWA** - Mistral
- 🟣 **SWA + MoE** - Mixtral
- 🟢 **Gemma-2** - Gemma 系列
- 🟣 **Phi-3 Fused** - Phi-3 系列
- 🟠 **GLMBlock** - GLM 系列
- 🟡 **MLA + MoE** - DeepSeek-V4
- 🟡 **GQA + MoE** - MiMo

### 5. 模型无刷新切换 ⭐ 新

使用右上角模型选择器切换模型时：
- ✅ 不再刷新整个页面
- ✅ 自动清理旧场景（防止内存泄漏）
- ✅ URL hash 自动更新（可分享）
- ✅ 推理演示自动停止（切换时）

---

## 🔗 URL 参数

### 基础参数

```
viewer.html#?hf=<owner>/<repo>       加载 HuggingFace 模型
viewer.html#?model=data/<local-dir>  加载本地模型
viewer.html                          使用默认演示模型
```

### 高级参数

```
viewer.html#?hf=Qwen/Qwen3.6-27B&compact=1   紧凑模式（隐藏UI，用于嵌入）
```

`compact=1` 模式用于 `compare-3d.html` 中的 iframe 嵌入。

### 示例 URL

| 用途 | URL |
|------|-----|
| 查看 Qwen3.6-27B | `viewer.html#?hf=Qwen/Qwen3.6-27B` |
| 查看 DeepSeek-V4-Pro | `viewer.html#?hf=deepseek-ai/DeepSeek-V4-Pro` |
| 查看 Mixtral 8x7B | `viewer.html#?hf=mistralai/Mixtral-8x7B-v0.1` |
| 查看 Gemma-2 9B | `viewer.html#?hf=google/gemma-2-9b` |
| 查看 Phi-3 Mini | `viewer.html#?hf=microsoft/Phi-3-mini-4k-instruct` |
| 查看 GLM-4 9B | `viewer.html#?hf=THUDM/glm-4-9b` |

---

## 🎯 支持的模型架构

### 11 种精确支持的模型

| # | 模型 | 机构 | 架构特性 |
|---|------|-----|---------|
| 1 | LLaMA 3.1 8B | Meta | GQA (32Q/8KV) + SwiGLU + RoPE |
| 2 | Mistral 7B | Mistral AI | GQA (32Q/8KV) + SWA (4K window) |
| 3 | Mixtral 8x7B | Mistral AI | GQA + SWA + MoE (8 experts, top-2) |
| 4 | Qwen 2.5 7B | Alibaba | GQA (28Q/4KV) + RoPE extended |
| 5 | **Qwen 3.6-27B** | Alibaba | **Linear + Full Attn 混合 + 多模态 (VL 27L)** |
| 6 | **Gemma-2 9B** | Google | **交替 SWA/Global + Pre/Post Norm + GeGLU + Softcap** |
| 7 | **Phi-3 Mini** | Microsoft | **Fused QKV + gate_up + longrope** |
| 8 | **GLM-4 9B** | THUDM | **GLMBlock: dense_h_to_4h (×2) + Pre/Post Norm** |
| 9 | DeepSeek-V4-Pro | DeepSeek | MLA + 256 experts MoE + Hash Attn |
| 10 | MiMo V2.5-Pro | Xiaomi | GQA + 384 experts fine-grained MoE |
| 11 | 通用 Transformer | - | GQA Dense 回退方案 |

---

## ⌨️ 快捷键

| 按键 | 功能 |
|-----|------|
| `R` | 重置相机 |
| `F` | 聚焦选中模块 |
| `H` | 切换标签显示 |
| `Esc` | 关闭详情面板 |

---

## 🔧 高级用法

### 1. 嵌入到自己的页面

```html
<iframe src="https://islinxu.github.io/Vitriol/viewer.html#?hf=Qwen/Qwen3.6-27B&compact=1"
        width="100%" height="600" frameborder="0"></iframe>
```

### 2. 程序化切换模型（iframe）

```javascript
// 假设 iframe 已加载
const frame = document.getElementById('myViewer');
frame.contentWindow.loadModelPreset('deepseek-ai/DeepSeek-V4-Pro');
```

### 3. 双模对比同步镜头

在 `compare-3d.html` 中点击 🔗 **同步视角** 按钮：
- 读取左侧镜头位置/旋转
- 应用到右侧
- 通过 `window.camera` 和 `window.controls` 跨 iframe 访问

### 4. 导出架构对比报告

在 `arch-compare.html` 点击：
- **📝 导出 Markdown** → `vitriol-arch-compare.md`
- **📊 导出 CSV** → `vitriol-arch-compare.csv`

生成的 Markdown 可直接粘贴到 README、Wiki 或论文附录。

### 5. 导出当前模型截图

在 viewer 中：
- **📷 PNG** - 保存当前视角截图
- **📦 HTML** - 导出独立 HTML（包含所有数据）

---

## 🐛 常见问题

### Q1: 为什么参数数量和 HF 网页显示不一致？

**A**: Vitriol 优先使用 `safetensors.parameters`（真实权重统计）。如果 HF 网页显示的是 "Model size" (含 bias/norm)，可能会有微小差异。

### Q2: 加载超大模型（如 DeepSeek-V4 862B）卡顿？

**A**: 这是 Three.js 对超大模型的性能限制。建议：
- 使用 Chrome/Edge（WebGL 2 支持更好）
- 关闭其他占用 GPU 的标签页
- 未来版本将引入 LOD（细节层次）优化

### Q3: iframe 嵌入的 viewer 不显示 UI？

**A**: 这是 `compact=1` 模式预期行为，用于节省空间。如需完整 UI，请去掉该参数。

### Q4: 模型切换后推理演示不工作？

**A**: 已修复 - 切换模型时会自动停止推理，需手动重新点击 `▶️ Inference`。

### Q5: 如何添加自定义模型？

**A**: 在 `data/` 目录下创建子目录，放入 `config.json`，然后：
```
viewer.html#?model=data/my-custom-model
```

---

## 📚 相关资源

- **GitHub 仓库**: https://github.com/isLinXu/Vitriol
- **GitHub Pages**: https://islinxu.github.io/Vitriol/
- **文档**: 项目 README.md / README_CN.md
- **贡献指南**: CONTRIBUTING.md

---

## 📝 更新日志

### 2026-05-04
- ✨ 新增 **模型无刷新切换**（loadModelPreset 改造）
- ✨ 新增 **双模 3D 并排对比页**（compare-3d.html）
- ✨ 新增 **compact 模式** URL 参数
- ✨ 新增 **Phi-3 / GLM-4 / Gemma-2 专用可视化**
- ✨ 新增 **架构对比 Markdown/CSV 导出**
- ✨ 新增 **KV Cache 实时计算显示**
- 📝 支持从 **3 → 11** 种精确架构

---

*Vitriol · LLM Quantization & NAS Framework · MIT License*
