# 大朝议 III · 优化交付物清单

> **交付日期**：2026-05-05  
> **针对项目**：`/Users/gatilin/PycharmProjects/dachaoyi3`  
> **交付目标**：在已有 8 轮优化基础上，识别"已建未用"问题并给出可落地的收敛路线

---

## 📦 交付物列表

### 1. 深度分析报告
**文件**：`DACHAOYI3_DEEP_REVIEW.md`（13.5 KB）

**核心发现**：
- ✅ 前 8 轮优化的安全/性能/可观测性增强已真正落地
- ⚠️ `llm_provider.py` / `vector_store.py` / `persistence_v2.py` 已建但主路径未切换
- 🔴 `main.py` 2105 行巨石，80+ 路由全在一个文件
- 🔴 `useImperialWS.ts` 32.8 KB，无法独立测试
- 🔴 根目录 11 份历史优化文档 + 6 个临时 `.js` 文件堆积
- 🔴 `backend/venv/` 物理存在于项目（虽在 `.gitignore`）
- 🔴 测试覆盖 64%，但核心文件 `edict_graph.py` 仅 39%

**给出的路线**：3 周 P0 收敛（接入抽象 + 拆巨石 + 归档文档）→ 3 周测试/清理 → 后续 P2/P3 可选增量。

---

### 2. 仓库清理与归档脚本
**文件**：`cleanup_and_archive.sh`（5 KB）

**功能**：
1. 将 11 份历史优化文档归档到 `docs/archive/`
2. 将 `test_coords_*.js` 归档到 `docs/archive/scratch/`
3. 删除可重现产物（`.coverage` / `htmlcov` / `.DS_Store` / `__pycache__`）
4. 生成 `docs/archive/INDEX.md` + `CHANGELOG.md` + `CONTRIBUTING.md` 占位
5. 保持 `README.md` / `Makefile` / `GITHUB_FIRST_PUSH_CHECKLIST.md` 在根目录

**使用**：
```bash
bash cleanup_and_archive.sh        # 预览（dryrun）
bash cleanup_and_archive.sh apply  # 真正执行
```

**预期效果**：
- 根目录 `.md` 文件从 13 个降至 3 个（README / CHANGELOG / CONTRIBUTING）
- 历史文档可查但不污染当前视线
- 新贡献者打开项目时能立即找到入口

---

### 3. P0 自动化修复脚本
**文件**：`apply_p0_fixes.sh`（7 KB）

**自动化任务**：
- [T1] 替换 `core_agents.py` / `dept_agents.py` 中的 `ChatOpenAI` → `get_llm`
- [T2] 在 `knowledge.py` 中添加 `vector_store` 导入（需手动实现语义搜索逻辑）
- [T3] 切换 `main.py` 从 `persistence` 到 `persistence_v2`
- [T4] 检测 `/health` 路由重复（提示手动删除）
- [T5] 提示初始化 git 仓库（可选，交互式）

**使用**：
```bash
bash apply_p0_fixes.sh        # 预览
bash apply_p0_fixes.sh apply  # 真正执行
```

**安全性**：
- 所有修改前自动备份为 `*.backup`
- dryrun 模式预览所有操作
- 可回滚：`mv backend/core_agents.py.backup backend/core_agents.py`

---

### 4. main.py 拆分重构示例
**文件**：`main_py_refactor_example.py`（8 KB）

**内容**：
- 完整的 `backend/api/` 目录结构设计
- 3 个示例模块（`health.py` / `decree.py` / `__init__.py`）
- 重构后的 `main.py`（< 300 行，只做组装）
- 迁移步骤清单（7 步，可复制粘贴）

**预期效果**：
- `main.py` 从 2105 行降至 **< 300 行**
- 每个业务域独立文件，单一职责
- 新增路由时不再污染主文件

---

### 5. 前端 WebSocket Hook 拆分示例
**文件**：`frontend_ws_refactor_example.py`（13 KB）

**内容**：
- 将 `useImperialWS.ts` 拆分为 6 个模块：
  - `types.ts` — 类型定义
  - `wsTransport.ts` — 连接/重连/心跳（纯网络层）
  - `wsReducer.ts` — 增量状态合并（纯函数，可测）
  - `wsAck.ts` — ACK 机制与离线队列
  - `wsRouter.ts` — 消息类型路由
  - `useImperialWS.ts` — 主 Hook（< 200 行）
- 每个模块的完整代码示例
- 单元测试示例（`wsReducer.test.ts`）

**预期效果**：
- 主 Hook 从 ~900 行降至 **< 200 行**
- 每个子模块可独立单测，覆盖率可达 90%+
- `wsTransport` / `wsAck` 可跨项目复用

---

### 6. 关键链路冒烟测试脚本
**文件**：`smoke_test.sh`（7 KB）

**测试内容**（8 项）：
1. 前置工具检查（`curl` / `jq`）
2. 后端健康检查（`/health`）
3. LLM Provider 可用性（`get_llm()` 能初始化）
4. `core_agents.py` 已切换到新 Provider
5. `main.py` 已切换到 `persistence_v2`
6. `/health` 路由无重复
7. 后端核心 API 可访问（`/api/decree` / `/api/knowledge/documents`）
8. 前端可访问（可选）
9. 后端关键测试用例（`test_edict_graph.py`）

**使用**：
```bash
# 在项目根目录执行（需后端已启动）
bash smoke_test.sh
```

**输出示例**：
```
✓ 后端健康检查通过 (HTTP 200)
✓ LLM Provider 可用: openai:qwen-plus
✓ core_agents.py 已切换到 llm_provider
⚠ main.py 仍在使用旧 persistence，未切换到 v2
✓ /health 路由无重复
✓ /api/decree 可访问 (HTTP 200)
```

---

## 🎯 如何使用这些交付物

### 第 1 天：理解现状
```bash
# 1. 阅读深度分析报告（15 分钟）
open DACHAOYI3_DEEP_REVIEW.md

# 2. 预览清理操作（2 分钟）
bash cleanup_and_archive.sh
```

### 第 2-3 天：仓库清理
```bash
# 3. 真正执行清理（不可逆，建议先提交现状）
cd /Users/gatilin/PycharmProjects/dachaoyi3
git init  # 如果还没初始化
git add .
git commit -m "chore: baseline before cleanup"

# 4. 执行清理
bash cleanup_and_archive.sh apply

# 5. 检查效果
ls -la  # 根目录应只剩 3 个 .md
ls docs/archive/  # 历史文档在这里
```

### 第 4-7 天：P0 修复
```bash
# 6. 预览 P0 修复
bash apply_p0_fixes.sh

# 7. 真正执行（会自动备份）
bash apply_p0_fixes.sh apply

# 8. 手动完成以下任务（脚本无法自动完成）：
#    - knowledge.py 中实现语义搜索（见深度报告 T2）
#    - main.py 删除重复 /health 路由（第 500 行附近）

# 9. 冒烟测试
cd backend && uvicorn main:app --reload &  # 后台启动
bash smoke_test.sh
```

### 第 8-14 天：拆分巨石
```bash
# 10. 参考重构示例
#     - main_py_refactor_example.py（后端）
#     - frontend_ws_refactor_example.py（前端）

# 11. 按示例分步执行（每完成一个模块就跑一次测试）
pytest tests/backend/test_main.py -v
npm test -- useImperialWS
```

---

## 📊 预期收益（量化）

| 指标 | 当前 | 3 周后 | 6 周后 |
|---|---|---|---|
| 根目录 `.md` 数量 | 13 | 3 | 3 |
| `main.py` 行数 | 2105 | 300 | 300 |
| `useImperialWS.ts` 行数 | ~900 | 200 | 200 |
| 后端测试覆盖 | 64% | 70% | 75% |
| LLM Provider 可切换 | ❌ | ✅ | ✅ |
| 知识库语义搜索 | ❌ | ✅ | ✅ |
| git 仓库 | ❌ | ✅ | ✅ + CI |

---

## ⚠️ 风险提示

1. **`apply_p0_fixes.sh` 使用 `sed` 自动替换代码**
   - 虽然有备份，但建议先在测试分支执行
   - 替换后务必手动检查 `core_agents.py` 第 27 行附近

2. **`knowledge.py` 的语义搜索需手动实现**
   - 脚本只能添加 `import`，不能自动改写 `search()` 方法
   - 参考 `vector_store.py` 的 API：`store.search(query, top_k=5)`

3. **`main.py` 拆分需逐步进行**
   - 不要一次性移动所有路由，容易出错
   - 建议按域名拆（先 health.py，再 decree.py，依次类推）
   - 每拆一个文件，跑一次 `pytest + smoke_test.sh`

4. **清理操作不可逆**
   - `cleanup_and_archive.sh apply` 会移动文件
   - 建议先提交 git 快照：`git commit -am "snapshot before cleanup"`

---

## 📞 后续支持

这些交付物都是**自解释**的（包含注释和使用说明），但如果遇到问题：

1. 脚本执行报错 → 先看脚本输出的提示信息
2. 不确定某步是否必要 → 参考深度报告的"风险与回归控制"章节
3. 想了解为什么这么做 → 参考深度报告的"未解决的真问题"章节

---

*本清单由 BoxAI 生成 · 基于完整源码分析与 8 轮历史优化审阅*
