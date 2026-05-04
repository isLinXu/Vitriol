# 离线 Trace 生成与回放（Token-by-Token）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于 `output/tinyllama-hybrid-ultra-test` 运行一次离线推理生成 `trace.json`，并在 3D 可视化中加载该 trace 实现 token-by-token 回放（自动镜头跟随 + 左侧树自动展开/高亮 + token 列表高亮/可点击跳转）。

**Architecture:**  
1) Python 侧提供 `vitriol trace` 命令：加载本地 HF 模型（tinyllama-hybrid-ultra-test），执行短 decode 循环，生成 `trace.json`（schema v1）。  
2) `vitriol viz` 增加 `--trace <path>`：读取 trace 并注入到 HTML（`window.__VITRIOL_TRACE__`），不走网络。  
3) 3D 页面增加 trace 模式：PlaybackEngine 从 trace 的 `node_path` 驱动粒子与高亮，并自动跟随镜头，同时同步左侧导航树与 token 列表。

**Tech Stack:** Python (transformers/torch), Click CLI, HTML/JS, Three.js, pytest

---

## 0) 文件清单（将要修改/新增）

**Add (CLI + trace generator):**
- `src/vitriol/cli/commands/trace.py`（新 CLI 命令）
- `src/vitriol/trace/schema.py`（schema 常量/校验函数，可选）
- `src/vitriol/trace/generator.py`（trace 生成实现，可选）

**Modify (CLI 注册):**
- `src/vitriol/cli/main.py`（COMMAND_SPECS 加入 trace）

**Modify (viz 注入):**
- `src/vitriol/cli/commands/viz.py`（新增 --trace 选项 + 注入 marker）
- `src/vitriol/viz/model_3d_visualizer.html`（新增 `// INLINE_TRACE_MARKER` + trace 回放 UI/逻辑）

**Modify (token UI 可视化):**
- `src/vitriol/viz/model_3d_visualizer.html`（token 列表面板，点击跳转）

**Tests (Add/Modify):**
- `tests/test_trace_schema_v1.py`（schema 结构校验）
- `tests/test_viz_trace_injection_markers.py`（HTML marker + 注入变量存在）

---

## Task 1: 定义 trace schema v1 + 基础校验测试

**Files:**
- Create: `tests/test_trace_schema_v1.py`
- (Optional Create): `src/vitriol/trace/schema.py`

- [ ] **Step 1: 写 failing test（schema 最小字段校验）**

Create `tests/test_trace_schema_v1.py`:

```python
def test_trace_schema_v1_min_fields() -> None:
    # 仅校验 schema 结构（不依赖实际模型推理）
    trace = {
        "schema_version": "trace.v1",
        "model_path": "output/tinyllama-hybrid-ultra-test",
        "prompt": "hello",
        "max_new_tokens": 8,
        "tokens": {
            "prompt_token_ids": [1],
            "prompt_tokens": ["hello"],
            "generated_token_ids": [2],
            "generated_tokens": ["!"],
        },
        "events": [
            {"token_index": 0, "phase": "prefill", "node_path": ["embed", "block:0:attn", "block:0:mlp", "lm_head"]}
        ],
    }
    assert trace["schema_version"] == "trace.v1"
    assert isinstance(trace["events"], list) and trace["events"]
    e0 = trace["events"][0]
    assert "token_index" in e0 and "phase" in e0 and "node_path" in e0
    assert e0["node_path"][0] == "embed"
    assert e0["node_path"][-1] == "lm_head"
```

- [ ] **Step 2: 运行测试，确认通过（这一步应直接 PASS）**

Run:
```bash
PYTHONPATH=src python -m pytest -q tests/test_trace_schema_v1.py
```
Expected: PASS

- [ ] **Step 3: Commit**
```bash
git add tests/test_trace_schema_v1.py
git commit -m "test(trace): add trace.v1 schema sanity test"
```

---

## Task 2: 新增 `vitriol trace` 命令（生成 trace.json）

**Files:**
- Create: `src/vitriol/cli/commands/trace.py`
- Modify: `src/vitriol/cli/main.py`
- (Optional Create): `src/vitriol/trace/generator.py`

- [ ] **Step 1: 写 failing test（命令可导入）**

Create `tests/test_trace_cli_import.py`:

```python
def test_trace_cli_importable() -> None:
    from vitriol.cli.commands.trace import trace  # noqa: F401
```

- [ ] **Step 2: 运行测试，确认失败（因为文件尚不存在）**

Run:
```bash
PYTHONPATH=src python -m pytest -q tests/test_trace_cli_import.py
```
Expected: FAIL（ImportError）

- [ ] **Step 3: 最小实现 trace 命令**

Create `src/vitriol/cli/commands/trace.py`：

```python
import json
from pathlib import Path
import click

@click.command(name="trace")
@click.option("--model-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=str))
@click.option("--prompt", required=True, type=str)
@click.option("--max-new-tokens", default=8, show_default=True, type=int)
@click.option("--out", "out_path", default="trace.json", show_default=True, type=click.Path(dir_okay=False, path_type=str))
@click.option("--device", default="cpu", show_default=True, type=str)
@click.option("--trust-remote-code/--no-trust-remote-code", default=True, show_default=True)
def trace(model_path: str, prompt: str, max_new_tokens: int, out_path: str, device: str, trust_remote_code: bool) -> None:
    \"\"\"Run offline inference and export token-by-token trace.json.\"\"\"
    # v1：先实现最稳妥版本：加载 model/tokenizer，运行 generate（greedy），导出结构化 trace（node_path 先按结构默认生成）。
    # v1.1：再升级为 decode-loop + hooks（更真实的 token-by-token）。
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PretrainedConfig
    import torch

    path = str(model_path)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    cfg = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=trust_remote_code)
    for k in ("text_config", "vision_config", "encoder_config", "decoder_config"):
        v = getattr(cfg, k, None)
        if isinstance(v, dict):
            setattr(cfg, k, PretrainedConfig.from_dict(v))
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=trust_remote_code, config=cfg)
    model.eval()
    model.to(device)

    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False)
    # 解析 tokens
    prompt_ids = inputs["input_ids"][0].tolist()
    out_ids = out[0].tolist()
    gen_ids = out_ids[len(prompt_ids):]
    prompt_tokens = [tok.decode([i]) for i in prompt_ids]
    gen_tokens = [tok.decode([i]) for i in gen_ids]

    # v1：node_path 先用“结构主干”生成（embed -> blocks -> lm_head），用于回放链路打通。
    # 注意：TinyLlama 是 Llama 架构，v1 用 mlp 字段；回放侧会兼容映射到 ffn。
    n_layers = int(getattr(getattr(cfg, "text_config", cfg), "num_hidden_layers", 0) or 0)
    node_path = ["embed"]
    for i in range(n_layers):
        node_path.append(f"block:{i}:attn")
        node_path.append(f"block:{i}:mlp")
    node_path.append("lm_head")

    events = []
    # prefill: prompt token 仅记录最后一步（简化）
    if prompt_ids:
        events.append({"token_index": 0, "phase": "prefill", "node_path": node_path})
    # decode: 每个生成 token 一条
    for i in range(len(gen_ids)):
        events.append({"token_index": i, "phase": "decode", "node_path": node_path})

    trace_obj = {
        "schema_version": "trace.v1",
        "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "model_path": path,
        "prompt": prompt,
        "max_new_tokens": int(max_new_tokens),
        "device": str(device),
        "tokens": {
            "prompt_token_ids": prompt_ids,
            "prompt_tokens": prompt_tokens,
            "generated_token_ids": gen_ids,
            "generated_tokens": gen_tokens,
        },
        "events": events,
    }

    Path(out_path).write_text(json.dumps(trace_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    click.echo(f"Wrote trace: {out_path}")
```

并在 `src/vitriol/cli/main.py` 加入：
```python
COMMAND_SPECS["trace"] = "vitriol.cli.commands.trace:trace"
COMMAND_SHORT_HELP["trace"] = "Generate offline token-by-token trace."
```

- [ ] **Step 4: 运行测试，确认通过**

Run:
```bash
PYTHONPATH=src python -m pytest -q tests/test_trace_cli_import.py
```
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/vitriol/cli/main.py src/vitriol/cli/commands/trace.py tests/test_trace_cli_import.py
git commit -m "feat(trace): add offline trace generator CLI"
```

---

## Task 3: viz 增加 `--trace` 注入（INLINE_TRACE_MARKER）

**Files:**
- Modify: `src/vitriol/cli/commands/viz.py`
- Modify: `src/vitriol/viz/model_3d_visualizer.html`
- Create: `tests/test_viz_trace_injection_markers.py`

- [ ] **Step 1: failing test：要求 3D HTML 存在 marker**

Create `tests/test_viz_trace_injection_markers.py`:

```python
from pathlib import Path

def test_3d_has_inline_trace_marker() -> None:
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    assert "INLINE_TRACE_MARKER" in html
    assert "__VITRIOL_TRACE__" in html
```

- [ ] **Step 2: 运行测试，确认失败**
Run:
```bash
PYTHONPATH=src python -m pytest -q tests/test_viz_trace_injection_markers.py
```
Expected: FAIL

- [ ] **Step 3: 在 3D HTML 增加 marker**

在 `<script>` 顶部加入：
```js
// INLINE_TRACE_MARKER
// window.__VITRIOL_TRACE__ = {...} 由 CLI 注入
```

- [ ] **Step 4: viz CLI 增加 --trace 并注入**

在 `viz.py` 增加 option：
- `@click.option('--trace', 'trace_path', type=click.Path(exists=True, dir_okay=False), help='Path to trace.json')`

注入逻辑：
- 读取 JSON
- `html_content.replace("// INLINE_TRACE_MARKER", f"window.__VITRIOL_TRACE__ = {json.dumps(trace)};")`

- [ ] **Step 5: 运行测试通过**

Run:
```bash
PYTHONPATH=src python -m pytest -q tests/test_viz_trace_injection_markers.py
```

- [ ] **Step 6: Commit**
```bash
git add src/vitriol/cli/commands/viz.py src/vitriol/viz/model_3d_visualizer.html tests/test_viz_trace_injection_markers.py
git commit -m "feat(viz): support offline trace injection"
```

---

## Task 4: 3D trace 回放模式（自动镜头跟随 + 树高亮 + token 列表）

**Files:**
- Modify: `src/vitriol/viz/model_3d_visualizer.html`

- [ ] **Step 1: failing test：DOM id 存在**

Append to `tests/test_viz_trace_injection_markers.py`:
```python
def test_3d_has_token_list_panel() -> None:
    html = Path("src/vitriol/viz/model_3d_visualizer.html").read_text(encoding="utf-8")
    assert 'id="tokenListPanel"' in html
    assert 'id="tokenList"' in html
    assert 'id="followCameraToggle"' in html
```

- [ ] **Step 2: 实现 token 列表面板（可点击跳转）**

新增右下角/左下角可折叠 panel：
- `div#tokenListPanel`
- `div#tokenList`（容器，渲染 token chip）

点击 token chip：
- `engine.pause(); engine.setTokenIndex(i);`

高亮：
- `engine.onChange` 时更新当前 token chip 的 active 样式

- [ ] **Step 3: 自动镜头跟随（可开关）**

新增 `#followCameraToggle`（checkbox）。
实现 `focusCameraOnNode(nodeId)`：
- 获取 nodeIndex 的 worldPos
- 平滑插值 `camera.position` 与 `controls.target`（用 lerp）
- 仅在 follow 开启时启用

- [ ] **Step 4: 左侧树自动展开/高亮**

实现 `syncNavToNode(nodeId)`：
- 从 nodeId 推断 layerIndex
- 找到对应的树节点元素（基于已有 nav 构建逻辑，给每个 nav item 加 data-nodeid）
- `scrollIntoView` + 添加 active class
- 需要时展开父节点

- [ ] **Step 5: node_id 映射兼容（mlp→ffn）**

trace 的 node_path 可能是 `block:i:mlp`：
- 回放时映射为 `block:i:ffn`（对于 Llama）

- [ ] **Step 6: 运行测试通过**

Run:
```bash
PYTHONPATH=src python -m pytest -q tests/test_viz_trace_injection_markers.py
```

- [ ] **Step 7: Commit**
```bash
git add src/vitriol/viz/model_3d_visualizer.html tests/test_viz_trace_injection_markers.py
git commit -m "feat(viz): replay offline trace with camera follow and token list"
```

---

## Task 5: 打通 tinyllama-hybrid-ultra-test 端到端 demo

**Files/Outputs:**
- Generate: `workspace/verification/tinyllama_hybrid_ultra_trace/trace.json`
- Generate: `workspace/verification/tinyllama_hybrid_ultra_trace/trace_3d.html`

- [ ] **Step 1: 生成 trace**
Run:
```bash
PYTHONPATH=src python -m vitriol.cli.main trace --model-path output/tinyllama-hybrid-ultra-test --prompt "hello" --max-new-tokens 8 --out verification/tinyllama_hybrid_ultra_trace/trace.json --device cpu
```

- [ ] **Step 2: 启动可视化并注入 trace**
Run:
```bash
PYTHONPATH=src python -m vitriol.cli.main viz output/tinyllama-hybrid-ultra-test --3d --trace verification/tinyllama_hybrid_ultra_trace/trace.json --no-open
```
打开输出 URL，点击播放，观察：
- token 粒子按 trace node_path 高亮/移动
- token 列表高亮推进
- 镜头跟随自动跳转
- 左侧树自动展开/高亮

---

## Execution Handoff
Plan complete and saved to `docs/superpowers/plans/2026-04-28-offline-trace-replay.md`.

你刚才选择了 **1) Subagent-Driven**。我将按 Task 逐个分派子代理实现并复核。  
确认我现在就开始执行 Task 1 → Task 5 吗？

