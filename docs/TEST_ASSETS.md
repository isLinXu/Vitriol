# Vitriol 测试大资产（本地外部目录）使用说明

本仓库默认不再提交体积巨大的测试资产目录（例如 `tests/offload/`、`tests/offload_inference/`、`tests/output/`），以保证仓库可分发、CI 可快速离线运行。

如果你需要运行依赖这些大资产的测试，请将资产保存在**本地外部目录**，然后通过脚本把它们以**软链接**（默认）或**复制**方式挂载到 `tests/` 下。

---

## 1. 约定：环境变量

设置你的本地测试资产根目录：

```bash
export VITRIOL_TEST_ASSETS_DIR="/path/to/vitriol-test-assets"
```

该目录建议包含（全部或部分）以下子目录：

```text
Vitriol-models/
├── offload/
├── offload_inference/
└── output/
```

> 说明：你可以只提供其中一部分；脚本会只挂载存在的目录。

---

## 2. 准备资产（软链接默认，推荐）

### 2.1 创建软链接到 tests/

在仓库根目录执行：

```bash
python scripts/prepare_test_assets.py
```

它会尝试：
- `tests/offload  ->  $VITRIOL_TEST_ASSETS_DIR/offload`
- `tests/offload_inference  ->  $VITRIOL_TEST_ASSETS_DIR/offload_inference`
- `tests/output  ->  $VITRIOL_TEST_ASSETS_DIR/output`

### 2.2 强制复制（不推荐，体积大）

```bash
python scripts/prepare_test_assets.py --copy
```

---

## 3. 测试行为（缺失资产时）

默认 CI / 默认本地运行不要求这些大资产。缺失资产时：
- 依赖大资产的测试应当 **自动 skip**，并提示如何设置 `VITRIOL_TEST_ASSETS_DIR`。

运行默认离线测试套件：

```bash
pytest -m "not slow and not network" tests/ --ignore=tests/integration -v
```

---

## 4. 常见问题

### Q1：为什么不用 Git LFS？
可以用，但会提高贡献者门槛、拉取体验也会变复杂。此仓库采用“本地外部目录 + 可选挂载”的方式，保持默认路径简单可用。

### Q2：软链接在 Windows 上不工作怎么办？
Windows 创建软链接可能需要管理员权限。你可以改用 `--copy`，或者用管理员权限运行。
