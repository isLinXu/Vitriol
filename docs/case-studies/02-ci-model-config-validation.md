# Case Study 02: CI 中的模型 Config 验证

> **目标**：在模型 PR 或发布流水线中，用 Vitriol 验证 HuggingFace config 可加载、可生成最小权重、可通过 validate。  
> **适用读者**：MLOps 工程师、模型平台维护者  
> **所需环境**：Python 3.9+、`pip install -e ".[dev,viz]"`、纯 CPU CI runner

---

## 背景

新模型入库前常见检查项：

- `config.json` 是否合法、能否被 `transformers` 解析？
- 分片结构是否与 index 一致？
- 自定义 `model_type` 是否需要 adapter / `trust_remote_code`？

Vitriol 的 **`vitriol check`** 把上述步骤合成一条命令，适合嵌入 GitHub Actions / GitLab CI。

---

## 本地复现（离线 fixture）

```bash
# 使用本地 config 目录（无需下载权重）
vitriol --offline check ./path/to/model-config -o ci-report/ --fast --strategy compact
```

`--fast` 跳过 forward 推理与权重分布哈希，适合 CI 快速门禁。

**期望产物**：

```text
ci-report/
├── index.html
├── check-report.json
├── validation.json
└── weights/
    ├── config.json
    ├── vitriol-manifest.json
    └── *.safetensors 或 *.bin
```

`check-report.json` 中 `"success": true` 即可作为门禁判据。

---

## GitHub Actions 集成

仓库已提供示例 workflow：`.github/workflows/vitriol-check-example.yml`。

最小用法：

```yaml
- name: Vitriol structure check
  run: |
    pip install -e ".[viz]"
    vitriol --offline check "${{ inputs.model_id }}" -o vitriol-report --fast
  env:
    PYTHONPATH: src
```

### 判据建议

| 级别 | 检查项 |
|------|--------|
| **必过** | `check-report.json` → `success == true` |
| **必过** | `weights/vitriol-manifest.json` 存在 |
| **建议** | `validation.json` → `model_loadable == true` |
| **可选** | 上传 `index.html` 为 CI artifact |

---

## 与 `validate` 单独使用

若已完成 `vitriol generate`，可只做加载验证：

```bash
vitriol generate org/model -o staging/model --strategy compact
vitriol validate staging/model --no-inference   # 仅加载，不跑 forward
```

---

## 安全默认值

CI 中应使用：

```bash
vitriol --no-trust-remote-code --offline check ...
```

仅在明确信任自定义代码的模型仓库时，才添加 `--trust-remote-code`。

---

## 相关文档

- [Case Study 01: 零下载架构对比](./01-zero-download-architecture-compare.md)
- [CI 集成指南](../ci-integration.md)
- [Release Validation Checklist](../release-validation.md)
