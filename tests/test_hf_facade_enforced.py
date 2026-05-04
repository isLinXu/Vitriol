from __future__ import annotations

import ast
from pathlib import Path


def test_transformers_loading_must_go_through_hf_loading_facade() -> None:
    """
    "Deeper" security governance: not only forbid hard-coding trust_remote_code=True,
    but also forbid scattered direct calls to Auto*.from_pretrained/from_config across the codebase.

    Goal: centralize HF loading into vitriol.utils.hf_loading for auditing/testing/consistency.
    """
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src" / "vitriol"

    allowed_files = {
        "src/vitriol/utils/hf_loading.py",
    }

    offenders: list[str] = []
    for path in sorted(src_root.rglob("*.py")):
        rel = str(path.relative_to(repo_root))
        if rel in allowed_files:
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(text, filename=rel)
        except SyntaxError:
            # Ignore non-standard syntax files (e.g., generated/concatenated fragments).
            continue

        bad = False

        def _root_name(expr: ast.AST) -> str | None:
            if isinstance(expr, ast.Name):
                return expr.id
            if isinstance(expr, ast.Attribute):
                return _root_name(expr.value)
            return None

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in {"from_pretrained", "from_config"}:
                continue

            root = _root_name(node.func.value)
            if root in {"AutoConfig", "AutoTokenizer", "AutoModel", "AutoModelForCausalLM"}:
                bad = True
                break

        if bad:
            offenders.append(rel)

    assert offenders == [], (
        "Found transformers loading calls that bypass hf_loading; please migrate to vitriol.utils.hf_loading:\n"
        + "\n".join(offenders)
    )
