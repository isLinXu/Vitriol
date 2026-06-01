from __future__ import annotations

import ast
import re
from pathlib import Path


def test_no_hardcoded_trust_remote_code_true_in_src() -> None:
    """
    Security governance: the repository must not contain hard-coded trust_remote_code=True.

    Rule: all HF loading must explicitly propagate trust_remote_code from config/CLI,
    instead of hard-coding True at call sites.
    """
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src" / "vitriol"

    offenders: list[str] = []
    for path in sorted(src_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "trust_remote_code=True" in text:
            offenders.append(str(path.relative_to(repo_root)))

    assert offenders == [], "Found hard-coded trust_remote_code=True:\n" + "\n".join(offenders)


def test_no_trust_remote_code_true_defaults_in_src() -> None:
    """trust_remote_code must be an explicit opt-in, not a function/dataclass default."""
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src" / "vitriol"
    default_pattern = re.compile(r"trust_remote_code\s*:\s*bool\s*=\s*True")

    offenders: list[str] = []
    for path in sorted(src_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if default_pattern.search(text):
            offenders.append(str(path.relative_to(repo_root)))

    assert offenders == [], "Found trust_remote_code=True defaults:\n" + "\n".join(offenders)


def test_torch_load_calls_use_weights_only_in_src() -> None:
    """All torch.load call sites must opt into safe weight-only loading."""
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src" / "vitriol"

    offenders: list[str] = []
    for path in sorted(src_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "load"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
            ):
                continue
            weights_only_kw = next((kw for kw in node.keywords if kw.arg == "weights_only"), None)
            if not (
                weights_only_kw is not None
                and isinstance(weights_only_kw.value, ast.Constant)
                and weights_only_kw.value.value is True
            ):
                offenders.append(f"{path.relative_to(repo_root)}:{node.lineno}")

    assert offenders == [], "Found torch.load without weights_only=True:\n" + "\n".join(offenders)
