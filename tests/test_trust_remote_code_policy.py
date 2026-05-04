from __future__ import annotations

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
