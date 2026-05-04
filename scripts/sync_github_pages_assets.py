#!/usr/bin/env python3
# Generated: 2026-03-31T00:00Z
# Rules-Ver: 3.0.2
# Context-ID: GITHUB-PAGES-SYNC
"""
Sync a subset of local `output/` artifacts into `docs/` for GitHub Pages static publishing.

Design goals:
- Pages deploys only `docs/`, so we copy in the small set of static assets needed for visualization
- By default we sync only "lightweight assets" (e.g., config.json / a small number of HTML files / images)
  to avoid committing large files
- Generate manifests (JSON) so `docs/viz-models/` and `docs/vocab-viz/` can render automatically

Usage:
  python3 scripts/sync_github_pages_assets.py \
    --model-dir output/Qwen3.5-397B-A17B-Vitriol-ultra-dummy \
    --output-root output
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    _safe_mkdir(dst.parent)
    shutil.copy2(src, dst)
    return True


def _write_json(path: Path, payload: Any) -> None:
    _safe_mkdir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _slugify(name: str) -> str:
    # Minimal URL-safe slugify: keep alnum, dot, underscore, hyphen; convert the rest to hyphens.
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    out = []
    for ch in name:
        out.append(ch if ch in allowed else "-")
    slug = "".join(out).strip("-")
    return slug or "model"


def sync_viz_model(model_dir: Path, docs_dir: Path) -> Dict[str, Any]:
    """
    Sync GitHub Pages assets for a single model:
    - docs/data/<slug>/config.json (+ optional config_meta.json)
    - docs/viz-models/<slug>/(architecture.html, png...)
    """
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    slug = _slugify(model_dir.name)
    data_dst = docs_dir / "data" / slug
    viz_dst = docs_dir / "viz-models" / slug

    copied: Dict[str, bool] = {}
    copied["config.json"] = _copy_if_exists(model_dir / "config.json", data_dst / "config.json")
    copied["meta-config.json"] = _copy_if_exists(model_dir / "meta-config.json", data_dst / "meta-config.json")
    copied["config_meta.json"] = _copy_if_exists(model_dir / "config_meta.json", data_dst / "config_meta.json")
    copied["architecture.html"] = _copy_if_exists(model_dir / "architecture.html", viz_dst / "architecture.html")
    copied["architecture.png"] = _copy_if_exists(model_dir / "architecture.png", viz_dst / "architecture.png")
    copied["architecture_detail.png"] = _copy_if_exists(model_dir / "architecture_detail.png", viz_dst / "architecture_detail.png")

    if not copied["config.json"]:
        raise FileNotFoundError(f"config.json not found: {model_dir / 'config.json'}")

    # Read model_name / architectures from config.json (best-effort) for display.
    model_title = model_dir.name
    try:
        cfg = json.loads((data_dst / "config.json").read_text(encoding="utf-8"))
        model_title = cfg.get("model_name") or cfg.get("name") or model_title
    except Exception:
        pass

    return {
        "slug": slug,
        "title": model_title,
        "source_dir": str(model_dir.as_posix()),
        "pages": {
            "viewer_hash": f"viewer.html#?model=data/{slug}",
            "architecture": f"viz-models/{slug}/architecture.html" if copied["architecture.html"] else None,
        },
        "copied": copied,
    }


def sync_vocab_pages(output_root: Path, docs_dir: Path, patterns: List[str]) -> Dict[str, Any]:
    """
    Sync vocab-viz static HTML into docs/vocab-viz/pages/ and generate a manifest.
    """
    if not output_root.exists():
        raise FileNotFoundError(f"output root does not exist: {output_root}")

    pages_dst = docs_dir / "vocab-viz" / "pages"
    _safe_mkdir(pages_dst)

    copied_pages: List[Dict[str, str]] = []
    for pat in patterns:
        for src in sorted(output_root.glob(pat)):
            if not src.is_file():
                continue
            dst = pages_dst / src.name
            shutil.copy2(src, dst)
            copied_pages.append(
                {
                    "title": src.stem,
                    "path": f"vocab-viz/pages/{src.name}",
                }
            )

    # De-duplicate (multiple patterns may match the same file).
    uniq: Dict[str, Dict[str, str]] = {p["path"]: p for p in copied_pages}
    out = list(uniq.values())
    out.sort(key=lambda x: x["path"])
    return {"count": len(out), "pages": out, "patterns": patterns}


def main() -> int:
    repo = _repo_root()
    docs_dir = repo / "docs"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="output/Qwen3.5-397B-A17B-Vitriol-ultra-dummy",
        help="Default model directory to publish to Pages (must contain config.json).",
    )
    parser.add_argument(
        "--output-root",
        default="output",
        help="Vitriol output root directory (used to collect vocab-viz HTML, etc.).",
    )
    parser.add_argument(
        "--vocab-pattern",
        action="append",
        default=["vocab_*.html"],
        help="Glob(s) of vocab visualization HTML pages to sync into docs (repeatable). Default: vocab_*.html",
    )
    parser.add_argument(
        "--extra-pattern",
        action="append",
        default=[],
        help="Extra output/*.html page glob(s) to sync (e.g., qwen35_*.html).",
    )

    args = parser.parse_args()

    model_dir = (repo / args.model_dir).resolve()
    output_root = (repo / args.output_root).resolve()

    # 1) Sync default model
    viz_entry = sync_viz_model(model_dir=model_dir, docs_dir=docs_dir)

    # 2) Sync vocab pages (and optional extra pages)
    patterns = list(args.vocab_pattern or [])
    patterns.extend(args.extra_pattern or [])
    vocab_manifest = sync_vocab_pages(output_root=output_root, docs_dir=docs_dir, patterns=patterns)

    # 3) Write manifests
    manifests_dir = docs_dir / "manifests"
    _write_json(manifests_dir / "viz_models.json", {"default": viz_entry, "models": [viz_entry]})
    _write_json(manifests_dir / "vocab_viz.json", vocab_manifest)

    print("[OK] Generated/updated GitHub Pages assets and manifests:")
    print(f" - {manifests_dir / 'viz_models.json'}")
    print(f" - {manifests_dir / 'vocab_viz.json'}")
    print("")
    print("Next steps:")
    print("  1) Preview locally: cd docs && python3 -m http.server 8000")
    print("  2) After verifying, commit docs/ to GitHub to deploy Pages automatically")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
