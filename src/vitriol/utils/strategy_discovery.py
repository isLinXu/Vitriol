"""Lightweight helpers for discovering strategy metadata without importing ML deps."""

from __future__ import annotations

import ast
from pathlib import Path


def discover_strategy_names() -> list[str]:
    init_path = Path(__file__).resolve().parents[1] / "strategies" / "__init__.py"
    tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))

    names: list[str] = []

    def _visit_node(node):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "STRATEGY_REGISTRY":
                    if isinstance(node.value, ast.Dict):
                        for key in node.value.keys:
                            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                                names.append(key.value)
                    continue

                if isinstance(target, ast.Subscript):
                    value = target.value
                    if not isinstance(value, ast.Name) or value.id != "STRATEGY_REGISTRY":
                        continue

                    slice_node = target.slice
                    if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
                        names.append(slice_node.value)

        # Recursively visit child nodes
        for child in ast.iter_child_nodes(node):
            _visit_node(child)

    for node in tree.body:
        _visit_node(node)

    return list(dict.fromkeys(names))
