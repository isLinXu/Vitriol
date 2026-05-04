from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class Case:
    name: str
    prompt: str


def default_prompt_suite() -> List[Tuple[str, str]]:
    return [
        ("code", "Write a short Python function that checks if a number is prime."),
        ("math", "Solve: If f(x)=x^2+3x+2, what is f(5)? Give the final answer only."),
        ("zh", "Explain in Chinese what KV cache is and why it affects long-context inference speed."),
        ("reasoning", "Write 5 bullet points comparing BFS and DFS."),
    ]


def prefix_match_tokens(a: List[int], b: List[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n
