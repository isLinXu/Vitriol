from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

from .context import GenerationContext


class Step(Protocol):
    name: str

    def run(self, ctx: GenerationContext) -> None:
        ...


@dataclass
class GenerationPipeline:
    steps: List[Step]

    def run(self, ctx: GenerationContext) -> None:
        for step in self.steps:
            step.run(ctx)

