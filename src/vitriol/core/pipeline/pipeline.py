from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

from .context import GenerationContext


class Step(Protocol):
    """Protocol defining a single pipeline step interface."""
    name: str

    def run(self, ctx: GenerationContext) -> None:
        ...


@dataclass
class GenerationPipeline:
    """Orchestrator that runs generation steps sequentially."""
    steps: List[Step]

    def run(self, ctx: GenerationContext) -> None:
        for step in self.steps:
            step.run(ctx)

