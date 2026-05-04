from __future__ import annotations

from ..context import GenerationContext


class LegacyGenerateStep:
    """Legacy wrapper step.

    In the first stage of pipelineization, we keep the original generator logic
    intact and execute it as-is to avoid behavioral changes.
    """

    name = "legacy_generate"

    def run(self, ctx: GenerationContext) -> None:
        if ctx.generator is None:
            raise RuntimeError("GenerationContext.generator is required for LegacyGenerateStep")

        # Delegate to the generator instance; this preserves behavior.
        ctx.generator._generate_legacy_impl()

