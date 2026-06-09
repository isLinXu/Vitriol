import importlib
from pathlib import Path
from typing import List, Optional

import click

from ..utils.logging import setup_logging
from ..version import __version__

COMMAND_SPECS = {
    "infer": "vitriol.cli.commands.infer:infer",
    "generate": "vitriol.cli.commands.generate:generate",
    "trace": "vitriol.cli.commands.trace:trace",
    "validate": "vitriol.cli.commands.validate:validate",
    "analyze": "vitriol.cli.commands.analyze:analyze",
    "batch": "vitriol.cli.commands.batch:batch",
    "bench": "vitriol.cli.commands.bench:bench_group",
    "export": "vitriol.cli.commands.export:export",
    "visualize": "vitriol.cli.commands.visualize:visualize",
    "viz": "vitriol.cli.commands.viz:visualize",
    "arch-viz": "vitriol.cli.commands.arch_viz:arch_viz",
    "nas": "vitriol.cli.commands.nas:nas",
    "vocab-viz": "vitriol.cli.commands.vocab_viz:vocab_viz",
    "weight-viz": "vitriol.cli.commands.weight_viz:weight_viz",
    "evolve": "vitriol.cli.commands.evolve:evolve_group",
    "hash": "vitriol.cli.commands.hash:hash_model",
    "webui": "vitriol.cli.commands.webui:launch_webui",
    "exobrain": "vitriol.cli.commands.exobrain:exobrain_group",
}

COMMAND_SHORT_HELP = {
    "analyze": "Analyze model architecture.",
    "arch-viz": "Visualize model architecture from config.",
    "batch": "Generate multiple models from a YAML config.",
    "bench": "Benchmark KV cache inference presets.",
    "evolve": "Architecture evolution tools.",
    "export": "Export a generated model.",
    "exobrain": "ExoBrain inference & knowledge distillation for shell models.",
    "generate": "Generate minimal weights for a model.",
    "hash": "Compute model weight hash.",
    "infer": "Run single-prompt inference with TurboQuant presets.",
    "nas": "Run Neural Architecture Search (NAS).",
    "trace": "Generate an offline trace.json for replay.",
    "validate": "Validate a generated model.",
    "visualize": "Generate a weight visualization report.",
    "viz": "Launch the interactive model visualizer.",
    "vocab-viz": "Visualize tokenizer vocabulary sizes.",
    "webui": "Launch the Vitriol Web UI.",
    "weight-viz": "Visualize model weights in 3D.",
}


class LazyGroup(click.Group):
    """Load command modules only when Click actually needs them."""

    def list_commands(self, ctx: click.Context) -> List[str]:
        return sorted(COMMAND_SPECS)

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        rows = [
            (name, COMMAND_SHORT_HELP.get(name, ""))
            for name in self.list_commands(ctx)
        ]
        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)

    def get_command(self, ctx: click.Context, name: str) -> Optional[click.Command]:
        spec = COMMAND_SPECS.get(name)
        if spec is None:
            return None

        module_name, attr_name = spec.split(":", 1)
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)


@click.group(cls=LazyGroup)
@click.version_option(__version__, prog_name="vitriol")
@click.option("--log-level", default="INFO", help="Set logging level")
@click.option("--config", type=click.Path(exists=True), help="Path to configuration file")
@click.option(
    "--trust-remote-code/--no-trust-remote-code",
    default=False,
    help=(
        "Whether to allow executing remote model code (trust_remote_code) when loading configs/models. "
        "Disabled by default for safety; pass --trust-remote-code only for trusted model repositories."
    ),
)
@click.option(
    "--allow-network/--no-allow-network",
    default=True,
    help="Whether HuggingFace downloads/network access is allowed. Disable for offline/CI.",
)
@click.option(
    "--local-files-only",
    is_flag=True,
    help="Force local_files_only=True when loading from HuggingFace/transformers.",
)
@click.option(
    "--offline",
    is_flag=True,
    help="Alias for --no-allow-network --local-files-only.",
)
@click.pass_context
def cli(ctx, log_level, config, trust_remote_code, allow_network, local_files_only, offline) -> None:
    """Vitriol: Unified framework for model structure visualization, compression, pruning, quantization and efficient inference."""
    setup_logging(level=log_level)
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = Path(config) if config else None
    ctx.obj["trust_remote_code"] = bool(trust_remote_code)
    allow_network = bool(allow_network) and (not bool(offline))
    local_files_only = bool(local_files_only) or bool(offline) or (not allow_network)
    ctx.obj["allow_network"] = allow_network
    ctx.obj["local_files_only"] = local_files_only


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
