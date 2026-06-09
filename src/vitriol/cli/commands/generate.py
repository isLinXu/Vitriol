
import sys
from pathlib import Path
from typing import List, Optional

import click

from ...config.manager import build_generation_config
from ...utils.strategy_discovery import discover_strategy_names

MinimalWeightGenerator = None


def _get_strategy_choices() -> List[str]:
    try:
        return discover_strategy_names()
    except Exception:
        # Fallback: keep CLI usable even if strategy registry import fails for any reason.
        return [
            "random",
            "sparse",
            "compact",
            "ultra",
            "ternary",
            "binary",
            "quantized",
            "lowrank",
            "structured_sparse",
        ]


def _validate_strategy(_ctx: click.Context, _param: click.Parameter, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    choices = _get_strategy_choices()
    if value not in choices:
        raise click.BadParameter(f"Unknown strategy: {value}. Available strategies: {', '.join(choices)}")
    return value

@click.command()
@click.argument('model_id')
@click.option('--output-dir', '--output', '-o', required=True, help='Output directory for generated model')
@click.option('--max-shard-size', default="5GB", help='Maximum size per shard (e.g. 5GB, 100MB)')
@click.option('--dtype', help='Data type for generated weights/config metadata (e.g. bfloat16, float16, float32)')
@click.option('--strategy', type=str, callback=_validate_strategy, help='Weight generation strategy')
@click.option('--sparse', is_flag=True, help='Deprecated: Use --strategy sparse')
@click.option('--compact', is_flag=True, help='Deprecated: Use --strategy compact')
@click.option('--ultra', is_flag=True, help='Deprecated: Use --strategy ultra')
@click.option('--n-bits', type=int, default=8, help='Number of bits for quantized strategy')
@click.option('--rank', type=int, default=16, help='Rank for lowrank strategy')
@click.option('--sparsity', type=float, default=0.5, help='Sparsity ratio for structured_sparse strategy')
@click.option('--save-dummy-config', is_flag=True, help='Save config_dummy.json for testing purposes')
@click.option('--shrink/--no-shrink', default=None, help='Shrink model configuration (reduce layers/heads). Defaults to True for ultra strategy, False otherwise.')
@click.option('--visualize', is_flag=True, help='Generate visualization report after generation')
@click.pass_context
def generate(ctx, model_id, output_dir, max_shard_size, dtype, strategy, sparse, compact, ultra, n_bits, rank, sparsity, save_dummy_config, shrink, visualize) -> None:
    """Generate minimal weights for a model"""
    global MinimalWeightGenerator
    current_module = sys.modules.get(__name__)
    generator_cls = getattr(current_module, "MinimalWeightGenerator", MinimalWeightGenerator)
    build_config = getattr(current_module, "build_generation_config", build_generation_config)

    if generator_cls is None:
        from ...core.generator import MinimalWeightGenerator as _MinimalWeightGenerator

        MinimalWeightGenerator = _MinimalWeightGenerator
        if current_module is not None:
            current_module.MinimalWeightGenerator = _MinimalWeightGenerator
        generator_cls = _MinimalWeightGenerator

    trust_remote_code = bool(ctx.obj.get("trust_remote_code", False))
    allow_network = bool(ctx.obj.get("allow_network", True))
    local_files_only = bool(ctx.obj.get("local_files_only", False))

    # Determine CLI strategy (if any)
    cli_strategy = strategy
    if cli_strategy is None:
        if ultra:
            cli_strategy = "ultra"
        elif compact:
            cli_strategy = "compact"
        elif sparse:
            cli_strategy = "sparse"

    parameter_source = click.core.ParameterSource
    overrides = {
        "trust_remote_code": trust_remote_code,
        "allow_network": allow_network,
        "local_files_only": local_files_only,
    }
    if cli_strategy is not None:
        overrides["strategy"] = cli_strategy
    if ctx.get_parameter_source("max_shard_size") != parameter_source.DEFAULT:
        overrides["max_shard_size"] = max_shard_size
    if dtype is not None:
        overrides["dtype"] = dtype
    if ctx.get_parameter_source("n_bits") != parameter_source.DEFAULT:
        overrides["n_bits"] = n_bits
    if ctx.get_parameter_source("rank") != parameter_source.DEFAULT:
        overrides["rank"] = rank
    if ctx.get_parameter_source("sparsity") != parameter_source.DEFAULT:
        overrides["sparsity"] = sparsity

    config = build_config(
        config_path=ctx.obj.get('config_path'),
        overrides=overrides,
    )

    if trust_remote_code:
        click.echo(
            "[SECURITY WARNING] trust_remote_code is enabled: loading some HuggingFace models may execute remote code. "
            "For safer environments, re-run with --no-trust-remote-code.",
            err=True,
        )

    try:
        generator = generator_cls(
            model_id=model_id,
            output_dir=output_dir,
            config=config,
            save_dummy_config=save_dummy_config,
            shrink_config=shrink
        )
        generator.generate()

        if visualize:
            click.echo("\nGenerating visualization report...")
            # Optional dependency: visualization stack (e.g., plotly) may not be installed.
            from ...visualization.utils import load_weights
            from ...visualization.visualizer import WeightVisualizer

            # Load weights (limit to 50 tensors to be safe and fast)
            weights = load_weights(output_dir, limit=50)
            if weights:
                vis_out = str(Path(output_dir) / "visualization")
                vis = WeightVisualizer()
                vis.generate_comprehensive_report(weights, vis_out)
                click.echo(f"Visualization report saved to {vis_out}")
            else:
                click.echo("Failed to load weights for visualization.", err=True)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
