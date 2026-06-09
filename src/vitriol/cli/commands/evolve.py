"""
CLI commands for Evolution features (Architecture Tree, Compare, Simulate)
"""

import logging

import click

from vitriol.evolution import (
    ArchComparator,
    ArchitectureRecommender,
    ArchSimulator,
    ComparisonReport,
    EvolutionTree,
    InnovationTimeline,
    TreeVisualizer,
    UseCase,
)
from vitriol.utils.hf_loading import load_config as hf_load_config

logger = logging.getLogger(__name__)


@click.group(name="evolve")
def evolve_group() -> None:
    """Evolution: Architecture Tree, Compare, and Simulation commands."""
    pass


@evolve_group.command(name="tree")
@click.argument("models", nargs=-1, required=False)
@click.option("-o", "--output", "output_path", default="output/evolution_tree.html",
              help="Output HTML path for visualization")
@click.option("--title", default="Architecture Evolution Tree",
              help="Title for the visualization")
@click.option("--build/--no-build", default=True,
              help="Build the complete tree with all models")
def build_tree(models, output_path, title, build) -> None:
    """Build and visualize architecture evolution tree.

    Examples:
        vitriol evolve tree
        vitriol evolve tree --output my_tree.html
    """
    click.echo("Building architecture evolution tree...")

    # Initialize tree
    tree = EvolutionTree()

    # Add custom models if provided
    ctx = click.get_current_context(silent=True)
    trust_remote_code = bool((ctx.obj or {}).get("trust_remote_code", False)) if ctx else False
    allow_network = bool((ctx.obj or {}).get("allow_network", True)) if ctx else True
    local_files_only = bool((ctx.obj or {}).get("local_files_only", False)) if ctx else False
    for model_id in models:
        click.echo(f"Adding model: {model_id}")
        try:
            config = hf_load_config(
                model_id,
                security={
                    "trust_remote_code": trust_remote_code,
                    "allow_network": allow_network,
                    "local_files_only": local_files_only,
                },
            )
            tree.add_model(model_id, config.to_dict())
        except Exception as e:
            click.echo(f"Warning: Could not load {model_id}: {e}")

    # Build tree
    if build:
        tree.build()

    # Visualize
    visualizer = TreeVisualizer(tree)
    output = visualizer.generate_html(output_path, title=title)

    click.echo(f"Evolution tree saved to: {output}")

    # Print summary
    click.echo("\nSummary:")
    click.echo(f"  Total models: {len(tree.nodes)}")
    click.echo(f"  Families: {len(tree.families)}")


@evolve_group.command(name="compare")
@click.argument("model1")
@click.argument("model2")
@click.option("-o", "--output", "output_path", default=None,
              help="Output file path (optional)")
@click.option("--format", "output_format", type=click.Choice(["markdown", "json", "html"]),
              default="markdown", help="Output format")
def compare_models(model1, model2, output_path, output_format) -> None:
    """Compare two model architectures.

    Examples:
        vitriol evolve compare Qwen/Qwen2.5-7B DeepSeek-V3/DeepSeek-V3
        vitriol evolve compare meta-llama/Llama-3-8B Qwen/Qwen2.5-72B --format json
    """
    click.echo(f"Comparing {model1} vs {model2}...")

    # Load configs
    try:
        ctx = click.get_current_context(silent=True)
        trust_remote_code = bool((ctx.obj or {}).get("trust_remote_code", False)) if ctx else False
        allow_network = bool((ctx.obj or {}).get("allow_network", True)) if ctx else True
        local_files_only = bool((ctx.obj or {}).get("local_files_only", False)) if ctx else False
        config1 = hf_load_config(
            model1,
            security={
                "trust_remote_code": trust_remote_code,
                "allow_network": allow_network,
                "local_files_only": local_files_only,
            },
        ).to_dict()
        config2 = hf_load_config(
            model2,
            security={
                "trust_remote_code": trust_remote_code,
                "allow_network": allow_network,
                "local_files_only": local_files_only,
            },
        ).to_dict()
    except Exception as e:
        click.echo(f"Error loading models: {e}")
        return

    # Compare
    comparator = ArchComparator()
    result = comparator.compare_from_ids(model1, model2, config1, config2)

    # Format output
    if output_format == "markdown":
        output = ComparisonReport.to_markdown(result)
    elif output_format == "json":
        output = ComparisonReport.to_json(result)
    else:
        output = ComparisonReport.to_html(result)

    # Write or print
    if output_path:
        with open(output_path, "w") as f:
            f.write(output)
        click.echo(f"Comparison saved to: {output_path}")
    else:
        click.echo("\n" + output)


@evolve_group.command(name="simulate")
@click.argument("model", required=False)
@click.option("--config", "config_path", type=click.Path(exists=True),
              help="Path to config.json file")
@click.option("--batch-size", default=1, help="Batch size for simulation")
@click.option("--seq-length", default=512, help="Sequence length")
@click.option("--dtype", default="bfloat16", help="Data type")
@click.option("--gpu", default="A100", help="GPU model for estimation")
@click.option("-o", "--output", "output_path", default=None,
              help="Output JSON file path")
def simulate_model(model, config_path, batch_size, seq_length, dtype, gpu, output_path) -> None:
    """Simulate performance metrics for a model architecture.

    Examples:
        vitriol evolve simulate Qwen/Qwen2.5-7B
        vitriol evolve simulate --config config.json --gpu H100
    """
    import json

    # Load config
    if model:
        try:
            ctx = click.get_current_context(silent=True)
            trust_remote_code = bool((ctx.obj or {}).get("trust_remote_code", False)) if ctx else False
            allow_network = bool((ctx.obj or {}).get("allow_network", True)) if ctx else True
            local_files_only = bool((ctx.obj or {}).get("local_files_only", False)) if ctx else False
            config = hf_load_config(
                model,
                security={
                    "trust_remote_code": trust_remote_code,
                    "allow_network": allow_network,
                    "local_files_only": local_files_only,
                },
            ).to_dict()
            model_id = model
        except Exception as e:
            click.echo(f"Error loading model: {e}")
            return
    elif config_path:
        with open(config_path) as f:
            config = json.load(f)
        model_id = config.get("model_type", "custom_model")
    else:
        click.echo("Error: Must provide either model ID or config path")
        return

    click.echo(f"Simulating {model_id}...")

    # Simulate
    simulator = ArchSimulator(dtype=dtype, gpu_model=gpu)
    result = simulator.simulate(model_id, config, batch_size, seq_length)

    # Output
    output_dict = result.to_dict()

    if output_path:
        with open(output_path, "w") as f:
            json.dump(output_dict, f, indent=2)
        click.echo(f"Simulation results saved to: {output_path}")
    else:
        click.echo("\n=== Simulation Results ===")
        click.echo(f"Model: {model_id}")
        click.echo("\nParameters:")
        click.echo(f"  Total: {output_dict['total_params']:,}")
        click.echo(f"  Active per token: {output_dict['active_params_per_token']:,}")

        click.echo("\nMemory (VRAM):")
        click.echo(f"  Full model: {output_dict['vram_full_model']:.2f} GB")
        click.echo(f"  Inference: {output_dict['vram_inference']:.2f} GB")
        click.echo(f"  Training: {output_dict['vram_training']:.2f} GB")

        click.echo("\nPerformance:")
        click.echo(f"  FLOPs/token: {output_dict['flops_per_token']:.0f}")
        click.echo(f"  Tokens/sec: {output_dict['tokens_per_second']:.1f}")
        click.echo(f"  Latency: {output_dict['inference_latency_ms']:.2f} ms/token")

        click.echo("\nEfficiency:")
        click.echo(f"  Params/GB VRAM: {output_dict['params_per_vram']:.2f}M")


@evolve_group.command(name="families")
def list_families() -> None:
    """List all known model families."""
    families = EvolutionTree().families

    click.echo("Known Model Families:\n")
    for name, data in families.items():
        root = data.get("root", "Unknown")
        members = list(data.get("members", {}).keys())
        click.echo(f"  {name}")
        click.echo(f"    Root: {root}")
        click.echo(f"    Members: {len(members)}")
        click.echo()


@evolve_group.command(name="timeline")
@click.option("-o", "--output", "output_path", default="output/innovation_timeline.html",
              help="Output HTML path for visualization")
@click.option("--title", default="LLM Architecture Innovation Timeline",
              help="Title for the visualization")
def show_timeline(output_path, title) -> None:
    """Show the timeline of architecture innovations.

    Examples:
        vitriol evolve timeline
        vitriol evolve timeline --output timeline.html
    """
    click.echo("Building innovation timeline...")

    timeline = InnovationTimeline()
    timeline.build_events()
    timeline.save_html(output_path)

    click.echo(f"Timeline saved to: {output_path}")
    click.echo("\nSummary:")
    click.echo(f"  Total innovations: {len(timeline.events)}")
    click.echo(f"  Model families: {len(timeline.tree.families)}")

    # Show high-impact innovations
    click.echo("\nHigh Impact Innovations:")
    for event in timeline.events:
        if event.impact == "high":
            click.echo(f"  [{event.year}] {event.innovation} ({event.family})")


@evolve_group.command(name="recommend")
@click.option("--max-params", type=float, help="Maximum parameters in billions")
@click.option("--max-vram", type=float, help="Maximum VRAM in GB")
@click.option("--use-case", type=click.Choice(["chat", "code", "embedding", "long_context", "general"]),
              default="general", help="Primary use case")
@click.option("--prefer-moe", is_flag=True, help="Prefer MoE architectures")
@click.option("--require-gqa", is_flag=True, help="Require GQA support")
@click.option("--require-long-context", is_flag=True, help="Require 128K+ context")
@click.option("--families", help="Comma-separated list of preferred families")
def recommend_arch(max_params, max_vram, use_case, prefer_moe, require_gqa, require_long_context, families) -> None:
    """Recommend architectures based on requirements.

    Examples:
        vitriol evolve recommend --max-params 7 --max-vram 24
        vitriol evolve recommend --use-case code --prefer-moe
        vitriol evolve recommend --require-long-context --families Qwen,LLaMA
    """
    click.echo("Finding best architectures...\n")

    # Parse families
    preferred_families = None
    if families:
        preferred_families = [f.strip() for f in families.split(",")]

    # Get recommendations
    recommender = ArchitectureRecommender()
    recommendations = recommender.recommend(
        max_params=max_params,
        max_vram=max_vram,
        use_case=UseCase(use_case),
        prefer_moe=prefer_moe,
        require_gqa=require_gqa,
        require_long_context=require_long_context,
        preferred_families=preferred_families,
    )

    if not recommendations:
        click.echo("No matching architectures found. Try relaxing constraints.")
        return

    click.echo(f"Found {len(recommendations)} recommendations:\n")

    for i, rec in enumerate(recommendations[:5], 1):
        click.echo(f"{i}. {rec.model_id}")
        click.echo(f"   Family: {rec.family}")
        click.echo(f"   Parameters: {rec.params_b:.1f}B")
        click.echo(f"   VRAM: {rec.vram_gb:.1f}GB")
        click.echo(f"   Score: {rec.score:.2f}")
        if rec.match_reasons:
            click.echo(f"   Reasons: {', '.join(rec.match_reasons[:3])}")
        if rec.innovations:
            click.echo(f"   Innovations: {', '.join(rec.innovations[:5])}")
        click.echo()


def register(cli_group) -> None:
    """Register evolution commands with CLI."""
    cli_group.add_command(evolve_group)
