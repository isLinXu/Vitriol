import logging
import os

import click

from vitriol.nas.controller import NASController

logger = logging.getLogger(__name__)


@click.command()
@click.option('--algorithm', type=click.Choice(['random', 'evolutionary', 'targeted', 'rl']), default='random',
              help='Search algorithm (random, evolutionary, targeted, or rl)')
@click.option('--iterations', type=int, default=10, help='Number of iterations for random search')
@click.option('--generations', type=int, default=10, help='Number of generations for evolutionary search')
@click.option('--episodes', type=int, default=None, help='Number of episodes for RL search (defaults to --iterations)')
@click.option('--population', type=int, default=20, help='Population size for evolutionary search')
@click.option('--output-dir', type=click.Path(), default='output/nas_results', help='Directory for NAS artifacts')
@click.option('--resume', is_flag=True, help='Resume from checkpoint if available')
@click.option('--device', type=click.Choice(['cpu', 'cuda', 'mps']), default='cpu', help='Device to use for evaluation')
@click.option('--dataset', type=str, default=None, help='Dataset name (e.g. wikitext)')
@click.option('--dataset-config', type=str, default=None, help='Dataset config name (e.g. wikitext-2-v1)')
@click.option('--dataset-split', type=str, default='train', help='Dataset split to use')
@click.option('--n-samples', type=int, default=100, help='Number of samples to use for evaluation')
# Targeted NAS options
@click.option('--target-vram', type=float, default=None,
              help='Maximum VRAM constraint in GB (for targeted search)')
@click.option('--target-params', type=float, default=None,
              help='Maximum parameters in millions (for targeted search)')
@click.option('--objective', type=click.Choice(['minimize-params', 'minimize-vram', 'maximize-efficiency']),
              default='maximize-efficiency', help='Optimization objective')
@click.pass_context
def nas(ctx, algorithm, iterations, generations, episodes, population, output_dir, resume, device,
        dataset, dataset_config, dataset_split, n_samples, target_vram, target_params, objective) -> None:
    """Run Neural Architecture Search (NAS)

    Examples:
        # Basic random search
        vitriol nas --algorithm random --iterations 20

        # Evolutionary search
        vitriol nas --algorithm evolutionary --generations 10 --population 20

        # Reinforcement-learning search (experimental)
        vitriol nas --algorithm rl --episodes 50

        # Targeted search: find best architecture under 24GB VRAM
        vitriol nas --algorithm targeted --target-vram 24

        # Targeted search: maximize efficiency with 70B param limit
        vitriol nas --algorithm targeted --target-params 70 --objective maximize-efficiency
    """
    logger.info(f"Starting NAS with algorithm={algorithm}, device={device}")

    dataset_cfg = None
    if dataset:
        dataset_cfg = {
            "name": dataset,
            "config": dataset_config,
            "split": dataset_split,
            "n_samples": n_samples
        }

    try:
        controller = NASController(output_dir=output_dir, device=device)

        if algorithm == "random":
            controller.run(algorithm="random", n_iterations=iterations, resume=resume,
                          dataset_config=dataset_cfg)
        elif algorithm == "evolutionary":
            controller.run(
                algorithm="evolutionary",
                n_iterations=generations,
                population_size=population,
                resume=resume,
                dataset_config=dataset_cfg
            )
        elif algorithm == "rl":
            controller.run(
                algorithm="rl",
                n_iterations=episodes if episodes is not None else iterations,
                resume=resume,
                dataset_config=dataset_cfg,
            )
        elif algorithm == "targeted":
            # Run targeted/constraint-based NAS
            from vitriol.nas.targeted_nas import (
                Constraint,
                ConstraintOptimizer,
                ConstraintType,
                ObjectiveType,
                OptimizationTarget,
            )

            # Create optimizer with constraints
            optimizer = ConstraintOptimizer()

            if target_vram:
                optimizer.add_constraint(Constraint(ConstraintType.MAX_VRAM, target_vram))
                click.echo(f"Constraint: max VRAM = {target_vram} GB")

            if target_params:
                optimizer.add_constraint(Constraint(ConstraintType.MAX_PARAMS, target_params * 1e6))
                click.echo(f"Constraint: max params = {target_params}M")

            # Set objective
            if objective == 'minimize-params':
                optimizer.add_objective(OptimizationTarget(ObjectiveType.MINIMIZE_PARAMS))
            elif objective == 'minimize-vram':
                optimizer.add_objective(OptimizationTarget(ObjectiveType.MINIMIZE_VRAM))
            else:
                optimizer.add_objective(OptimizationTarget(ObjectiveType.MAXIMIZE_EFFICIENCY))

            # Run optimization
            click.echo(f"Running targeted NAS (objective: {objective})...")
            from vitriol.nas.search_space import LLMSearchSpace

            gene, score, metrics = optimizer.optimize(
                LLMSearchSpace(),
                None,
                n_iterations=iterations,
                verbose=True
            )

            # Display results
            click.echo("\n=== Targeted NAS Results ===")
            click.echo("Best architecture found:")
            click.echo(f"  Layers: {gene.n_layers}")
            click.echo(f"  Hidden size: {gene.hidden_size}")
            click.echo(f"  Attention: {gene.attention_type}")
            click.echo(f"  FFN: {gene.ffn_type}")
            click.echo("\nMetrics:")
            click.echo(f"  Params: {metrics['params_millions']:.1f}M")
            click.echo(f"  VRAM: {metrics['vram_gb']:.2f} GB")
            click.echo(f"  FLOPs/param: {metrics['flops_per_param']:.2f}")

            # Save result
            import json
            result_path = f"{output_dir}/targeted_result.json"
            os.makedirs(output_dir, exist_ok=True)
            with open(result_path, 'w') as f:
                json.dump({
                    "gene": gene.to_dict(),
                    "metrics": metrics,
                    "constraints": {
                        "target_vram": target_vram,
                        "target_params": target_params,
                    },
                    "objective": objective,
                }, f, indent=2)
            click.echo(f"\nResult saved to: {result_path}")

        logger.info("NAS completed successfully")

    except Exception as e:
        logger.error(f"NAS failed: {e}")
        raise click.ClickException(str(e)) from e
