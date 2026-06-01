
import logging
import sys
from pathlib import Path

import click

logger = logging.getLogger(__name__)


def _missing_viz_dependency(exc: Exception) -> click.ClickException:
    return click.ClickException(
        f"Missing optional visualization dependency ({exc}). Install it with: pip install -e '.[viz]' (package extra: vitriol[viz])"
    )

@click.command()
@click.argument('model_dir', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default=None, help='Output directory for report (default: model_dir/visualization)')
@click.option('--layer-pattern', '-p', default=None, help='Regex pattern to filter layers (e.g. "layers.0")')
@click.option('--limit', '-l', type=int, default=None, help='Limit number of tensors to load')
def visualize(model_dir, output_dir, layer_pattern, limit):
    """Generate visualization report for model weights"""
    try:
        from ...visualization.utils import load_weights
        from ...visualization.visualizer import WeightVisualizer
    except (ImportError, ModuleNotFoundError) as exc:
        raise _missing_viz_dependency(exc) from exc

    if output_dir is None:
        output_dir = str(Path(model_dir) / "visualization")

    click.echo(f"Loading weights from {model_dir}...")
    weights = load_weights(model_dir, pattern=layer_pattern, limit=limit)

    if not weights:
        click.echo("No weights loaded. Check directory or filter pattern.", err=True)
        sys.exit(1)

    click.echo(f"Loaded {len(weights)} tensors.")

    visualizer = WeightVisualizer()
    click.echo(f"Generating report in {output_dir}...")

    try:
        visualizer.generate_comprehensive_report(weights, output_dir)
        click.echo("Visualization complete!")
    except Exception as e:
        click.echo(f"Visualization failed: {e}", err=True)
        sys.exit(1)
