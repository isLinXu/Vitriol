
import click
import sys

@click.command()
@click.argument('config_file', type=click.Path(exists=True))
def batch(config_file):
    """Batch generate models from config file"""
    try:
        from ...core.batch import BatchGenerator

        generator = BatchGenerator(config_file)
        generator.generate_all()
    except Exception as e:
        click.echo(f"Batch generation failed: {e}", err=True)
        sys.exit(1)
