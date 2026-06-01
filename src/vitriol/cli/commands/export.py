
import click


@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--output', '-o', required=True, help='Output file path')
@click.option('--format', type=click.Choice(['json', 'gguf']), default='json', help='Export format')
def export(input_dir, output, format):
    """Export model structure or convert format"""
    from ...core.exporter import ModelExporter

    exporter = ModelExporter(input_dir)

    if format == 'json':
        exporter.export_structure(output)
    elif format == 'gguf':
        exporter.export_gguf_prep(output)
