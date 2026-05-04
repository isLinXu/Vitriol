
import click

@click.command()
@click.argument('model_id')
@click.pass_context
def analyze(ctx, model_id):
    """Analyze a model architecture"""
    from ...core.analyzer import ModelAnalyzer

    analyzer = ModelAnalyzer(
        model_id,
        trust_remote_code=bool(ctx.obj.get("trust_remote_code", True)) if getattr(ctx, "obj", None) else True,
        allow_network=bool(ctx.obj.get("allow_network", True)) if getattr(ctx, "obj", None) else True,
        local_files_only=bool(ctx.obj.get("local_files_only", False)) if getattr(ctx, "obj", None) else False,
    )
    try:
        analysis = analyzer.analyze()
        
        click.echo(f"Model Analysis: {model_id}")
        click.echo(f"  Architecture: {analysis.architecture}")
        click.echo(f"  Total Params: {analysis.total_params:,}")
        click.echo(f"  Layers: {analysis.layer_count}")
        click.echo(f"  Hidden Size: {analysis.hidden_size}")
        click.echo(f"  Vocab Size: {analysis.vocab_size}")
        click.echo(f"  Special Features: {', '.join(analysis.special_features)}")
        click.echo("\nEstimated File Sizes:")
        for strategy, size in analysis.estimated_file_size.items():
            click.echo(f"  {strategy}: {size:.4f} GB")
            
    except Exception as e:
        click.echo(f"Error analyzing model: {e}", err=True)
