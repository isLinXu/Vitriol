
import click
import sys

@click.command()
@click.argument('model_id')
@click.option('--output', '-o', default=None, help='Output path (for single file) or directory (for --all)')
@click.option('--all', 'generate_all', is_flag=True, help='Generate all visualization types')
@click.option('--block', is_flag=True, help='Generate block diagram')
@click.option('--detail', is_flag=True, help='Generate detailed diagram')
@click.option('--html', is_flag=True, help='Generate interactive HTML')
@click.option('--style', default='default', help='Visualization style')
@click.pass_context
def arch_viz(ctx, model_id, output, generate_all, block, detail, html, style):
    """Visualize model architecture from config"""
    from ...arch_viz.visualizer import ArchitectureVisualizer
    
    try:
        viz = ArchitectureVisualizer(
            model_id,
            style=style,
            trust_remote_code=bool(ctx.obj.get("trust_remote_code", True)) if getattr(ctx, "obj", None) else True,
            local_files_only=bool(ctx.obj.get("local_files_only", False)) if getattr(ctx, "obj", None) else False,
        )
        
        if generate_all:
            if not output:
                output = f"{model_id.split('/')[-1]}_viz"
            viz.generate_all(output)
            return

        # Default to block diagram if nothing specified
        if not (block or detail or html):
            block = True
            
        if block:
            path = output if output else "architecture_block.png"
            viz.generate_block_diagram(path)
            
        if detail:
            path = output if output else "architecture_detail.png"
            viz.generate_detailed_diagram(path)
            
        if html:
            path = output if output else "architecture.html"
            viz.generate_interactive_html(path)
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
