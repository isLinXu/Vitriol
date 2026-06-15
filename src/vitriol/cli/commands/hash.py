from pathlib import Path

import click


@click.command(name="hash")
@click.argument('model_path', type=click.Path(exists=True, path_type=Path))
@click.option('--fast', is_flag=True, help='Only compute architecture hash (skip weights)')
def hash_model(model_path, fast) -> None:
    """
    Generate a cryptographic fingerprint for a model.
    """
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
    except ModuleNotFoundError as exc:
        raise click.ClickException(
            f"Missing optional visualization dependency ({exc}). Install it with: pip install -e '.[viz]' (package extra: vitriol[viz])"
        ) from exc

    from ...core.hasher import ModelHasher

    console = Console()
    console.print(f"[bold blue]Analyzing model:[/bold blue] {model_path.name}")

    hasher = ModelHasher(model_path)

    with console.status("[bold green]Computing Architecture Hash...[/bold green]"):
        arch_hash = hasher.compute_architecture_hash()

    with console.status("[bold green]Computing Behavioral DNA Hash...[/bold green]"):
        behavior_hash = hasher.compute_activation_signature_hash()

    weight_hash = "Skipped (--fast)"
    if not fast:
        with console.status("[bold green]Computing Weight Distribution Hash...[/bold green]"):
            weight_hash = hasher.compute_weight_distribution_hash(max_tensors=50)

    if not fast and arch_hash != "N/A" and weight_hash != "N/A":
        import hashlib
        combined = f"{arch_hash}_{weight_hash}_{behavior_hash}"
        signature = f"arx_{hashlib.sha256(combined.encode('utf-8')).hexdigest()[:16]}"
    else:
        signature = "N/A"

    table = Table(show_header=False, box=None)
    table.add_column("Property", style="cyan", width=25)
    table.add_column("Hash Value", style="yellow")

    table.add_row("Architecture Hash", arch_hash)
    table.add_row("Behavioral DNA Hash", behavior_hash)
    table.add_row("Weight Stats Hash", weight_hash)
    table.add_row("Vitriol Signature", f"[bold green]{signature}[/bold green]")

    panel = Panel(
        table,
        title="[bold]Model Identity Fingerprint[/bold]",
        border_style="blue",
        expand=False
    )

    console.print(panel)
