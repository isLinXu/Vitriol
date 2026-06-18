"""CLI command: ``vitriol check`` — Structure-First golden path."""

from __future__ import annotations

import sys

import click

from ...core.check_runner import CheckOptions, StructureCheckRunner


@click.command()
@click.argument("model_id")
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output directory for the check report bundle",
)
@click.option(
    "--strategy",
    default="compact",
    show_default=True,
    help="Weight generation strategy for the validate step",
)
@click.option(
    "--fast",
    is_flag=True,
    help="Skip inference validation and weight distribution hashing",
)
@click.option(
    "--skip-generate",
    is_flag=True,
    help="Skip weight generation (analyze + arch-viz only)",
)
@click.option(
    "--skip-validate",
    is_flag=True,
    help="Skip model validation after generation",
)
@click.pass_context
def check(
    ctx,
    model_id: str,
    output: str,
    strategy: str,
    fast: bool,
    skip_generate: bool,
    skip_validate: bool,
) -> None:
    """Run the Structure-First golden path for a model.

    Orchestrates analyze → arch-viz → generate → validate → fingerprint and
    writes ``index.html`` plus ``check-report.json`` under OUTPUT.
    """
    ctx_obj = getattr(ctx, "obj", None) or {}
    options = CheckOptions(
        model_id=model_id,
        output_dir=output,
        strategy=strategy,
        trust_remote_code=bool(ctx_obj.get("trust_remote_code", False)),
        allow_network=bool(ctx_obj.get("allow_network", True)),
        local_files_only=bool(ctx_obj.get("local_files_only", False)),
        run_inference=not fast,
        compute_weight_hash=not fast,
        skip_generate=skip_generate,
        skip_validate=skip_validate,
    )

    click.echo(f"Running Vitriol check for {model_id}")
    click.echo(f"  output: {output}")
    click.echo(f"  strategy: {strategy}")

    report = StructureCheckRunner(options).run()

    for step in report.steps:
        status = "OK" if step.success else "FAIL"
        click.echo(f"  [{status}] {step.name} ({step.duration_seconds:.2f}s)")
        if step.error:
            click.echo(f"         {step.error}", err=True)

    click.echo("")
    if report.success:
        click.echo(f"Check passed. Open {output}/index.html")
    else:
        click.echo(f"Check failed. See {output}/check-report.json", err=True)
        sys.exit(1)
