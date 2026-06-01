
import sys

import click


@click.command()
@click.argument('output_dir', type=click.Path(exists=True))
@click.option('--run-inference/--no-inference', default=True, help='Run inference test')
@click.pass_context
def validate(ctx, output_dir, run_inference):
    """Validate a generated model"""
    from ...core.validator import ModelValidator

    trust_remote_code = bool(ctx.obj.get("trust_remote_code", False))
    if trust_remote_code:
        click.echo(
            "[SECURITY WARNING] trust_remote_code is enabled: loading this model may execute remote code. "
            "For safer environments, re-run with --no-trust-remote-code.",
            err=True,
        )

    validator = ModelValidator(output_dir, trust_remote_code=trust_remote_code)
    report = validator.validate(run_inference=run_inference)

    click.echo(f"Validation Report for {output_dir}:")
    click.echo(f"  Success: {report.success}")
    click.echo(f"  Model Loadable: {report.model_loadable}")
    click.echo(f"  Tokenizer Loadable: {report.tokenizer_loadable}")
    click.echo(f"  Inference Test: {report.inference_test}")
    if report.memory_usage_gb:
        click.echo(f"  Memory Usage: {report.memory_usage_gb:.2f} GB")

    if report.errors:
        click.echo("\nErrors:")
        for err in report.errors:
            click.echo(f"  - {err}")

    if report.warnings:
        click.echo("\nWarnings:")
        for warn in report.warnings:
            click.echo(f"  - {warn}")

    if not report.success:
        sys.exit(1)
