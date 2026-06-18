"""
CLI command for launching the Web UI.
"""

import logging

import click

logger = logging.getLogger(__name__)


def _load_webui_launch():
    from vitriol.webui import launch

    return launch


def _launch_kwargs(ctx, *, share, port, debug):
    """Forward non-default runtime controls without widening the default call contract."""
    ctx_obj = getattr(ctx, "obj", None) or {}
    kwargs = {
        "share": share,
        "port": port,
        "debug": debug,
    }
    if bool(ctx_obj.get("trust_remote_code", False)):
        kwargs["trust_remote_code"] = True
    if not bool(ctx_obj.get("allow_network", True)):
        kwargs["allow_network"] = False
    if bool(ctx_obj.get("local_files_only", False)):
        kwargs["local_files_only"] = True
    return kwargs


from ...utils.experimental import experimental


@experimental("Gradio Web UI", detail="Install with pip install -e '.[webui]'.")
@click.command(name="webui")
@click.option("--port", "-p", default=7860, help="Port to run the web UI on")
@click.option("--share", is_flag=True, help="Create a public share link")
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.pass_context
def launch_webui(ctx, port, share, debug) -> None:
    """
    Launch the Vitriol Web UI.

    Examples:
        vitriol webui
        vitriol webui --port 8080
        vitriol webui --share
    """
    logger.info(f"Launching Vitriol Web UI on port {port}...")
    click.echo(f"🚀 Starting Vitriol Web UI at http://localhost:{port}")

    try:
        launch = _load_webui_launch()
        launch(**_launch_kwargs(ctx, share=share, port=port, debug=debug))
    except KeyboardInterrupt:
        click.echo("\n👋 Web UI stopped.")
    except Exception as e:
        logger.error(f"Failed to launch web UI: {e}")
        raise click.ClickException(str(e)) from e
