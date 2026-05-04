"""
CLI command for launching the Web UI.
"""

import click
import logging

logger = logging.getLogger(__name__)


def _load_webui_launch():
    from vitriol.webui import launch

    return launch


@click.command(name="webui")
@click.option("--port", "-p", default=7860, help="Port to run the web UI on")
@click.option("--share", is_flag=True, help="Create a public share link")
@click.option("--debug", is_flag=True, help="Enable debug mode")
def launch_webui(port, share, debug):
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
        launch(share=share, port=port, debug=debug)
    except KeyboardInterrupt:
        click.echo("\n👋 Web UI stopped.")
    except Exception as e:
        logger.error(f"Failed to launch web UI: {e}")
        raise click.ClickException(str(e))
