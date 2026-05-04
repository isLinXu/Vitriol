from __future__ import annotations


def test_trace_command_importable_and_registered() -> None:
    import click

    from vitriol.cli.commands.trace import trace
    from vitriol.cli.main import COMMAND_SPECS, COMMAND_SHORT_HELP

    assert isinstance(trace, click.core.Command)
    assert "trace" in COMMAND_SPECS
    assert "trace" in COMMAND_SHORT_HELP

