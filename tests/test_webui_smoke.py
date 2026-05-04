
from click.testing import CliRunner

from vitriol.cli.main import cli


def test_webui_launch_calls_vitriol_webui_launch(monkeypatch) -> None:
    runner = CliRunner()
    captured = {}

    def fake_launch(*, share, port, debug):
        captured.update({"share": share, "port": port, "debug": debug})

    monkeypatch.setattr("vitriol.cli.commands.webui._load_webui_launch", lambda: fake_launch)

    result = runner.invoke(cli, ["webui", "--port", "7861", "--debug"])

    assert result.exit_code == 0
    assert captured["port"] == 7861
    assert captured["debug"] is True
    assert captured["share"] is False


def test_webui_launch_wraps_runtime_failure(monkeypatch) -> None:
    runner = CliRunner()

    def fail_launch():
        def _inner(*, share, port, debug):
            raise RuntimeError("boom")

        return _inner

    monkeypatch.setattr("vitriol.cli.commands.webui._load_webui_launch", fail_launch)

    result = runner.invoke(cli, ["webui"])

    assert result.exit_code != 0
    assert "boom" in result.output
