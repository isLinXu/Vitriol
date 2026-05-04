from click.testing import CliRunner

from vitriol.cli.main import cli


def test_nas_help_lists_rl_algorithm() -> None:
    result = CliRunner().invoke(cli, ["nas", "--help"])

    assert result.exit_code == 0
    assert "random|evolutionary|targeted|rl" in result.output
    assert "--episodes" in result.output


def test_nas_rl_invokes_controller_with_episode_count(monkeypatch) -> None:
    captured = {}

    class DummyController:
        def __init__(self, output_dir, device):
            captured["output_dir"] = output_dir
            captured["device"] = device

        def run(self, **kwargs):
            captured["run"] = kwargs
            return {"best_gene": None, "history": []}

    monkeypatch.setattr("vitriol.cli.commands.nas.NASController", DummyController)

    result = CliRunner().invoke(
        cli,
        [
            "nas",
            "--algorithm",
            "rl",
            "--episodes",
            "7",
            "--device",
            "cpu",
            "--output-dir",
            "output/test-nas-rl",
        ],
    )

    assert result.exit_code == 0
    assert captured["device"] == "cpu"
    assert captured["run"]["algorithm"] == "rl"
    assert captured["run"]["n_iterations"] == 7
