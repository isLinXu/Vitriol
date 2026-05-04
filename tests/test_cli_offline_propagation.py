from __future__ import annotations

from click.testing import CliRunner


def test_cli_offline_propagates_to_hf_loading(monkeypatch) -> None:
    """
    P2 end-to-end consistency: CLI --offline must actually affect all HF loading calls:
    allow_network=False => local_files_only=True (ultimately enforced by hf_loading).
    """
    from vitriol.cli.main import cli
    import vitriol.cli.commands.evolve as evolve

    captured = {}

    def fake_load_config(model_id: str, *, security, **kwargs):
        captured["model_id"] = model_id
        captured["security"] = dict(security)

        class _Cfg:
            def to_dict(self):
                return {"model_type": "fake"}

        return _Cfg()

    monkeypatch.setattr(evolve, "hf_load_config", fake_load_config)

    runner = CliRunner()
    result = runner.invoke(cli, ["--offline", "evolve", "tree", "demo/model", "--no-build"])
    assert result.exit_code == 0, result.output

    assert captured["model_id"] == "demo/model"
    assert captured["security"]["allow_network"] is False
    assert captured["security"]["local_files_only"] is True
