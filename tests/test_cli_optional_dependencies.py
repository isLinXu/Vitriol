from pathlib import Path
import sys

from click.testing import CliRunner

from vitriol.cli.main import cli


def test_pyproject_declares_viz_optional_dependencies() -> None:
    content = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "viz = [" in content
    assert '"rich' in content or "'rich" in content
    assert '"matplotlib' in content or "'matplotlib" in content
    assert '"plotly' in content or "'plotly" in content


def test_hash_reports_missing_viz_dependency(monkeypatch, tmp_path) -> None:
    runner = CliRunner()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("rich"):
            raise ModuleNotFoundError("No module named 'rich'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    result = runner.invoke(cli, ["hash", str(model_dir), "--fast"])

    assert result.exit_code != 0
    assert "vitriol[viz]" in result.output


def test_visualize_reports_missing_viz_dependency(monkeypatch, tmp_path) -> None:
    import sys
    # Evict previously imported visualization modules so the import blocker works
    for mod in list(sys.modules):
        if mod.startswith("vitriol.visualization"):
            sys.modules.pop(mod, None)

    runner = CliRunner()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("vitriol.visualization") or name.startswith("matplotlib") or name.startswith("seaborn"):
            raise ModuleNotFoundError("No module named 'matplotlib'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    result = runner.invoke(cli, ["visualize", str(model_dir)])

    assert result.exit_code != 0
    assert "vitriol[viz]" in result.output


def test_vocab_viz_reports_missing_viz_dependency_for_non_3d(monkeypatch) -> None:
    runner = CliRunner()
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("vitriol.vocab_viz") or name.startswith("plotly") or name.startswith("pandas"):
            raise ModuleNotFoundError("No module named 'plotly'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    result = runner.invoke(cli, ["vocab-viz", "--type", "treemap"])

    assert result.exit_code != 0
    assert "vitriol[viz]" in result.output


def _block_heavy_runtime_imports(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        root = name.split(".", 1)[0]
        if root in {"torch", "transformers", "accelerate"}:
            raise ModuleNotFoundError(f"No module named '{root}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)


def _evict_modules(*prefixes: str) -> None:
    safe_prefixes = set(prefixes)
    # These heavy runtimes keep process-global extension/lazy-registry state and
    # are not safe to remove from sys.modules mid-suite. The import blocker still
    # protects the lightweight-help path when they are not already loaded.
    safe_prefixes.difference_update({"torch", "transformers", "accelerate"})
    for module_name in list(sys.modules):
        if any(module_name == prefix or module_name.startswith(f"{prefix}.") for prefix in safe_prefixes):
            sys.modules.pop(module_name, None)


def test_lightweight_subcommand_help_survives_without_heavy_runtime(monkeypatch) -> None:
    runner = CliRunner()
    _block_heavy_runtime_imports(monkeypatch)
    _evict_modules(
        "vitriol.cli.commands.generate",
        "vitriol.cli.commands.analyze",
        "vitriol.cli.commands.validate",
        "vitriol.cli.commands.export",
        "vitriol.cli.commands.batch",
        "vitriol.cli.commands.infer",
        "vitriol.bench.runner",
        "vitriol.strategies",
        "torch",
        "transformers",
        "accelerate",
    )

    for subcommand in ["generate", "analyze", "validate", "export", "batch", "infer"]:
        result = runner.invoke(cli, [subcommand, "--help"])
        assert result.exit_code == 0, f"{subcommand} help failed: {result.output}"
        assert "Usage:" in result.output


def test_analyze_local_model_works_without_transformers(monkeypatch, tmp_path) -> None:
    runner = CliRunner()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"model_type":"gpt2","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"vocab_size":1000}',
        encoding="utf-8",
    )

    _block_heavy_runtime_imports(monkeypatch)
    _evict_modules(
        "vitriol.cli.commands.analyze",
        "vitriol.core.analyzer",
        "vitriol.utils.hf_loading",
        "torch",
        "transformers",
        "accelerate",
    )

    result = runner.invoke(cli, ["analyze", str(model_dir)])

    assert result.exit_code == 0, result.output
    assert "Architecture: gpt2" in result.output
