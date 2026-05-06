import tarfile
import zipfile
import shutil
from pathlib import Path


def test_viz_html_templates_declared_as_package_data():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert '[tool.setuptools.package-data]' in pyproject
    assert '"vitriol.viz" = ["*.html", "*.js"]' in pyproject


def test_built_distributions_include_viz_html_templates(tmp_path):
    import build

    out_dir = tmp_path / "dist"
    project_builder = build.ProjectBuilder(".")
    try:
        sdist = project_builder.build("sdist", str(out_dir))
        wheel = project_builder.build("wheel", str(out_dir))
    finally:
        shutil.rmtree("build", ignore_errors=True)
        shutil.rmtree("src/vitriol.egg-info", ignore_errors=True)

    expected_wheel_files = {
        "vitriol/viz/model_3d_visualizer.html",
        "vitriol/viz/model_visualizer.html",
        "vitriol/viz/vocab_3d_visualizer.html",
        "vitriol/viz/weight_3d_visualizer.html",
    }
    expected_sdist_suffixes = {
        f"src/{path}" for path in expected_wheel_files
    }

    with zipfile.ZipFile(wheel) as zf:
        wheel_files = set(zf.namelist())
    assert expected_wheel_files <= wheel_files

    with tarfile.open(sdist) as tf:
        sdist_files = set(tf.getnames())
    assert all(any(name.endswith(suffix) for name in sdist_files) for suffix in expected_sdist_suffixes)
