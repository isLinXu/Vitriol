import importlib
import subprocess
import sys

import pytest


@pytest.mark.parametrize("module_name, name", [
    ("runner_kv", "analyze_kv_quantization"),
    ("runner_turboquant", "run_turboquantum_synthetic"),
    ("runner_turboquant", "compare_turboquantum_modes"),
    ("runner_turboquant", "run_turboquantum_on_model_kv"),
])
def test_extracted_bench_functions_keep_public_import_paths(module_name, name):
    import vitriol.bench as bench

    implementation = importlib.import_module(f"vitriol.bench.{module_name}")
    assert name in bench.__all__
    assert getattr(bench, name) is getattr(implementation, name)


def test_bench_package_import_does_not_load_heavy_runtimes():
    script = """
import importlib.abc
import sys

class BlockHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'accelerate'}:
            raise AssertionError(f'Unexpected heavy import: {fullname}')

sys.meta_path.insert(0, BlockHeavyImports())
import vitriol.bench as bench
assert bench.default_prompt_suite()
assert bench.prefix_match_tokens([1, 2], [1, 3]) == 1
for name in ('runner', 'runner_kv', 'runner_turboquant', 'ppl_evaluator'):
    assert f'vitriol.bench.{name}' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
