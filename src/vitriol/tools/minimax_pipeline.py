from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..adapters.registry import AdapterRegistry
from ..patches import PatchRegistry, apply_all_patches

DEFAULT_MODEL_ID = "MiniMaxAI/MiniMax-M2.7"
DEFAULT_STRATEGY = "ultra"
DEFAULT_MAX_SHARD_SIZE = "5GB"
DEFAULT_OUTPUT_RELATIVE = Path("output/minimax_m2_7_ultra")
NESTED_CONFIG_KEYS = ("text_config", "vision_config", "encoder_config", "decoder_config")


@dataclass(frozen=True)
class PipelineOptions:
    """Configuration options for the MiniMax pipeline."""
    repo_root: Path
    output_dir: Path
    python_bin: str = "python3"
    model_id: str = DEFAULT_MODEL_ID
    strategy: str = DEFAULT_STRATEGY
    max_shard_size: str = DEFAULT_MAX_SHARD_SIZE
    trust_remote_code: bool = False
    run_inference: bool = False
    serve_viz: bool = False
    no_open: bool = False
    port: int = 8765


@dataclass(frozen=True)
class PipelineStep:
    """Single step definition in the MiniMax pipeline."""
    name: str
    command: list[str] | None = None


def build_vitriol_command(python_bin: str, *args: str, trust_remote_code: bool = False) -> list[str]:
    command = [python_bin, "-m", "vitriol.cli.main"]
    command.append("--trust-remote-code" if trust_remote_code else "--no-trust-remote-code")
    command.extend(args)
    return command


def build_pipeline_plan(options: PipelineOptions) -> list[PipelineStep]:
    output_dir = str(options.output_dir)
    plan = [
        PipelineStep(
            name="generate",
            command=build_vitriol_command(
                options.python_bin,
                "generate",
                options.model_id,
                "--output-dir",
                output_dir,
                "--strategy",
                options.strategy,
                "--max-shard-size",
                options.max_shard_size,
                "--no-shrink",
                trust_remote_code=options.trust_remote_code,
            ),
        ),
        PipelineStep(name="validate-load"),
        PipelineStep(
            name="arch-viz-block",
            command=build_vitriol_command(
                options.python_bin,
                "arch-viz",
                output_dir,
                "--block",
                "--output",
                str(options.output_dir / "architecture.png"),
                trust_remote_code=options.trust_remote_code,
            ),
        ),
        PipelineStep(
            name="arch-viz-detail",
            command=build_vitriol_command(
                options.python_bin,
                "arch-viz",
                output_dir,
                "--detail",
                "--output",
                str(options.output_dir / "architecture_detail.png"),
                trust_remote_code=options.trust_remote_code,
            ),
        ),
        PipelineStep(
            name="arch-viz-html",
            command=build_vitriol_command(
                options.python_bin,
                "arch-viz",
                output_dir,
                "--html",
                "--output",
                str(options.output_dir / "architecture.html"),
                trust_remote_code=options.trust_remote_code,
            ),
        ),
    ]

    if options.serve_viz:
        command = build_vitriol_command(
            options.python_bin,
            "viz",
            output_dir,
            "--port",
            str(options.port),
            trust_remote_code=options.trust_remote_code,
        )
        if options.no_open:
            command.append("--no-open")
        plan.append(PipelineStep(name="viz-serve", command=command))

    return plan


def repo_root_from_module() -> Path:
    return Path(__file__).resolve().parents[3]


def build_pythonpath(repo_root: Path) -> str:
    src_path = str(repo_root / "src")
    existing = os.environ.get("PYTHONPATH")
    return f"{src_path}:{existing}" if existing else src_path


def run_subprocess(command: list[str], repo_root: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = build_pythonpath(repo_root)
    pretty = " ".join(shlex.quote(part) for part in command)
    print(f"\n==> {pretty}", flush=True)
    subprocess.run(command, cwd=repo_root, env=env, check=True)


def apply_validation_runtime_patches() -> None:
    apply_all_patches()


def patch_config_for_validation(config, model_id_or_path: str) -> Any:
    PatchRegistry.apply(config, model_id_or_path)
    adapter = AdapterRegistry.get_adapter(model_id_or_path, config)
    if adapter:
        return adapter.patch_config(config)
    return config


def normalize_nested_subconfigs(config) -> None:
    from transformers import PretrainedConfig

    for key in NESTED_CONFIG_KEYS:
        value = getattr(config, key, None)
        if isinstance(value, dict):
            setattr(config, key, PretrainedConfig.from_dict(value))


def validate_model_dir(model_dir: Path, run_inference: bool, trust_remote_code: bool) -> dict[str, object]:
    apply_validation_runtime_patches()

    import torch

    from ..utils.hf_loading import load_causallm as hf_load_causallm
    from ..utils.hf_loading import load_config as hf_load_config
    from ..utils.hf_loading import load_tokenizer as hf_load_tokenizer

    report: dict[str, object] = {
        "model_dir": str(model_dir),
        "run_inference": run_inference,
        "config_load": None,
        "tokenizer_load": None,
        "model_load": None,
        "inference": "SKIP",
        "errors": [],
    }

    config_timer = time.perf_counter()
    config = hf_load_config(
        str(model_dir),
        security={
            "trust_remote_code": trust_remote_code,
            "allow_network": False,
            "local_files_only": True,
        },
    )
    config = patch_config_for_validation(config, str(model_dir))
    normalize_nested_subconfigs(config)
    report["config_load"] = f"OK ({time.perf_counter() - config_timer:.2f}s)"
    report["config_type"] = getattr(config, "model_type", "unknown")

    tokenizer_timer = time.perf_counter()
    tokenizer = hf_load_tokenizer(
        str(model_dir),
        security={
            "trust_remote_code": trust_remote_code,
            "allow_network": False,
            "local_files_only": True,
        },
    )
    report["tokenizer_load"] = f"OK ({time.perf_counter() - tokenizer_timer:.2f}s)"
    report["vocab_size"] = len(tokenizer)

    model_timer = time.perf_counter()
    model = hf_load_causallm(
        str(model_dir),
        security={
            "trust_remote_code": trust_remote_code,
            "allow_network": False,
            "local_files_only": True,
        },
        config=config,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    report["model_load"] = f"OK ({time.perf_counter() - model_timer:.2f}s)"
    report["parameter_count"] = int(sum(param.numel() for param in model.parameters()))

    if run_inference:
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer("hello", return_tensors="pt")
        inference_timer = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=3, do_sample=False)
        report["inference"] = f"OK ({time.perf_counter() - inference_timer:.2f}s)"
        report["generated_text_preview"] = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return report


def write_validation_report(model_dir: Path, report: dict[str, object]) -> Path:
    output_path = model_dir / "pipeline_validation.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


def parse_args(argv: list[str] | None = None) -> PipelineOptions:
    parser = argparse.ArgumentParser(
        description="One-click pipeline for MiniMax-M2.7 export, load validation, and visualization.",
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_module())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--python", dest="python_bin", default="python3")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--max-shard-size", default=DEFAULT_MAX_SHARD_SIZE)
    parser.add_argument("--with-inference", action="store_true", help="Run a tiny generate() smoke test after load validation.")
    parser.add_argument("--serve-viz", action="store_true", help="Launch the interactive viz server after static artifacts are generated.")
    parser.add_argument("--no-open", action="store_true", help="Do not auto-open the browser when --serve-viz is used.")
    parser.add_argument("--port", type=int, default=8765)
    trust_group = parser.add_mutually_exclusive_group()
    trust_group.add_argument(
        "--trust-remote-code",
        dest="trust_remote_code",
        action="store_true",
        default=False,
        help="Enable trust_remote_code for trusted model repositories.",
    )
    trust_group.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        help="Disable trust_remote_code for all load steps (default).",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else (repo_root / DEFAULT_OUTPUT_RELATIVE)
    return PipelineOptions(
        repo_root=repo_root,
        output_dir=output_dir,
        python_bin=args.python_bin,
        model_id=args.model_id,
        strategy=args.strategy,
        max_shard_size=args.max_shard_size,
        trust_remote_code=bool(args.trust_remote_code),
        run_inference=args.with_inference,
        serve_viz=args.serve_viz,
        no_open=args.no_open,
        port=args.port,
    )


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    plan = build_pipeline_plan(options)

    print("MiniMax-M2.7 pipeline", flush=True)
    print(f"  repo_root: {options.repo_root}", flush=True)
    print(f"  output_dir: {options.output_dir}", flush=True)
    print(f"  run_inference: {options.run_inference}", flush=True)
    print(f"  serve_viz: {options.serve_viz}", flush=True)

    for step in plan:
        if step.name == "validate-load":
            print("\n==> validate-load", flush=True)
            report = validate_model_dir(
                model_dir=options.output_dir,
                run_inference=options.run_inference,
                trust_remote_code=options.trust_remote_code,
            )
            report_path = write_validation_report(options.output_dir, report)
            print(f"Validation report written to {report_path}", flush=True)
            continue

        if step.command is None:
            continue
        run_subprocess(step.command, options.repo_root)

    print("\nPipeline completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
