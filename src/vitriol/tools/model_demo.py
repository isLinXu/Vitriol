from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_ID = "zai-org/GLM-5.1"
DEFAULT_OUTPUT_SLUG = "glm_5_1_demo"
PORT_STRIDE = 2


@dataclass(frozen=True)
class DemoOptions:
    repo_root: Path
    output_dir: Path
    python_bin: str = "python3"
    model_id: str = DEFAULT_MODEL_ID
    source_path: Path | None = None
    models_file: Path | None = None
    export_script: Path | None = None
    export_markdown_report: Path | None = None
    write_targets_template: Path | None = None
    log_dir: str = "output/demo_logs"
    arch_port: int = 8765
    weight_port: int = 8781
    no_open: bool = False
    static_arch_viz: bool = False
    list_supported: bool = False
    dry_run: bool = False
    only_families: frozenset[str] = frozenset()
    exclude_families: frozenset[str] = frozenset()
    precheck_config: bool = False
    precheck_viz: bool = False
    trust_remote_code: bool = True


@dataclass(frozen=True)
class DemoStep:
    name: str
    command: list[str] | None = None


def list_supported_families() -> list[str]:
    adapters_dir = repo_root_from_module() / "src" / "vitriol" / "adapters"
    ignored = {"__init__", "base", "registry"}
    families = sorted(
        path.stem
        for path in adapters_dir.glob("*.py")
        if path.stem not in ignored
    )
    return families


def build_vitriol_command(python_bin: str, *args: str, trust_remote_code: bool = True) -> list[str]:
    command = [python_bin, "-m", "vitriol.cli.main"]
    if args and args[0] != "weight-viz":
        command.append("--trust-remote-code" if trust_remote_code else "--no-trust-remote-code")
    command.extend(args)
    return command


def parse_family_filter(raw_value: str | None) -> frozenset[str]:
    if not raw_value:
        return frozenset()
    return frozenset(
        item.strip().lower()
        for item in raw_value.split(",")
        if item.strip()
    )


def infer_target_family(model_id: str, model_path: Path | None) -> str:
    source = model_id or str(model_path or "")
    normalized = source.lower()
    if "hy3" in normalized or "hy_v3" in normalized:
        return "hy3"
    families = list_supported_families()
    for family in families:
        if family in normalized:
            return family
    if "tinyllama" in normalized:
        return "llama"
    return "unknown"


def filter_demo_targets(
    targets: list[dict[str, str | Path | None]],
    only_families: set[str] | frozenset[str],
    exclude_families: set[str] | frozenset[str],
) -> list[dict[str, str | Path | None]]:
    filtered: list[dict[str, str | Path | None]] = []
    for target in targets:
        family = infer_target_family(str(target["model_id"] or ""), target["model_path"])
        if only_families and family not in only_families:
            continue
        if family in exclude_families:
            continue
        filtered.append(target)
    return filtered


def load_demo_targets(file_path: Path) -> list[dict[str, str | Path | None]]:
    targets: list[dict[str, str | Path | None]] = []
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("path:"):
            targets.append({"model_id": "", "model_path": Path(line[5:]).expanduser()})
            continue
        if line.startswith("id:"):
            targets.append({"model_id": line[3:], "model_path": None})
            continue
        if line.startswith(("~", ".", "/")):
            targets.append({"model_id": "", "model_path": Path(line).expanduser()})
            continue
        targets.append({"model_id": line, "model_path": None})
    return targets


def probe_demo_target(
    model_id: str,
    model_path: Path | None,
    trust_remote_code: bool = True,
) -> dict[str, object]:
    from ..utils.hf_loading import load_config_or_raw

    if model_path is not None:
        config = load_config_or_raw(
            str(model_path),
            security={
                "trust_remote_code": trust_remote_code,
                "allow_network": False,
                "local_files_only": True,
            },
        )
        return {
            "source": str(model_path),
            "model_type": getattr(config, "model_type", "unknown"),
            "config_dict": config.to_dict(),
        }

    config = load_config_or_raw(
        model_id,
        security={
            "trust_remote_code": trust_remote_code,
            "allow_network": True,
            "local_files_only": False,
        },
    )
    return {
        "source": model_id,
        "model_type": getattr(config, "model_type", "unknown"),
        "config_dict": config.to_dict(),
    }


def precheck_demo_targets(
    targets: list[dict[str, str | Path | None]],
    trust_remote_code: bool = True,
) -> list[dict[str, str]]:
    reports: list[dict[str, str]] = []
    for target in targets:
        model_id = str(target["model_id"] or "")
        model_path = target["model_path"]
        source = str(model_path or model_id)
        try:
            meta = probe_demo_target(model_id, model_path, trust_remote_code=trust_remote_code)
            reports.append({
                "source": source,
                "status": "ok",
                "model_type": meta["model_type"],
            })
        except Exception as exc:
            reports.append({
                "source": source,
                "status": "fail",
                "error": str(exc),
            })
    return reports


def check_viz_metadata(config_dict: dict[str, object]) -> dict[str, object]:
    text_cfg = config_dict.get("text_config", config_dict)
    if not isinstance(text_cfg, dict):
        text_cfg = config_dict

    required = ("hidden_size", "num_hidden_layers", "vocab_size")
    missing = [key for key in required if text_cfg.get(key) in (None, "")]
    if missing:
        return {
            "ok": False,
            "error": f"missing required visualization fields: {', '.join(missing)}",
        }
    return {
        "ok": True,
        "hidden_size": text_cfg.get("hidden_size"),
        "num_hidden_layers": text_cfg.get("num_hidden_layers"),
        "vocab_size": text_cfg.get("vocab_size"),
    }


def precheck_viz_targets(
    targets: list[dict[str, str | Path | None]],
    trust_remote_code: bool = True,
) -> list[dict[str, str]]:
    reports: list[dict[str, str]] = []
    for target in targets:
        model_id = str(target["model_id"] or "")
        model_path = target["model_path"]
        source = str(model_path or model_id)
        try:
            meta = probe_demo_target(model_id, model_path, trust_remote_code=trust_remote_code)
            viz_check = check_viz_metadata(meta["config_dict"])  # type: ignore[index]
            if not bool(viz_check["ok"]):
                raise RuntimeError(str(viz_check["error"]))
            reports.append({
                "source": source,
                "status": "ok",
                "model_type": str(meta["model_type"]),
            })
        except Exception as exc:
            reports.append({
                "source": source,
                "status": "fail",
                "error": str(exc),
            })
    return reports


def render_targets_template(families: list[str]) -> str:
    lines = [
        "# Demo targets for ./scripts/run_model_demo.sh",
        "# One entry per line. Supports model IDs or local paths.",
        "# Lines starting with # are ignored.",
        "",
        "# Remote model IDs",
    ]
    examples = {
        "glm": "zai-org/GLM-5.1",
        "qwen": "Qwen/Qwen2.5-7B",
        "llama": "meta-llama/Llama-3.1-8B",
        "gemma": "google/gemma-2-9b",
        "mistral": "mistralai/Mistral-7B-v0.3",
        "phi": "microsoft/Phi-3.5-mini-instruct",
        "cohere": "CohereForAI/aya-expanse-8b",
        "deepseek": "deepseek-ai/DeepSeek-V2-Lite",
        "minimax": "MiniMaxAI/MiniMax-M2.7",
        "stablelm": "stabilityai/stablelm-2-1_6b",
    }
    for family in families:
        example = examples.get(family)
        if example:
            lines.append(example)
    lines.extend([
        "tencent/Hy3-preview",
        "",
        "# Local model paths",
        "path:/absolute/path/to/local/model",
        "~/models/example-local-model",
        "",
    ])
    return "\n".join(lines)


def render_markdown_report(
    title: str,
    config_reports: list[dict[str, str]] | None,
    viz_reports: list[dict[str, str]] | None,
    final_targets: list[str],
) -> str:
    lines = [f"# {title}", ""]

    if config_reports is not None:
        lines.extend([
            "## Config Precheck",
            "",
            "| Source | Status | Model Type | Error |",
            "| --- | --- | --- | --- |",
        ])
        for report in config_reports:
            lines.append(
                f"| {report['source']} | {report['status']} | "
                f"{report.get('model_type', '-')} | {report.get('error', '-')} |"
            )
        lines.append("")

    if viz_reports is not None:
        lines.extend([
            "## Visualization Precheck",
            "",
            "| Source | Status | Model Type | Error |",
            "| --- | --- | --- | --- |",
        ])
        for report in viz_reports:
            lines.append(
                f"| {report['source']} | {report['status']} | "
                f"{report.get('model_type', '-')} | {report.get('error', '-')} |"
            )
        lines.append("")

    lines.extend(["## Final Targets", ""])
    if final_targets:
        for target in final_targets:
            lines.append(f"- `{target}`")
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def build_demo_plan(options: DemoOptions) -> list[DemoStep]:
    steps: list[DemoStep] = []
    model_target = str(options.output_dir)

    if options.source_path is None:
        steps.append(DemoStep(name="prepare-demo-dir"))

    if options.static_arch_viz:
        steps.extend([
            DemoStep(
                name="arch-viz-block",
                command=build_vitriol_command(
                    options.python_bin,
                    "arch-viz",
                    model_target,
                    "--block",
                    "--output",
                    str(options.output_dir / "architecture.png"),
                    trust_remote_code=options.trust_remote_code,
                ),
            ),
            DemoStep(
                name="arch-viz-detail",
                command=build_vitriol_command(
                    options.python_bin,
                    "arch-viz",
                    model_target,
                    "--detail",
                    "--output",
                    str(options.output_dir / "architecture_detail.png"),
                    trust_remote_code=options.trust_remote_code,
                ),
            ),
            DemoStep(
                name="arch-viz-html",
                command=build_vitriol_command(
                    options.python_bin,
                    "arch-viz",
                    model_target,
                    "--html",
                    "--output",
                    str(options.output_dir / "architecture.html"),
                    trust_remote_code=options.trust_remote_code,
                ),
            ),
        ])

    weight_cmd = build_vitriol_command(
        options.python_bin,
        "weight-viz",
        "-m",
        model_target,
        "--port",
        str(options.weight_port),
    )
    arch_cmd = build_vitriol_command(
        options.python_bin,
        "viz",
        model_target,
        "--port",
        str(options.arch_port),
        trust_remote_code=options.trust_remote_code,
    )
    if options.no_open:
        weight_cmd.append("--no-open")
        arch_cmd.append("--no-open")

    steps.extend([
        DemoStep(name="weight-viz", command=weight_cmd),
        DemoStep(name="arch-viz", command=arch_cmd),
    ])
    return steps


def build_batch_launch_script(
    targets: list[dict[str, str | Path | None]],
    arch_port_base: int,
    weight_port_base: int,
    no_open: bool,
    static_arch_viz: bool,
    log_dir: str = "output/demo_logs",
) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f'mkdir -p "{log_dir}"',
        "",
    ]
    for index, target in enumerate(targets):
        arch_port = arch_port_base + index * PORT_STRIDE
        weight_port = weight_port_base + index * PORT_STRIDE
        cmd = ["nohup", "./scripts/run_model_demo.sh"]
        model_id = target["model_id"]
        model_path = target["model_path"]
        if model_path is not None:
            cmd.extend(["--model-path", str(model_path)])
        elif isinstance(model_id, str) and model_id:
            cmd.extend(["--model-id", model_id])
        cmd.extend(["--arch-port", str(arch_port), "--weight-port", str(weight_port)])
        if static_arch_viz:
            cmd.append("--static-arch-viz")
        if no_open:
            cmd.append("--no-open")
        log_name = f"demo_{index+1}.log"
        pretty = " ".join(shlex.quote(part) for part in cmd)
        lines.append(f'{pretty} > "{log_dir}/{log_name}" 2>&1 &')
    lines.append("")
    return "\n".join(lines)


def render_plan_lines(plan: list[DemoStep]) -> list[str]:
    lines: list[str] = []
    for index, step in enumerate(plan, start=1):
        if step.command is None:
            lines.append(f"{index}. {step.name}")
        else:
            pretty = " ".join(shlex.quote(part) for part in step.command)
            lines.append(f"{index}. {step.name}: {pretty}")
    return lines


def repo_root_from_module() -> Path:
    return Path(__file__).resolve().parents[3]


def build_pythonpath(repo_root: Path) -> str:
    src_path = str(repo_root / "src")
    existing = os.environ.get("PYTHONPATH")
    return f"{src_path}:{existing}" if existing else src_path


def default_output_dir(repo_root: Path, model_id: str) -> Path:
    slug = model_id.split("/")[-1].lower().replace(".", "_").replace("-", "_")
    return repo_root / "output" / f"{slug}_demo"


def default_markdown_report_path(repo_root: Path) -> Path:
    return repo_root / "output" / "demo_precheck_report.md"


def default_launch_group_dir(repo_root: Path, group_name: str) -> Path:
    return repo_root / "output" / "demo_groups" / group_name


def prepare_demo_dir(model_id: str, output_dir: Path, trust_remote_code: bool) -> None:
    from ..utils.hf_loading import load_config_or_raw

    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config_or_raw(
        model_id,
        security={
            "trust_remote_code": trust_remote_code,
            "allow_network": True,
            "local_files_only": False,
        },
    )
    config.save_pretrained(output_dir)
    meta_path = output_dir / "meta-config.json"
    meta_path.write_text(json.dumps(config.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_background(command: list[str], repo_root: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = build_pythonpath(repo_root)
    pretty = " ".join(shlex.quote(part) for part in command)
    print(f"\n==> {pretty}", flush=True)
    return subprocess.Popen(command, cwd=repo_root, env=env)


def parse_args(argv: list[str] | None = None) -> DemoOptions:
    parser = argparse.ArgumentParser(description="One-click model demo launcher for weight and architecture visualization.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_module())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--models-file", type=Path, default=None)
    parser.add_argument("--export-script", type=Path, default=None)
    parser.add_argument("--export-markdown-report", type=Path, default=None)
    parser.add_argument("--write-targets-template", type=Path, default=None)
    parser.add_argument("--log-dir", default="output/demo_logs")
    parser.add_argument("--python", dest="python_bin", default="python3")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--arch-port", type=int, default=8765)
    parser.add_argument("--weight-port", type=int, default=8781)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--static-arch-viz", action="store_true")
    parser.add_argument("--list-supported", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", default="")
    parser.add_argument("--exclude", default="")
    parser.add_argument("--precheck-config", action="store_true")
    parser.add_argument("--precheck-viz", action="store_true")
    parser.add_argument("--no-trust-remote-code", action="store_true")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    source_path = args.model_path.resolve() if args.model_path else None
    models_file = args.models_file.resolve() if args.models_file else None
    export_script = args.export_script.resolve() if args.export_script else None
    export_markdown_report = args.export_markdown_report.resolve() if args.export_markdown_report else None
    write_targets_template = args.write_targets_template.resolve() if args.write_targets_template else None
    if source_path is not None:
        output_dir = args.output_dir.resolve() if args.output_dir else source_path
        model_id = ""
    else:
        output_dir = args.output_dir.resolve() if args.output_dir else default_output_dir(repo_root, args.model_id)
        model_id = args.model_id
    return DemoOptions(
        repo_root=repo_root,
        output_dir=output_dir,
        python_bin=args.python_bin,
        model_id=model_id,
        source_path=source_path,
        models_file=models_file,
        export_script=export_script,
        export_markdown_report=export_markdown_report,
        write_targets_template=write_targets_template,
        log_dir=args.log_dir,
        arch_port=args.arch_port,
        weight_port=args.weight_port,
        no_open=args.no_open,
        static_arch_viz=args.static_arch_viz,
        list_supported=args.list_supported,
        dry_run=args.dry_run,
        only_families=parse_family_filter(args.only),
        exclude_families=parse_family_filter(args.exclude),
        precheck_config=args.precheck_config,
        precheck_viz=args.precheck_viz,
        trust_remote_code=not args.no_trust_remote_code,
    )


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    if options.list_supported:
        print("Supported adapter families:", flush=True)
        for family in list_supported_families():
            print(f"  - {family}", flush=True)
        return 0

    if options.write_targets_template is not None:
        content = render_targets_template(list_supported_families())
        options.write_targets_template.parent.mkdir(parents=True, exist_ok=True)
        options.write_targets_template.write_text(content, encoding="utf-8")
        print(f"Targets template written to {options.write_targets_template}", flush=True)
        return 0

    if options.models_file is not None:
        targets = load_demo_targets(options.models_file)
        targets = filter_demo_targets(
            targets,
            only_families=options.only_families,
            exclude_families=options.exclude_families,
        )
        config_reports: list[dict[str, str]] | None = None
        viz_reports: list[dict[str, str]] | None = None
        if options.precheck_config:
            reports = precheck_demo_targets(targets, trust_remote_code=options.trust_remote_code)
            config_reports = reports
            print("Precheck report:", flush=True)
            ok_sources: set[str] = set()
            for report in reports:
                if report["status"] == "ok":
                    print(f"  OK   {report['source']} [{report['model_type']}]", flush=True)
                    ok_sources.add(report["source"])
                else:
                    print(f"  FAIL {report['source']} :: {report['error']}", flush=True)
            targets = [
                target for target in targets
                if str(target["model_path"] or target["model_id"]) in ok_sources
            ]
        if options.precheck_viz:
            reports = precheck_viz_targets(targets, trust_remote_code=options.trust_remote_code)
            viz_reports = reports
            print("Visualization precheck report:", flush=True)
            ok_sources: set[str] = set()
            for report in reports:
                if report["status"] == "ok":
                    print(f"  OK   {report['source']}", flush=True)
                    ok_sources.add(report["source"])
                else:
                    print(f"  FAIL {report['source']} :: {report['error']}", flush=True)
            targets = [
                target for target in targets
                if str(target["model_path"] or target["model_id"]) in ok_sources
            ]
        final_targets = [str(target["model_path"] or target["model_id"]) for target in targets]
        if options.export_markdown_report is not None:
            report_content = render_markdown_report(
                title="Demo Precheck Report",
                config_reports=config_reports,
                viz_reports=viz_reports,
                final_targets=final_targets,
            )
            options.export_markdown_report.parent.mkdir(parents=True, exist_ok=True)
            options.export_markdown_report.write_text(report_content, encoding="utf-8")
            print(f"Markdown report written to {options.export_markdown_report}", flush=True)
        script = build_batch_launch_script(
            targets=targets,
            arch_port_base=options.arch_port,
            weight_port_base=options.weight_port,
            no_open=options.no_open,
            static_arch_viz=options.static_arch_viz,
            log_dir=options.log_dir,
        )
        if options.export_script is not None:
            options.export_script.parent.mkdir(parents=True, exist_ok=True)
            options.export_script.write_text(script, encoding="utf-8")
            options.export_script.chmod(0o755)
            print(f"Batch launch script written to {options.export_script}", flush=True)
        else:
            print(script, flush=True)
        return 0

    plan = build_demo_plan(options)

    print("Model demo launcher", flush=True)
    if options.source_path is not None:
        print(f"  model_path: {options.source_path}", flush=True)
    else:
        print(f"  model_id: {options.model_id}", flush=True)
    print(f"  output_dir: {options.output_dir}", flush=True)
    print(f"  weight_viz: http://127.0.0.1:{options.weight_port}/weight_3d_visualizer.html", flush=True)
    print(f"  arch_viz:   http://127.0.0.1:{options.arch_port}/", flush=True)

    if options.dry_run:
        print("\nPlanned steps:", flush=True)
        for line in render_plan_lines(plan):
            print(f"  {line}", flush=True)
        return 0

    if options.source_path is None:
        prepare_demo_dir(options.model_id, options.output_dir, options.trust_remote_code)
    processes: list[subprocess.Popen] = []

    try:
        for step in plan:
            if step.command is None:
                continue
            processes.append(run_background(step.command, options.repo_root))
            time.sleep(1.0)

        print("\nBoth demo servers are running. Press Ctrl+C to stop them.", flush=True)
        while True:
            time.sleep(1.0)
            for proc in processes:
                code = proc.poll()
                if code is not None and code != 0:
                    raise RuntimeError(f"Subprocess exited early with code {code}")
    except KeyboardInterrupt:
        print("\nStopping demo servers...", flush=True)
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            if proc.poll() is None:
                proc.wait(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
