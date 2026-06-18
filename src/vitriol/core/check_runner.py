"""Golden-path orchestration for ``vitriol check``.

Runs the Structure-First research workflow end-to-end:
analyze → arch-viz → generate → validate → fingerprint → report.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config.manager import GenerationConfig, build_generation_config
from ..version import __version__

logger = logging.getLogger(__name__)


@dataclass
class CheckOptions:
    """Runtime options for a check run."""

    model_id: str
    output_dir: str
    strategy: str = "compact"
    trust_remote_code: bool = False
    allow_network: bool = True
    local_files_only: bool = False
    run_inference: bool = True
    compute_weight_hash: bool = True
    skip_generate: bool = False
    skip_validate: bool = False


@dataclass
class CheckStepResult:
    name: str
    success: bool
    duration_seconds: float
    artifacts: Dict[str, str] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "success": self.success,
            "duration_seconds": round(self.duration_seconds, 3),
            "artifacts": dict(self.artifacts),
        }
        if self.summary:
            payload["summary"] = dict(self.summary)
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass
class CheckReport:
    model_id: str
    output_dir: str
    success: bool
    vitriol_version: str
    generated_at: str
    steps: List[CheckStepResult] = field(default_factory=list)
    fingerprint: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "vitriol_version": self.vitriol_version,
            "generated_at": self.generated_at,
            "model_id": self.model_id,
            "output_dir": self.output_dir,
            "success": self.success,
            "steps": [step.to_dict() for step in self.steps],
            "fingerprint": self.fingerprint,
        }


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


class StructureCheckRunner:
    """Execute the Structure-First golden path for one model."""

    STEP_ANALYZE = "analyze"
    STEP_ARCH_VIZ = "arch_viz"
    STEP_GENERATE = "generate"
    STEP_VALIDATE = "validate"
    STEP_FINGERPRINT = "fingerprint"

    def __init__(self, options: CheckOptions):
        self.options = options
        self.output_root = Path(options.output_dir)
        self.weights_dir = self.output_root / "weights"

    def run(self) -> CheckReport:
        self.output_root.mkdir(parents=True, exist_ok=True)
        steps: List[CheckStepResult] = []
        fingerprint: Optional[Dict[str, str]] = None

        steps.append(self._run_analyze())
        steps.append(self._run_arch_viz())

        if not self.options.skip_generate:
            steps.append(self._run_generate())
        else:
            steps.append(
                CheckStepResult(
                    name=self.STEP_GENERATE,
                    success=True,
                    duration_seconds=0.0,
                    summary={"skipped": True},
                )
            )

        if not self.options.skip_validate and not self.options.skip_generate:
            steps.append(self._run_validate())
        else:
            steps.append(
                CheckStepResult(
                    name=self.STEP_VALIDATE,
                    success=True,
                    duration_seconds=0.0,
                    summary={"skipped": True},
                )
            )

        if self.weights_dir.exists():
            steps.append(self._run_fingerprint())
            fingerprint = steps[-1].summary.get("fingerprint")  # type: ignore[assignment]

        success = all(step.success for step in steps)
        report = CheckReport(
            model_id=self.options.model_id,
            output_dir=str(self.output_root),
            success=success,
            vitriol_version=__version__,
            generated_at=_utc_now(),
            steps=steps,
            fingerprint=fingerprint if isinstance(fingerprint, dict) else None,
        )

        self._write_json(self.output_root / "check-report.json", report.to_dict())
        self._write_index_html(report)
        return report

    def _run_step(self, name: str, fn) -> CheckStepResult:
        started = time.monotonic()
        try:
            artifacts, summary = fn()
            return CheckStepResult(
                name=name,
                success=True,
                duration_seconds=time.monotonic() - started,
                artifacts=artifacts,
                summary=summary,
            )
        except Exception as exc:
            logger.exception("Check step %s failed", name)
            return CheckStepResult(
                name=name,
                success=False,
                duration_seconds=time.monotonic() - started,
                error=str(exc),
            )

    def _run_analyze(self) -> CheckStepResult:
        def _execute():
            from .analyzer import ModelAnalyzer

            analyzer = ModelAnalyzer(
                self.options.model_id,
                trust_remote_code=self.options.trust_remote_code,
                allow_network=self.options.allow_network,
                local_files_only=self.options.local_files_only,
            )
            analysis = analyzer.analyze()
            analysis_path = self.output_root / "analysis.json"
            payload = dataclasses.asdict(analysis)
            self._write_json(analysis_path, payload)
            summary = {
                "architecture": analysis.architecture,
                "total_params": analysis.total_params,
                "layer_count": analysis.layer_count,
                "hidden_size": analysis.hidden_size,
                "special_features": list(analysis.special_features),
            }
            return {"analysis.json": _rel(analysis_path, self.output_root)}, summary

        return self._run_step(self.STEP_ANALYZE, _execute)

    def _run_arch_viz(self) -> CheckStepResult:
        def _execute():
            from ..arch_viz.visualizer import ArchitectureVisualizer

            html_path = self.output_root / "architecture.html"
            viz = ArchitectureVisualizer(
                self.options.model_id,
                trust_remote_code=self.options.trust_remote_code,
                local_files_only=self.options.local_files_only or not self.options.allow_network,
            )
            viz.generate_interactive_html(str(html_path))
            if not html_path.exists() or html_path.stat().st_size == 0:
                raise RuntimeError("architecture.html was not created")
            return {"architecture.html": _rel(html_path, self.output_root)}, {
                "model_type": getattr(viz.architecture, "model_type", "unknown"),
                "total_params": getattr(viz.architecture, "total_params", 0),
            }

        return self._run_step(self.STEP_ARCH_VIZ, _execute)

    def _run_generate(self) -> CheckStepResult:
        def _execute():
            from .generator import MinimalWeightGenerator

            self.weights_dir.mkdir(parents=True, exist_ok=True)
            config: GenerationConfig = build_generation_config(
                overrides={
                    "strategy": self.options.strategy,
                    "trust_remote_code": self.options.trust_remote_code,
                    "allow_network": self.options.allow_network,
                    "local_files_only": self.options.local_files_only,
                }
            )
            generator = MinimalWeightGenerator(
                model_id=self.options.model_id,
                output_dir=str(self.weights_dir),
                config=config,
                shrink_config=False,
            )
            result = generator.generate()
            artifacts = {
                "weights/": "weights/",
            }
            if result.manifest_path:
                artifacts["weights/vitriol-manifest.json"] = _rel(
                    Path(result.manifest_path), self.output_root
                )
            if result.index_path:
                artifacts["weights/index"] = _rel(Path(result.index_path), self.output_root)
            summary = {
                "strategy": self.options.strategy,
                "total_size_bytes": result.total_size,
            }
            return artifacts, summary

        return self._run_step(self.STEP_GENERATE, _execute)

    def _run_validate(self) -> CheckStepResult:
        def _execute():
            from .validator import ModelValidator

            validator = ModelValidator(
                str(self.weights_dir),
                trust_remote_code=self.options.trust_remote_code,
            )
            validation = validator.validate(run_inference=self.options.run_inference)
            validation_path = self.output_root / "validation.json"
            payload = {
                "success": validation.success,
                "model_loadable": validation.model_loadable,
                "tokenizer_loadable": validation.tokenizer_loadable,
                "inference_test": validation.inference_test,
                "memory_usage_gb": validation.memory_usage_gb,
                "errors": list(validation.errors or []),
                "warnings": list(validation.warnings or []),
            }
            self._write_json(validation_path, payload)
            if not validation.success:
                raise RuntimeError("; ".join(validation.errors or ["validation failed"]))
            return {"validation.json": _rel(validation_path, self.output_root)}, payload

        return self._run_step(self.STEP_VALIDATE, _execute)

    def _run_fingerprint(self) -> CheckStepResult:
        def _execute():
            from .hasher import ModelHasher

            hasher = ModelHasher(self.weights_dir)
            arch_hash = hasher.compute_architecture_hash()
            behavior_hash = hasher.compute_activation_signature_hash()
            weight_hash = "skipped"
            if self.options.compute_weight_hash:
                weight_hash = hasher.compute_weight_distribution_hash(max_tensors=50)

            if weight_hash != "skipped" and arch_hash != "N/A" and weight_hash != "N/A":
                combined = f"{arch_hash}_{weight_hash}_{behavior_hash}"
                signature = f"arx_{hashlib.sha256(combined.encode('utf-8')).hexdigest()[:16]}"
            else:
                signature = "N/A"

            fingerprint = {
                "architecture_hash": arch_hash,
                "behavioral_dna_hash": behavior_hash,
                "weight_distribution_hash": weight_hash,
                "vitriol_signature": signature,
            }
            fingerprint_path = self.output_root / "fingerprint.json"
            self._write_json(fingerprint_path, fingerprint)
            return {"fingerprint.json": _rel(fingerprint_path, self.output_root)}, {
                "fingerprint": fingerprint
            }

        return self._run_step(self.STEP_FINGERPRINT, _execute)

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _write_index_html(self, report: CheckReport) -> None:
        from .check_report import render_check_index_html

        html = render_check_index_html(report)
        index_path = self.output_root / "index.html"
        index_path.write_text(html, encoding="utf-8")
