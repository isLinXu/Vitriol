"""Multi-strategy empirical benchmark: generate → validate → CIS score."""

from __future__ import annotations

import datetime
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..config.manager import build_generation_config
from ..metrics.compression_intelligence import CompressionIntelligenceScorer, compute_theoretical_psi
from ..version import __version__

logger = logging.getLogger(__name__)

DEFAULT_COMPARE_STRATEGIES = ("random", "compact", "ultra")


@dataclass
class StrategyCompareOptions:
    model_id: str
    output_dir: str
    strategies: Sequence[str] = DEFAULT_COMPARE_STRATEGIES
    trust_remote_code: bool = False
    allow_network: bool = True
    local_files_only: bool = False
    run_inference: bool = False
    cis_tensor_limit: int = 50


@dataclass
class StrategyCompareRow:
    strategy: str
    success: bool
    empirical_psi: Optional[float] = None
    theoretical_psi: Optional[float] = None
    validate_success: bool = False
    model_loadable: bool = False
    total_size_bytes: int = 0
    duration_seconds: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "strategy": self.strategy,
            "success": self.success,
            "empirical_psi": self.empirical_psi,
            "theoretical_psi": self.theoretical_psi,
            "validate_success": self.validate_success,
            "model_loadable": self.model_loadable,
            "total_size_bytes": self.total_size_bytes,
            "duration_seconds": round(self.duration_seconds, 3),
        }
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass
class StrategyCompareReport:
    model_id: str
    output_dir: str
    success: bool
    vitriol_version: str
    generated_at: str
    rows: List[StrategyCompareRow] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "vitriol_version": self.vitriol_version,
            "generated_at": self.generated_at,
            "model_id": self.model_id,
            "output_dir": self.output_dir,
            "success": self.success,
            "strategies": [row.to_dict() for row in self.rows],
        }


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total


def _load_weights_for_cis(model_path: Path, *, limit: int) -> Dict[str, Any]:
    from ..visualization.utils import load_weights

    weights = load_weights(str(model_path), limit=limit)
    if not weights:
        raise RuntimeError(f"No tensors found under {model_path}")
    return weights


def render_compare_markdown(report: StrategyCompareReport) -> str:
    lines = [
        "# Vitriol Strategy Compare Report",
        "",
        f"- **Model**: `{report.model_id}`",
        f"- **Vitriol**: v{report.vitriol_version}",
        f"- **Generated**: {report.generated_at}",
        f"- **Overall**: {'PASS' if report.success else 'FAIL'}",
        "",
        "| Strategy | Empirical PSI | Theoretical PSI | Validate | Loadable | Size (MB) | Time (s) |",
        "|----------|---------------|-----------------|----------|----------|-----------|----------|",
    ]
    for row in report.rows:
        emp = f"{row.empirical_psi:.4f}" if row.empirical_psi is not None else "—"
        theo = f"{row.theoretical_psi:.4f}" if row.theoretical_psi is not None else "—"
        size_mb = row.total_size_bytes / (1024 * 1024)
        lines.append(
            f"| {row.strategy} | {emp} | {theo} | "
            f"{'✓' if row.validate_success else '✗'} | "
            f"{'✓' if row.model_loadable else '✗'} | "
            f"{size_mb:.2f} | {row.duration_seconds:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


class StrategyCompareRunner:
    """Generate, validate, and CIS-score multiple strategies for one model."""

    def __init__(self, options: StrategyCompareOptions):
        self.options = options
        self.output_root = Path(options.output_dir)

    def run(self) -> StrategyCompareReport:
        self.output_root.mkdir(parents=True, exist_ok=True)
        rows: List[StrategyCompareRow] = []

        for strategy in self.options.strategies:
            rows.append(self._benchmark_one(strategy))

        success = all(row.success for row in rows)
        report = StrategyCompareReport(
            model_id=self.options.model_id,
            output_dir=str(self.output_root),
            success=success,
            vitriol_version=__version__,
            generated_at=_utc_now(),
            rows=rows,
        )

        json_path = self.output_root / "compare-report.json"
        md_path = self.output_root / "compare-report.md"
        json_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(render_compare_markdown(report), encoding="utf-8")
        return report

    def _benchmark_one(self, strategy: str) -> StrategyCompareRow:
        started = time.monotonic()
        strategy_dir = self.output_root / strategy
        theoretical = round(compute_theoretical_psi(strategy), 4)

        try:
            from .generator import MinimalWeightGenerator
            from .validator import ModelValidator

            strategy_dir.mkdir(parents=True, exist_ok=True)
            config = build_generation_config(
                overrides={
                    "strategy": strategy,
                    "trust_remote_code": self.options.trust_remote_code,
                    "allow_network": self.options.allow_network,
                    "local_files_only": self.options.local_files_only,
                }
            )
            generator = MinimalWeightGenerator(
                model_id=self.options.model_id,
                output_dir=str(strategy_dir),
                config=config,
                shrink_config=False,
            )
            generator.generate()

            validator = ModelValidator(
                str(strategy_dir),
                trust_remote_code=self.options.trust_remote_code,
            )
            validation = validator.validate(run_inference=self.options.run_inference)

            weights = _load_weights_for_cis(strategy_dir, limit=self.options.cis_tensor_limit)
            metrics = CompressionIntelligenceScorer().score_strategy(
                strategy_name=strategy,
                weights=weights,
            )

            row = StrategyCompareRow(
                strategy=strategy,
                success=validation.success,
                empirical_psi=round(metrics.psi_score, 4),
                theoretical_psi=theoretical,
                validate_success=validation.success,
                model_loadable=validation.model_loadable,
                total_size_bytes=_directory_size(strategy_dir),
                duration_seconds=time.monotonic() - started,
            )
            if not validation.success:
                row.error = "; ".join(validation.errors or ["validation failed"])
            return row
        except Exception as exc:
            logger.exception("Strategy compare failed for %s", strategy)
            return StrategyCompareRow(
                strategy=strategy,
                success=False,
                theoretical_psi=theoretical,
                total_size_bytes=_directory_size(strategy_dir),
                duration_seconds=time.monotonic() - started,
                error=str(exc),
            )
