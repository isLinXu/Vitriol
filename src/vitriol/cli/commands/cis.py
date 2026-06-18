"""CLI: ``vitriol cis`` — Compression Intelligence Score (CIS) tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click

from ...core.strategy_benchmark import DEFAULT_COMPARE_STRATEGIES
from ...metrics.compression_intelligence import (
    CompressionIntelligenceScorer,
    STRATEGY_SCORE_MATRIX,
    generate_score_comparison_table,
)
from ...utils.strategy_discovery import discover_strategy_names


def _build_theoretical_ranking() -> list[dict]:
    scorer = CompressionIntelligenceScorer()
    rows = []
    for rank, (name, psi) in enumerate(scorer.score_all_strategies(), start=1):
        info, storage, express, train = STRATEGY_SCORE_MATRIX[name]
        rows.append(
            {
                "rank": rank,
                "strategy": name,
                "psi": round(psi, 4),
                "eta_info": info,
                "eta_storage": storage,
                "eta_express": express,
                "trainability": train,
            }
        )
    return rows


def _load_weights(model_path: Path, *, limit: int = 50) -> dict:
    try:
        from ...visualization.utils import load_weights
    except ImportError as exc:
        raise click.ClickException(
            "Empirical CIS scoring requires visualization dependencies. "
            "Install with: pip install -e '.[viz]'"
        ) from exc
    weights = load_weights(str(model_path), limit=limit)
    if not weights:
        raise click.ClickException(f"No tensors found under {model_path}")
    return weights


@click.group()
def cis_group() -> None:
    """Compression Intelligence Score (CIS) ranking and reports."""


@cis_group.command("rank")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON")
def cis_rank(as_json: bool) -> None:
    """Print theoretical CIS ranking for all known strategies."""
    rows = _build_theoretical_ranking()
    if as_json:
        click.echo(json.dumps(rows, indent=2, ensure_ascii=False))
        return

    click.echo("Theoretical CIS ranking (Ψ = α·η_info + β·η_storage + γ·η_express + δ·T_train)")
    click.echo("")
    click.echo(f"{'Rank':<5} {'Strategy':<18} {'PSI':<8} η_info η_storage η_express T_train")
    for row in rows:
        click.echo(
            f"{row['rank']:<5} {row['strategy']:<18} {row['psi']:<8.4f} "
            f"{row['eta_info']:.2f}   {row['eta_storage']:.2f}      "
            f"{row['eta_express']:.2f}     {row['trainability']:.2f}"
        )

    registered = set(discover_strategy_names())
    matrix = set(STRATEGY_SCORE_MATRIX)
    missing = sorted(registered - matrix)
    if missing:
        click.echo("")
        click.echo(f"Note: strategies without CIS matrix entries: {', '.join(missing)}")


@cis_group.command("table")
@click.option("--output", "-o", type=click.Path(), default=None, help="Write markdown table to file")
def cis_table(output: Optional[str]) -> None:
    """Print or save the markdown CIS comparison table."""
    table = generate_score_comparison_table()
    if output:
        Path(output).write_text(table + "\n", encoding="utf-8")
        click.echo(f"Wrote {output}")
    else:
        click.echo(table)


@cis_group.command("score")
@click.argument("model_path", type=click.Path(exists=True, path_type=Path))
@click.option("--strategy", required=True, help="Strategy name used to generate the weights")
@click.option("--output", "-o", type=click.Path(), default=None, help="Write JSON report")
@click.option("--limit", default=50, show_default=True, help="Max tensors to score")
def cis_score(model_path: Path, strategy: str, output: Optional[str], limit: int) -> None:
    """Empirically score generated weights for one strategy."""
    weights = _load_weights(model_path, limit=limit)
    scorer = CompressionIntelligenceScorer()
    metrics = scorer.score_strategy(strategy_name=strategy, weights=weights)
    payload = {
        "strategy": metrics.strategy_name,
        "psi": round(metrics.psi_score, 4),
        "compression_ratio": metrics.compression_ratio,
        "scores": {
            "info_preservation": metrics.scores.info_preservation,
            "storage_efficiency": metrics.scores.storage_efficiency,
            "expressive_power": metrics.scores.expressive_power,
            "trainability": metrics.scores.trainability,
        },
        "radar_vector": metrics.radar_vector,
        "layers_scored": len(metrics.layer_metrics),
    }
    if output:
        Path(output).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        click.echo(f"Wrote {output}")
    else:
        click.echo(json.dumps(payload, indent=2, ensure_ascii=False))


@cis_group.command("report")
@click.option("--output", "-o", required=True, type=click.Path(), help="Output markdown report path")
def cis_report(output: str) -> None:
    """Write a combined theoretical CIS report (table + top strategies)."""
    table = generate_score_comparison_table()
    rows = _build_theoretical_ranking()
    top = rows[:5]
    summary_lines = [
        "# Vitriol CIS Strategy Report",
        "",
        "## Top 5 Strategies (Theoretical PSI)",
        "",
    ]
    for row in top:
        summary_lines.append(
            f"- **#{row['rank']} {row['strategy']}** — PSI `{row['psi']:.4f}` "
            f"(η_info={row['eta_info']:.2f}, η_storage={row['eta_storage']:.2f}, "
            f"η_express={row['eta_express']:.2f}, T_train={row['trainability']:.2f})"
        )
    summary_lines.extend(["", "## Full Comparison Table", "", table, ""])
    Path(output).write_text("\n".join(summary_lines), encoding="utf-8")
    click.echo(f"Wrote {output}")


@cis_group.command("compare")
@click.argument("model_id")
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output directory for per-strategy weights and compare report",
)
@click.option(
    "--strategies",
    default=",".join(DEFAULT_COMPARE_STRATEGIES),
    show_default=True,
    help="Comma-separated strategy names to benchmark",
)
@click.option(
    "--run-inference",
    is_flag=True,
    help="Run forward-pass validation (slower)",
)
@click.option("--limit", default=50, show_default=True, help="Max tensors for empirical CIS scoring")
@click.pass_context
def cis_compare(
    ctx,
    model_id: str,
    output: str,
    strategies: str,
    run_inference: bool,
    limit: int,
) -> None:
    """Benchmark multiple strategies: generate → validate → empirical CIS."""
    from ...core.strategy_benchmark import StrategyCompareOptions, StrategyCompareRunner

    ctx_obj = getattr(ctx, "obj", None) or {}
    strategy_list = [item.strip() for item in strategies.split(",") if item.strip()]
    if not strategy_list:
        raise click.ClickException("At least one strategy is required")

    unknown = [s for s in strategy_list if s not in discover_strategy_names()]
    if unknown:
        raise click.ClickException(f"Unknown strategies: {', '.join(unknown)}")

    options = StrategyCompareOptions(
        model_id=model_id,
        output_dir=output,
        strategies=strategy_list or DEFAULT_COMPARE_STRATEGIES,
        trust_remote_code=bool(ctx_obj.get("trust_remote_code", False)),
        allow_network=bool(ctx_obj.get("allow_network", True)),
        local_files_only=bool(ctx_obj.get("local_files_only", False)),
        run_inference=run_inference,
        cis_tensor_limit=limit,
    )

    click.echo(f"Benchmarking {len(options.strategies)} strategies for {model_id}")
    report = StrategyCompareRunner(options).run()

    for row in report.rows:
        status = "OK" if row.success else "FAIL"
        psi = f"{row.empirical_psi:.4f}" if row.empirical_psi is not None else "N/A"
        click.echo(
            f"  [{status}] {row.strategy}: empirical_psi={psi}, "
            f"size={row.total_size_bytes / (1024 * 1024):.2f}MB, "
            f"time={row.duration_seconds:.1f}s"
        )
        if row.error:
            click.echo(f"         {row.error}", err=True)

    click.echo("")
    if report.success:
        click.echo(f"Compare passed. See {output}/compare-report.md")
    else:
        click.echo(f"Compare failed. See {output}/compare-report.json", err=True)
        raise SystemExit(1)
