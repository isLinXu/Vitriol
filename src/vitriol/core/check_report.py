"""HTML report renderer for ``vitriol check``."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .check_runner import CheckReport


def render_check_index_html(report: "CheckReport") -> str:
    step_rows = []
    for step in report.steps:
        status = "pass" if step.success else "fail"
        status_label = "PASS" if step.success else "FAIL"
        artifact_links = ", ".join(
            f'<a href="{html.escape(path)}">{html.escape(name)}</a>'
            for name, path in step.artifacts.items()
        ) or "—"
        error_block = ""
        if step.error:
            error_block = f'<div class="error">{html.escape(step.error)}</div>'
        step_rows.append(
            f"""
            <tr class="{status}">
              <td>{html.escape(step.name)}</td>
              <td><span class="badge {status}">{status_label}</span></td>
              <td>{step.duration_seconds:.2f}s</td>
              <td>{artifact_links}{error_block}</td>
            </tr>
            """
        )

    fingerprint_block = ""
    if report.fingerprint:
        fp = report.fingerprint
        fingerprint_block = f"""
        <section>
          <h2>Fingerprint</h2>
          <table>
            <tr><th>Architecture Hash</th><td><code>{html.escape(fp.get('architecture_hash', 'N/A'))}</code></td></tr>
            <tr><th>Behavioral DNA</th><td><code>{html.escape(fp.get('behavioral_dna_hash', 'N/A'))}</code></td></tr>
            <tr><th>Weight Distribution</th><td><code>{html.escape(fp.get('weight_distribution_hash', 'N/A'))}</code></td></tr>
            <tr><th>Vitriol Signature</th><td><code>{html.escape(fp.get('vitriol_signature', 'N/A'))}</code></td></tr>
          </table>
          <p><a href="fingerprint.json">fingerprint.json</a></p>
        </section>
        """

    overall = "pass" if report.success else "fail"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Vitriol Check — {html.escape(report.model_id)}</title>
  <style>
    :root {{
      --bg: #0b1020;
      --panel: #121933;
      --text: #e8ecff;
      --muted: #9aa7d6;
      --pass: #2ecc71;
      --fail: #e74c3c;
      --accent: #6c8cff;
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
      background: linear-gradient(180deg, #070b16 0%, var(--bg) 100%);
      color: var(--text);
      line-height: 1.5;
    }}
    main {{ max-width: 960px; margin: 0 auto; padding: 32px 20px 64px; }}
    h1 {{ margin: 0 0 8px; font-size: 1.8rem; }}
    .meta {{ color: var(--muted); margin-bottom: 24px; }}
    .hero {{
      background: var(--panel);
      border: 1px solid rgba(108, 140, 255, 0.25);
      border-radius: 16px;
      padding: 20px 24px;
      margin-bottom: 24px;
    }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 700;
    }}
    .badge.pass {{ background: rgba(46, 204, 113, 0.15); color: var(--pass); }}
    .badge.fail {{ background: rgba(231, 76, 60, 0.15); color: var(--fail); }}
    section {{
      background: var(--panel);
      border-radius: 16px;
      padding: 20px 24px;
      margin-bottom: 20px;
      border: 1px solid rgba(255,255,255,0.06);
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 10px 8px; border-bottom: 1px solid rgba(255,255,255,0.08); vertical-align: top; }}
    th {{ color: var(--muted); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.04em; }}
    tr.fail td {{ color: #ffb4aa; }}
    code {{ word-break: break-all; }}
    a {{ color: var(--accent); }}
    .links a {{ margin-right: 16px; }}
    .error {{ margin-top: 8px; color: #ffb4aa; font-size: 0.9rem; }}
  </style>
</head>
<body>
  <main>
    <div class="hero">
      <h1>Vitriol Structure Check</h1>
      <div class="meta">
        Model: <strong>{html.escape(report.model_id)}</strong><br>
        Vitriol v{html.escape(report.vitriol_version)} · {html.escape(report.generated_at)}<br>
        Overall: <span class="badge {overall}">{"PASS" if report.success else "FAIL"}</span>
      </div>
      <div class="links">
        <a href="check-report.json">check-report.json</a>
        <a href="analysis.json">analysis.json</a>
        <a href="architecture.html">architecture.html</a>
        <a href="validation.json">validation.json</a>
        <a href="weights/">weights/</a>
      </div>
    </div>

    <section>
      <h2>Pipeline Steps</h2>
      <table>
        <thead>
          <tr><th>Step</th><th>Status</th><th>Duration</th><th>Artifacts</th></tr>
        </thead>
        <tbody>
          {''.join(step_rows)}
        </tbody>
      </table>
    </section>

    {fingerprint_block}
  </main>
</body>
</html>
"""
