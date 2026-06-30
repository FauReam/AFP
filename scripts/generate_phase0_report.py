#!/usr/bin/env python3
"""Generate comprehensive Phase 0 experiment report from result JSONs.
Combines IVN and F-IVN results into a single styled HTML report.
Usage: python scripts/generate_phase0_report.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "docs" / "reports"
IVN_RESULTS = PROJECT / "experiments" / "phase0_ivn" / "ivn_results.json"
FIVN_RESULTS = PROJECT / "experiments" / "phase0_fivn" / "fivn_results.json"

CSS = """<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
         max-width: 960px; margin: 40px auto; padding: 0 24px; line-height: 1.75;
         color: #1a1a1a; background: #fafafa; }
  h1 { font-size: 2em; border-bottom: 3px solid #2563eb; padding-bottom: 12px; }
  h2 { font-size: 1.5em; margin-top: 48px; color: #1e40af; border-bottom: 1px solid #e5e7eb; padding-bottom: 8px; }
  h3 { font-size: 1.15em; margin-top: 32px; color: #374151; }
  table { border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 0.92em; }
  th { background: #1e3a5f; color: white; padding: 10px 14px; text-align: left; }
  td { padding: 10px 14px; border-bottom: 1px solid #e5e7eb; }
  tr:nth-child(even) td { background: #f8fafc; }
  .pass { color: #16a34a; font-weight: bold; }
  .fail { color: #dc2626; font-weight: bold; }
  .delta-pos { color: #16a34a; }
  .delta-neg { color: #dc2626; }
  pre { background: #1e293b; color: #e2e8f0; padding: 18px 22px; border-radius: 8px;
        overflow-x: auto; font-size: 0.88em; line-height: 1.6; }
  .signal { background: #dcfce7; border-left: 4px solid #16a34a; padding: 12px 20px;
            margin: 20px 0; border-radius: 0 8px 8px 0; }
  .noise { background: #fef3c7; border-left: 4px solid #f59e0b; padding: 12px 20px;
           margin: 20px 0; border-radius: 0 8px 8px 0; }
  a { color: #2563eb; }
  hr { margin: 40px 0; border: none; border-top: 1px solid #e5e7eb; }
</style>"""


def fmt_delta(v: float) -> str:
    cls = "delta-pos" if v > 0 else "delta-neg"
    return f'<span class="{cls}">{v:+.4f}</span>'


def verdict(ivn_net: float, afp_net: float, noise_net: float,
            cos: float) -> list[str]:
    """Generate verdict lines based on experiment results."""
    lines = []
    if ivn_net > afp_net:
        lines.append(f'✅ IVN net ({ivn_net:+.4f}) &gt; AFP net ({afp_net:+.4f}) → 多轮谈判优于单次更新')
    elif ivn_net > afp_net - 0.005:
        lines.append(f'⚠️ IVN net ({ivn_net:+.4f}) ≈ AFP net ({afp_net:+.4f}) → 多轮无增益，检查V收敛')
    else:
        lines.append(f'❌ IVN net ({ivn_net:+.4f}) &lt; AFP net ({afp_net:+.4f}) → 单次更新更好')

    signal = abs(ivn_net - noise_net)
    if signal > 0.01:
        lines.append(f'✅ Signal check: |IVN - Noise| = {signal:.4f} &gt; 0.01 → 真实信号，非随机扰动')
    else:
        lines.append(f'⚠️ Signal check: |IVN - Noise| = {signal:.4f} ≤ 0.01 → 可能是噪声')

    if cos < 0.5:
        lines.append(f'✅ Importance cosine = {cos:.3f} &lt; 0.5 → 领域高度互补')
    elif cos < 0.8:
        lines.append(f'⚠️ Importance cosine = {cos:.3f} ∈ [0.5, 0.8) → 部分重叠')
    else:
        lines.append(f'⚠️ Importance cosine = {cos:.3f} ≥ 0.8 → 领域过于相似')

    return lines


def build_ivn_table(results: dict) -> str:
    """Build IVN results table."""
    ne = results.get("no_exchange", {})
    nc = results.get("noise_control", {})
    fa = results.get("fedavg", {})
    afp = results.get("afp_oneshot", {})
    ivn = results.get("ivn", {})

    rows = [
        ("No Exchange", ne.get("a_self", 0), ne.get("a_cross", 0),
         ne.get("b_self", 0), ne.get("b_cross", 0), 0.0),
        ("Noise Control", nc.get("a_self", 0), nc.get("a_cross", 0),
         nc.get("b_self", 0), nc.get("b_cross", 0), nc.get("net", 0)),
        ("FedAvg (best α)", fa.get("a_self", 0), fa.get("a_cross", 0),
         fa.get("b_self", 0), fa.get("b_cross", 0), fa.get("net", 0)),
        ("AFP 1-shot", afp.get("a_self", 0), afp.get("a_cross", 0),
         afp.get("b_self", 0), afp.get("b_cross", 0), afp.get("net", 0)),
        ("IVN", ivn.get("a_self", 0), ivn.get("a_cross", 0),
         ivn.get("b_self", 0), ivn.get("b_cross", 0), ivn.get("net", 0)),
    ]

    table = """<table>
<tr><th>Method</th><th>A Self</th><th>A Cross</th><th>B Self</th><th>B Cross</th><th>Δ Net</th></tr>"""
    for name, a_s, a_c, b_s, b_c, net in rows:
        table += f"<tr><td>{name}</td><td>{a_s:.4f}</td><td>{a_c:.4f}</td>"
        table += f"<td>{b_s:.4f}</td><td>{b_c:.4f}</td><td>{fmt_delta(net)}</td></tr>"
    table += "</table>"

    # Extra info
    table += f"<p><strong>FedAvg α:</strong> {fa.get('alpha', 'N/A')} &nbsp;|&nbsp; "
    table += f"<strong>AFP τ:</strong> {afp.get('tau', 'N/A')} &nbsp;|&nbsp; "
    table += f"<strong>IVN rounds:</strong> {ivn.get('rounds', 'N/A')}</p>"
    return table


def build_fivn_table(results: dict) -> str:
    """Build F-IVN results table."""
    ne = results.get("no_exchange", {})
    fivn = results.get("fivn", {})
    rand = results.get("random_distill", {})

    table = """<table>
<tr><th>Method</th><th>S Self</th><th>S Cross</th><th>T Self</th><th>T Cross</th><th>Δ Net</th></tr>"""
    table += f"<tr><td>No Exchange</td><td>{ne.get('s_self',0):.4f}</td><td>{ne.get('s_cross',0):.4f}</td>"
    table += f"<td>{ne.get('t_self',0):.4f}</td><td>{ne.get('t_cross',0):.4f}</td><td>—</td></tr>"
    table += f"<tr><td>F-IVN</td><td>{fivn.get('s_self',0):.4f}</td><td>{fivn.get('s_cross',0):.4f}</td>"
    table += f"<td>{fivn.get('t_self',0):.4f}</td><td>{fivn.get('t_cross',0):.4f}</td>"
    table += f"<td>{fmt_delta(fivn.get('net',0))}</td></tr>"
    table += f"<tr><td>Random Distill</td><td>{rand.get('s_self',0):.4f}</td><td>{rand.get('s_cross',0):.4f}</td>"
    table += f"<td>—</td><td>—</td><td>—</td></tr>"
    table += "</table>"
    table += f"<p><strong>Negotiation rounds:</strong> {fivn.get('rounds', 'N/A')}</p>"
    return table


def main() -> int:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    html_parts = [f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AFP Phase 0 — Experiment Results</title>
{CSS}
</head>
<body>
<h1>AFP+IVN Phase 0 — 实验结果</h1>
<p>Generated: {now}</p>
"""]

    # --- IVN ---
    html_parts.append("<h2>Experiment A: IVN (Weight-Space)</h2>")
    if IVN_RESULTS.exists():
        results = json.loads(IVN_RESULTS.read_text())
        html_parts.append(f"<p><strong>Models:</strong> {results.get('teacher','?')} ⇄ {results.get('student','?')}</p>")
        html_parts.append(f"<p><strong>Domains:</strong> {results.get('domains',[])}</p>")

        # Verdict
        ivn = results.get("ivn", {})
        afp = results.get("afp_oneshot", {})
        nc = results.get("noise_control", {})
        cos = results.get("importance_cosine", 0)

        html_parts.append('<div class="signal">')
        html_parts.append("<h3>判断</h3><ul>")
        for line in verdict(ivn.get("net", 0), afp.get("net", 0), nc.get("net", 0), cos):
            html_parts.append(f"<li>{line}</li>")
        html_parts.append("</ul></div>")

        html_parts.append("<h3>结果对比</h3>")
        html_parts.append(build_ivn_table(results))

        html_parts.append(f"<h3>Importance Profile</h3>")
        html_parts.append(f"<p>Cosine similarity: {cos:.3f}</p>")
        html_parts.append(f"<pre>A: {[f'{v:.3f}' for v in results.get('imp_a',[])]}</pre>")
        html_parts.append(f"<pre>B: {[f'{v:.3f}' for v in results.get('imp_b',[])]}</pre>")

        html_parts.append(f"<h3>IVN Convergence</h3>")
        html_parts.append(f"<p>Rounds: {ivn.get('rounds','?')}, Final ΔV: {ivn.get('trajectory',[0])[-1]:.6f}</p>")
    else:
        html_parts.append(f'<p class="fail">Results not available — experiment may still be running or failed.</p>')

    # --- F-IVN ---
    html_parts.append("<h2>Experiment B: F-IVN (Function-Space)</h2>")
    if FIVN_RESULTS.exists():
        results = json.loads(FIVN_RESULTS.read_text())
        html_parts.append(f"<p><strong>Teacher:</strong> {results.get('teacher','?')} ({results.get('teacher_domain','?')})</p>")
        html_parts.append(f"<p><strong>Student:</strong> {results.get('student','?')} ({results.get('student_domain','?')})</p>")
        html_parts.append("<h3>结果对比</h3>")
        html_parts.append(build_fivn_table(results))
    else:
        html_parts.append(f'<p class="fail">Results not available — experiment may still be running or failed.</p>')

    html_parts.append(f"""
<hr>
<footer style="color:#94a3b8;font-size:0.85em;text-align:center;padding:20px 0;">
  AFP+IVN Project · Phase 0 · {now}
</footer>
</body>
</html>""")

    out_path = OUT_DIR / f"phase0-results-{datetime.now().strftime('%Y%m%d-%H%M%S')}.html"
    out_path.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"Report: {out_path} ({out_path.stat().st_size/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
