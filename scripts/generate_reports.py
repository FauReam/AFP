#!/usr/bin/env python3
"""Convert all AFP docs/internal/*.md to styled HTML reports in docs/reports/.

Uses markdown-it-py for parsing, with the same visual style as existing reports.
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

from markdown_it import MarkdownIt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERNAL_DIR = PROJECT_ROOT / "docs" / "internal"
REPORTS_DIR = PROJECT_ROOT / "docs" / "reports"
TODAY = date.today().isoformat()

CSS = """<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
         max-width: 860px; margin: 40px auto; padding: 0 24px; line-height: 1.75;
         color: #1a1a1a; background: #fafafa; }
  h1 { font-size: 2em; border-bottom: 3px solid #2563eb; padding-bottom: 12px; }
  h2 { font-size: 1.5em; margin-top: 48px; color: #1e40af; }
  h3 { font-size: 1.15em; margin-top: 32px; color: #374151; }
  h4 { font-size: 1.05em; margin-top: 24px; color: #4b5563; }
  blockquote { border-left: 4px solid #2563eb; margin: 16px 0; padding: 8px 20px;
               background: #eff6ff; font-style: italic; }
  blockquote p { margin: 4px 0; }
  table { border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 0.92em; }
  th { background: #1e3a5f; color: white; padding: 10px 14px; text-align: left; }
  td { padding: 10px 14px; border-bottom: 1px solid #e5e7eb; }
  tr:nth-child(even) td { background: #f8fafc; }
  pre { background: #1e293b; color: #e2e8f0; padding: 18px 22px; border-radius: 8px;
        overflow-x: auto; font-size: 0.88em; line-height: 1.6; }
  code { background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-size: 0.92em; }
  pre code { background: none; padding: 0; }
  .insight { background: #fef3c7; border-left: 4px solid #f59e0b; padding: 12px 20px;
             margin: 20px 0; border-radius: 0 8px 8px 0; }
  .vacuum { background: #dcfce7; border-left: 4px solid #16a34a; padding: 12px 20px;
            margin: 20px 0; border-radius: 0 8px 8px 0; }
  .v2 { background: #ede9fe; border-left: 4px solid #7c3aed; padding: 12px 20px;
        margin: 20px 0; border-radius: 0 8px 8px 0; }
  hr { margin: 40px 0; border: none; border-top: 1px solid #e5e7eb; }
  li { margin: 6px 0; }
  strong { color: #1e3a5f; }
  em { color: #64748b; }
  a { color: #2563eb; }
</style>"""

# Map md filenames to report titles
REPORT_TITLES = {
    "VISION": "AFP — 核心愿景与协议形式化",
    "DIRECTION": "AFP — 研究方向与文献地图",
    "EXPERIMENT_PLAN": "AFP Phase 0 — 实验方案 (v2)",
    "ENGINEERING": "AFP — 工程手册",
    "contribution-assessment": "AFP — 投稿级别评估",
}


def apply_special_blocks(body: str) -> str:
    """Convert markdown blockquote markers to styled divs.

    Blockquotes starting with special keywords get custom styling.
    """
    # Pattern: <blockquote>\n<p>**V2** or <blockquote>\n<p>v2
    for marker, cls in [
        ("v2", "v2"),
        ("V2", "v2"),
        ("insight", "insight"),
        ("vacuum", "vacuum"),
        ("真空", "vacuum"),
    ]:
        body = body.replace(
            f"<blockquote>\n<p>{marker}",
            f'<blockquote class="{cls}">\n<p>'
        )
        body = body.replace(
            f"<blockquote>\n<p><strong>{marker}",
            f'<blockquote class="{cls}">\n<p><strong>'
        )
    return body


def convert_md_to_html(md_path: Path, title: str) -> str:
    """Convert a single markdown file to styled HTML."""
    md_text = md_path.read_text(encoding="utf-8")

    md = MarkdownIt("commonmark")
    body_html = md.render(md_text)

    # Apply special block styling
    body_html = apply_special_blocks(body_html)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
{CSS}
</head>
<body>
{body_html}
<hr>
<footer style="color:#94a3b8;font-size:0.85em;text-align:center;padding:20px 0;">
  AFP Project · Generated {TODAY} · Phase 0 v2
</footer>
</body>
</html>"""
    return html


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Files to convert
    targets = [
        "VISION.md",
        "DIRECTION.md",
        "EXPERIMENT_PLAN.md",
        "ENGINEERING.md",
        "contribution-assessment.md",
    ]

    generated = []
    for filename in targets:
        md_path = INTERNAL_DIR / filename
        if not md_path.exists():
            print(f"  SKIP {filename} — not found")
            continue

        key = filename.replace(".md", "")
        title = REPORT_TITLES.get(key, key.replace("_", " ").title())
        html = convert_md_to_html(md_path, title)

        out_name = filename.replace(".md", ".html")
        out_path = REPORTS_DIR / out_name
        out_path.write_text(html, encoding="utf-8")
        size_kb = out_path.stat().st_size / 1024
        print(f"  ✓ {out_name} ({size_kb:.1f} KB)")
        generated.append(out_path)

    # Phase 0 summary (dated)
    summary_name = f"phase0-summary-{TODAY}.html"
    summary_path = REPORTS_DIR / summary_name
    phase0_md = INTERNAL_DIR / "EXPERIMENT_PLAN.md"
    if phase0_md.exists():
        title = f"AFP Phase 0 — 实验方案 ({TODAY})"
        html = convert_md_to_html(phase0_md, title)
        summary_path.write_text(html, encoding="utf-8")
        print(f"  ✓ {summary_name} ({summary_path.stat().st_size/1024:.1f} KB)")
        generated.append(summary_path)

    print(f"\n{len(generated)} reports generated -> {REPORTS_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
