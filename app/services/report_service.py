# services/report_service.py
from __future__ import annotations
import datetime as dt
from typing import Dict, Any, Tuple

from app.services.markdown_service import md_to_safe_html

# ---- Simple CSS for the standalone HTML report ----
_REPORT_CSS = """
:root { --bg:#0f172a; --panel:#111827; --text:#e5e7eb; --muted:#94a3b8; --accent:#22d3ee; --border:#1f2937; }
* { box-sizing: border-box; }
body { margin:0; padding:24px; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; background:#0b1220; color:var(--text); }
a { color: var(--accent); }
.report { max-width: 900px; margin: 0 auto; background:#0f172a; border:1px solid var(--border); border-radius:12px; padding:24px; }
.report h1, .report h2, .report h3 { margin-top: 18px; }
pre, code { background:#0b1220; border:1px solid #334155; border-radius:8px; }
.table { width:100%; border-collapse: collapse; margin:10px 0; }
.table th, .table td { border:1px solid #334155; padding:6px 8px; text-align:left; }
.meta { color: var(--muted); font-size: 12px; margin-bottom: 12px; }
hr { border: none; border-top: 1px solid #334155; margin: 16px 0; }
"""

def assemble_markdown(ir: Dict[str,Any],
                      explanation_md: str|None,
                      backtest: Dict[str,Any]|None) -> str:
    """Build a single Markdown document from IR + explanation + backtest."""
    def bullets(items):
        items = items or []
        return "\n".join([f"- {it}" for it in items]) if items else "- None"

    md_parts = []
    md_parts.append(f"# Strategy Report")
    md_parts.append(f"_Generated: {dt.datetime.now().isoformat(timespec='seconds')}_\n")

    # IR
    md_parts.append("## Indicators")
    md_parts.append(bullets(ir.get("indicators")) + "\n")

    md_parts.append("## Entry Rules")
    md_parts.append(bullets([r.get("when","") for r in ir.get("entry_rules", [])]) + "\n")

    md_parts.append("## Exit Rules")
    md_parts.append(bullets([r.get("when","") for r in ir.get("exit_rules", [])]) + "\n")

    if ir.get("risk_rules"):
        md_parts.append("## Risk Controls")
        md_parts.append(bullets([", ".join([f"{k}: {v}" for k,v in rr.items()]) for rr in ir["risk_rules"]]) + "\n")

    # Explanation
    if explanation_md:
        md_parts.append("---\n## LLM Explanation\n")
        md_parts.append(explanation_md.strip() + "\n")

    # Backtest
    if backtest:
        md_parts.append("---\n## Backtest Summary")
        md_parts.append(f"- Symbol: **{backtest.get('symbol','')}**")
        md_parts.append(f"- Period: {backtest.get('start','')} → {backtest.get('end','')}")
        md_parts.append(f"- Initial Cash: ${backtest.get('initial_cash',0):,.0f}")
        md_parts.append(f"- Final Equity: ${backtest.get('final_equity',0):,.0f}")
        md_parts.append(f"- CAGR: {backtest.get('cagr',0):.2%}")
        md_parts.append(f"- Sharpe: {backtest.get('sharpe',0):.2f}")
        md_parts.append(f"- Max Drawdown: {backtest.get('max_drawdown',0):.2%}\n")
        if backtest.get("equity_curve"):
            md_parts.append(f"![Equity Curve]({backtest['equity_curve']})\n")

    return "\n".join(md_parts)

def markdown_to_html_page(md_text: str, title: str="Strategy Report") -> str:
    """Wrap sanitized HTML into a standalone page with CSS."""
    body_html = md_to_safe_html(md_text)
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<style>{_REPORT_CSS}</style>
</head>
<body>
  <div class="report">
    <div class="meta">{title}</div>
    {body_html}
  </div>
</body>
</html>"""

def build_report_bytes(ir: Dict[str,Any],
                       explanation_md: str|None,
                       backtest: Dict[str,Any]|None,
                       fmt: str="html") -> Tuple[str, str, bytes]:
    """
    Return (filename, mimetype, bytes) for download.
    fmt: 'html' or 'md'
    """
    md = assemble_markdown(ir, explanation_md, backtest)
    if fmt == "md":
        return ("strategy_report.md", "text/markdown; charset=utf-8", md.encode("utf-8"))
    # default HTML
    html = markdown_to_html_page(md)
    return ("strategy_report.html", "text/html; charset=utf-8", html.encode("utf-8"))
