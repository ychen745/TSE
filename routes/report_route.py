from __future__ import annotations
from flask import Blueprint, request, jsonify, send_file, current_app
from io import BytesIO

from parsers.strategy_parser import parse_strategy
from services.report_service import build_report_bytes

# Optional services if you want the route to auto-run explain/backtest:
try:
    from services.llm_service import explain_strategy_html
except Exception:
    explain_strategy_html = None

try:
    from services.backtest_service import run_quick_backtest, BtConfig
except Exception:
    run_quick_backtest, BtConfig = None, None

report_bp = Blueprint("report", __name__)

def _json_error(msg, status=400):
    return jsonify({"error": msg}), status

@report_bp.route("/build", methods=["POST"])
@report_bp.route("/build/", methods=["POST"])
def build_report():
    """
    Body JSON:
      {
        "code": "...",                # required
        "include_explain": true,      # optional (default true)
        "include_backtest": true,     # optional (default false)
        "format": "html" | "md",      # optional (default "html")
        "symbol": "SPY",              # optional (for backtest)
        "start": "2018-01-01",        # optional
        "end": "2025-09-01",          # optional
        "initial_cash": 100000        # optional
      }
    """
    data = request.get_json(silent=True) or {}
    code = (data.get("code") or "").strip()
    if not code:
        return _json_error("No code provided.", 400)

    include_explain = data.get("include_explain", True)
    include_backtest = data.get("include_backtest", False)
    fmt = (data.get("format") or "html").lower()

    # Parse to IR
    try:
        ir = parse_strategy(code)
    except Exception as e:
        current_app.logger.exception("Report parse failed")
        return _json_error(f"Analyze failed: {e.__class__.__name__}", 500)

    # Explanation
    explanation_md = None
    if include_explain:
        if not explain_strategy_html:
            return _json_error("LLM service not available for explanation.", 501)
        try:
            rendered = explain_strategy_html(ir)
            explanation_md = (rendered or {}).get("markdown") if isinstance(rendered, dict) else str(rendered)
        except Exception as e:
            current_app.logger.exception("Report explain failed")
            return _json_error(f"Explain failed: {e.__class__.__name__}", 502)

    # Backtest
    backtest = None
    if include_backtest:
        if not run_quick_backtest or not BtConfig:
            return _json_error("Backtest service not available.", 501)
        try:
            cfg = BtConfig(
                symbol=data.get("symbol", "SPY"),
                start=data.get("start", "2018-01-01"),
                end=data.get("end"),
                initial_cash=float(data.get("initial_cash", 100_000.0)),
            )
            backtest = run_quick_backtest(ir, cfg)
            if "error" in (backtest or {}):
                return _json_error(backtest["error"], 400)
        except Exception as e:
            current_app.logger.exception("Report backtest failed")
            return _json_error(f"Backtest failed: {e.__class__.__name__}", 502)

    # Build the report and stream it
    try:
        filename, mimetype, blob = build_report_bytes(ir, explanation_md, backtest, fmt=fmt)
        return send_file(BytesIO(blob), as_attachment=True, download_name=filename, mimetype=mimetype)
    except Exception as e:
        current_app.logger.exception("Report build failed")
        return _json_error(f"Report build failed: {e.__class__.__name__}", 500)
