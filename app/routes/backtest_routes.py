from __future__ import annotations
from flask import Blueprint, request, jsonify
from app.parsers.strategy_parser import parse_strategy
from app.services.backtest_bt_service import run_real_backtest

backtest_bp = Blueprint("backtest", __name__)

@backtest_bp.route("/run", methods=["POST"])
@backtest_bp.route("/run/", methods=["POST"])
def run_backtest():
    data = request.get_json(silent=True) or {}
    code = (data.get("code") or "").strip()
    if not code:
        return jsonify({"error": "No code provided."}), 400

    symbol = data.get("symbol", "SPY")
    start = data.get("start", "2018-01-01")
    end = data.get("end", "2019-01-01")
    initial_cash = float(data.get("initial_cash", 100_000.0))
    commission_bps = float(data.get("commission_bps", 0.0))
    slippage_bps = float(data.get("slippage_bps", 0.0))

    try:
        ir = parse_strategy(code)
    except Exception as e:
        return jsonify({"error": f"Analyze failed: {e.__class__.__name__}: {e}"}), 500

    try:
        allow_shorts = bool(data.get("allow_shorts", False))
        res = run_real_backtest(
            ir, symbol=symbol, start=start, end=end,
            initial_cash=initial_cash,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            allow_shorts=allow_shorts
        )

        if "error" in res:
            return jsonify(res), 400
        return jsonify({"parsed_logic": ir, "backtest": res}), 200
    except Exception as e:
        return jsonify({"error": f"Backtest failed: {e.__class__.__name__}: {e}"}), 500

