from __future__ import annotations
from typing import Dict, Any
from services.engine_bt import run_backtrader_ir, BtExecConfig

def run_real_backtest(strategy_ir: Dict[str,Any],
                      symbol: str="SPY",
                      start: str="2018-01-01",
                      end: str|None=None,
                      initial_cash: float=100_000.0,
                      commission_bps: float=0.0,
                      slippage_bps: float=0.0,
                      allow_shorts=False) -> Dict[str,Any]:
    cfg = BtExecConfig(
        symbol=symbol, start=start, end=end, initial_cash=initial_cash,
        commission_bps=commission_bps, slippage_bps=slippage_bps,
        allow_shorts=allow_shorts
    )

    return run_backtrader_ir(strategy_ir, cfg)
