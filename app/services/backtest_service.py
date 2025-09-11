import io
import math
import json
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import pandas as pd
import yfinance as yf

import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

@dataclass
class BtConfig:
    symbol: str = "SPY"
    start: str = "2018-01-01"
    end: str   = None     # default: today
    initial_cash: float = 100_000.0
    slippage_bps: float = 0.0  # optional, not applied in this minimal version

def _to_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().fillna(0.0)

def _cagr(equity: pd.Series, freq_per_year=252) -> float:
    if len(equity) < 2: return 0.0
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = len(equity) / freq_per_year
    return (1.0 + total_ret) ** (1.0 / max(years, 1e-9)) - 1.0

def _sharpe(returns: pd.Series, rf=0.0, freq_per_year=252) -> float:
    # simple Sharpe (excess return / std * sqrt(252))
    r = returns - rf / freq_per_year
    sd = r.std()
    return float((r.mean() / sd * math.sqrt(freq_per_year))) if sd > 0 else 0.0

def _max_drawdown(equity: pd.Series) -> Tuple[float, float, float]:
    # returns (max_dd, peak_value, trough_value)
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    mdd = dd.min()
    return float(mdd), float(roll_max[dd.idxmin()]), float(equity[dd.idxmin()])

def _plot_equity(equity: pd.Series) -> bytes:
    fig, ax = plt.subplots(figsize=(6,3))
    equity.plot(ax=ax)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Date"); ax.set_ylabel("Equity ($)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def run_quick_backtest(strategy_ir: Dict[str, Any], cfg: BtConfig) -> Dict[str, Any]:
    """
    Minimal “IR-driven” backtest:
    - Long-only when ANY entry rule condition string contains a bullish cue
    - Flat when ANY exit rule triggers
    NOTE: This is a heuristic to give believable demo metrics without full engine.
    For a real engine, integrate Backtrader/`bt` and translate IR -> code.
    """
    # Fetch data
    end = cfg.end or dt.date.today().isoformat()
    df = yf.download(cfg.symbol, start=cfg.start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        return {"error": f"No data for {cfg.symbol} between {cfg.start} and {end}"}

    df.columns = [c[0] for c in df.columns]

    close = df["Close"].rename("close")
    rets  = _to_returns(close)

    # Naive signal from IR (demo): look for words suggesting trend/mean-reversion
    entry_text = " ".join([r["when"] for r in strategy_ir.get("entry_rules", [])]).lower()
    exit_text  = " ".join([r["when"] for r in strategy_ir.get("exit_rules", [])]).lower()

    # Very crude regime detector: if “crosses above” or “>” -> trend; if “< lower”/“rsi<” -> mean-reversion
    bullish_cues = ("crosses above", " > ", "rsi < 30", "close < bollingerbands.bot", "close < lower", "sma > ema")
    bearish_cues = ("crosses below", " < ", "rsi > 70", "close > bollingerbands.top", "close > upper", "sma < ema")

    bullish_bias = any(c in entry_text for c in bullish_cues)
    bearish_bias = any(c in entry_text for c in bearish_cues)

    # Position rule (demo):
    # - if bullish_bias: hold when daily return momentum is positive (5-day sum > 0)
    # - if mean-reversion-ish: hold when price below 10d SMA, exit when above
    s = pd.Series(0, index=close.index, dtype=float)

    if bullish_bias and not bearish_bias:
        mom = rets.rolling(5).sum()
        s = (mom > 0).astype(float)  # 1 long, 0 flat
    else:
        sma10 = close.rolling(10).mean()
        s = (close < sma10).astype(float)

    # Apply exits if exit_text has strong exit cues (basic): force flat on negative momentum days
    if any(c in exit_text for c in ("crosses below", "rsi > 50", "sell", "exit")):
        s = s.where(rets.rolling(2).sum() > -0.01, 0.0)

    # Equity curve
    strat_rets = s.shift(1).fillna(0.0) * rets  # enter at next bar
    equity = (1.0 + strat_rets).cumprod() * cfg.initial_cash

    # Metrics
    out = {
        "symbol": cfg.symbol,
        "start": cfg.start,
        "end": end,
        "initial_cash": cfg.initial_cash,
        "final_equity": float(equity.iloc[-1]),
        "cagr": _cagr(equity),
        "sharpe": _sharpe(strat_rets),
        "max_drawdown": _max_drawdown(equity)[0],
        "trades_approx": int((s.diff().abs() > 0).sum() / 2),
        "equity_curve": None,   # bytes -> base64 filled below
        "daily": {
            "dates": equity.index.strftime("%Y-%m-%d").tolist(),
            "equity": [float(x) for x in equity.values],
            "position": [float(x) for x in s.values],
            "returns": [float(x) for x in strat_rets.values],
        }
    }

    # Optional plot
    try:
        png_bytes = _plot_equity(equity)
        import base64
        out["equity_curve"] = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
    except Exception:
        pass

    return out
