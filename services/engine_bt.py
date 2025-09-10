# services/engine_bt.py
from __future__ import annotations
import io, re, math, base64, datetime as dt
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

import os
os.environ.setdefault("MPLBACKEND", "Agg")  # headless plotting before pyplot import
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import backtrader as bt
import ast

# =========================
# Config
# =========================
@dataclass
class BtExecConfig:
    symbol: str = "SPY"
    start: str = "2018-01-01"
    end: Optional[str] = None
    initial_cash: float = 100_000.0
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    allow_shorts: bool = False
    default_periods: Dict[str, int] = None
    overrides: Dict[str, int] = None  # optional: {"SMA": 50, "RSI": 10}

    def __post_init__(self):
        if self.default_periods is None:
            self.default_periods = {"SMA":20, "EMA":50, "RSI":14, "ATR":14, "BollingerBands":20}
        if self.overrides is None:
            self.overrides = {}

# --- add this helper near the top of engine_bt.py, alongside other helpers ---
def _labels_from_cond_text(text: str) -> set[str]:
    """
    Extract label-like tokens from a condition string, e.g.
    'SMA(20) > EMA(50) and close < BollingerBands.bot' ->
    {'SMA(20)','EMA(50)','close','BollingerBands.bot'}
    """
    src = _sanitize_cond_text(text or "")
    try:
        tree = ast.parse(src, mode="eval")
    except Exception:
        return set()

    labels = set()

    class _Walker(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name):
            labels.add(node.id)
        def visit_Attribute(self, node: ast.Attribute):
            # turn nested attributes into dotted names: foo.bar
            def _name(n):
                if isinstance(n, ast.Name): return n.id
                if isinstance(n, ast.Attribute): return _name(n.value) + "." + n.attr
                return None
            nm = _name(node)
            if nm: labels.add(nm)
        def visit_Call(self, node: ast.Call):
            # convert SMA(20) etc to string labels
            func = _attr_to_name(node.func)
            period = None
            if node.args:
                try: period = int(float(ast.literal_eval(node.args[0])))
                except Exception: period = None
            lbl = f"{func}({period})" if period is not None else func
            labels.add(lbl)
            self.generic_visit(node)
        def visit_Subscript(self, node: ast.Subscript):
            # e.g., SMA(20)[-1] -> SMA(20)
            self.generic_visit(node)

    _Walker().visit(tree)
    return labels


# =========================
# Data extraction (robust to MultiIndex)
# =========================
def _extract_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Return a single-index OHLCV DataFrame with columns:
    ['Open','High','Low','Close','Volume'].
    Works with yfinance single- or multi-index outputs.
    """
    if df is None or df.empty:
        raise KeyError("Empty DataFrame")

    if isinstance(df.columns, pd.MultiIndex):
        cols = {}
        for field in ["Open","High","Low","Close","Adj Close","Volume"]:
            try:
                cols[field] = df[(field, symbol)]
            except Exception:
                pass

        close = None
        if cols.get("Close") is not None:
            close = cols.get("Close")
        elif cols.get("Adj Close") is not None:
            close = cols.get("Adj Close")

        if close is None:
            # fallback: first available across symbols
            for top in ["Adj Close","Close"]:
                try:
                    close = df[top].iloc[:,0]
                    break
                except Exception:
                    pass
        if close is None:
            raise KeyError("Could not locate Close/Adj Close in MultiIndex DataFrame")
        out = pd.DataFrame({
            "Open":  pd.to_numeric(cols.get("Open",  close), errors="coerce"),
            "High":  pd.to_numeric(cols.get("High",  close), errors="coerce"),
            "Low":   pd.to_numeric(cols.get("Low",   close), errors="coerce"),
            "Close": pd.to_numeric(close,            errors="coerce"),
            "Volume":pd.to_numeric(cols.get("Volume", pd.Series(index=df.index, dtype=float)), errors="coerce"),
        }, index=df.index)
    else:
        close = df["Close"] if "Close" in df.columns else df.get("Adj Close")
        if close is None:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                raise KeyError("No numeric columns to use as Close")
            close = df[num_cols[0]]
        out = pd.DataFrame({
            "Open":  pd.to_numeric(df.get("Open",  close), errors="coerce"),
            "High":  pd.to_numeric(df.get("High",  close), errors="coerce"),
            "Low":   pd.to_numeric(df.get("Low",   close), errors="coerce"),
            "Close": pd.to_numeric(close,                  errors="coerce"),
            "Volume":pd.to_numeric(df.get("Volume", pd.Series(index=df.index, dtype=float)), errors="coerce"),
        }, index=df.index)

    out = out.astype(float).dropna()
    if out.empty:
        raise KeyError("OHLCV after cleaning is empty")
    return out

# =========================
# Metrics & plotting
# =========================
def _to_returns(prices: pd.Series) -> pd.Series:
    r = pd.to_numeric(prices, errors="coerce").pct_change()
    r.iloc[0] = 0.0
    return r.fillna(0.0).astype(float)

def _cagr(equity: pd.Series, freq_per_year=252) -> float:
    e = pd.to_numeric(equity, errors="coerce").dropna()
    if len(e) < 2: return 0.0
    total = float(e.iloc[-1] / e.iloc[0] - 1.0)
    years = max(len(e) / float(freq_per_year), 1e-9)
    return (1.0 + total)**(1.0/years) - 1.0

def _sharpe(returns: pd.Series, rf=0.0, freq_per_year=252) -> float:
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    sd = r.std(ddof=0)
    return float((r.mean() - rf/freq_per_year) / sd * math.sqrt(freq_per_year)) if sd > 1e-12 else 0.0

def _max_drawdown(equity: pd.Series) -> float:
    e = pd.to_numeric(equity, errors="coerce").fillna(method="ffill").fillna(method="bfill")
    rollmax = e.cummax()
    dd = e/rollmax - 1.0
    return float(dd.min())

def _plot_equity(equity: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(6,3))
    pd.to_numeric(equity, errors="coerce").plot(ax=ax)
    ax.set_title("Equity Curve"); ax.set_xlabel("Date"); ax.set_ylabel("Equity ($)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig); buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")

# =========================
# Analyzers
# =========================
class EquityTracker(bt.Analyzer):
    def start(self): self.values = []
    def next(self):  self.values.append(self.strategy.broker.getvalue())
    def get_analysis(self): return self.values

class PositionTracker(bt.Analyzer):
    def start(self): self.values = []
    def next(self):  self.values.append(int(self.strategy.position.size))
    def get_analysis(self): return self.values

class SignalTracker(bt.Analyzer):
    def start(self):
        self.enter = []; self.exit = []; self.short = []
    def next(self):
        s = self.strategy
        self.enter.append(bool(getattr(s, "_should_enter")()))
        self.exit.append(bool(getattr(s, "_should_exit")()))
        self.short.append(bool(getattr(s, "_should_short", lambda: False)()))
    def get_analysis(self):
        return {"enter": self.enter, "exit": self.exit, "short": self.short}

# =========================
# IR → Indicator resolution
# =========================
_IND_RE = re.compile(r"^([A-Za-z]+)(?:\(([^)]*)\))?$")  # e.g., SMA(20) -> ("SMA","20")

def _parse_label(label: str) -> Tuple[str, Optional[int]]:
    m = _IND_RE.match(label.strip())
    if not m: return label, None
    name, params = m.groups()
    if not params: return name, None
    try:
        p = int(float(params.split(",")[0]))
        return name, p
    except Exception:
        return name, None

def _mk_indicator(strategy: bt.Strategy, name: str, period: Optional[int], cfg: BtExecConfig):
    p = period or cfg.overrides.get(name) or cfg.default_periods.get(name, 20)
    data = strategy.data
    if name == "SMA":  return bt.ind.SMA(data.close, period=p)
    if name == "EMA":  return bt.ind.EMA(data.close, period=p)
    if name == "RSI":  return bt.ind.RSI(data.close, period=p)
    if name == "ATR":  return bt.ind.ATR(data, period=p)
    if name in ("BollingerBands","BBANDS"):
        return bt.ind.BollingerBands(data.close, period=p)  # returns .top/.mid/.bot
    # fallback
    return bt.ind.SMA(data.close, period=p)

def _series_for_label(strategy, obj_map: Dict[str, Any], label: str):
    l = label.strip()
    if l.lower() == "close": return strategy.data.close
    if l.lower() == "open":  return strategy.data.open
    if l.lower() == "high":  return strategy.data.high
    if l.lower() == "low":   return strategy.data.low

    if l.startswith("BollingerBands."):
        bb = obj_map.get("BollingerBands")
        if bb is None:
            bb = _mk_indicator(strategy, "BollingerBands", None, strategy._ir_cfg)
            obj_map["BollingerBands"] = bb
        comp = l.split(".",1)[1]
        if comp == "top": return bb.top
        if comp in ("mid","middle","midband"): return bb.mid
        if comp in ("bot","lower","low"): return bb.bot
        return bb.mid

    nm, per = _parse_label(l)
    key = nm if per is None else f"{nm}({per})"
    inst = obj_map.get(key)
    if inst is None:
        inst = _mk_indicator(strategy, nm, per, strategy._ir_cfg)
        obj_map[key] = inst
    # IMPORTANT: return the indicator object (behaves like its main line)
    return inst

# =========================
# Condition compiler (sanitize + arithmetic + prev + position)
# =========================
_OP_MAP = {" AND ": " and ", " OR ": " or ", " NOT ": " not "}
_TIGHT_OP_MAP = {" AND":" and","AND ":"and "," OR":" or","OR ":"or "," NOT":" not","NOT ":"not "}
_UNICODE_OPS = {"≥":">=", "≤":"<=", "≠":"!=", "，":",", "：":":"}

def _sanitize_cond_text(s: str) -> str:
    if not s: return s
    t = " " + s.strip() + " "
    for k, v in _OP_MAP.items(): t = t.replace(k, v)
    for k, v in _TIGHT_OP_MAP.items(): t = t.replace(k, v)
    for k, v in _UNICODE_OPS.items(): t = t.replace(k, v)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

_ALLOWED_OPS = (ast.Eq, ast.NotEq, ast.Gt, ast.GtE, ast.Lt, ast.LtE)
_ALLOWED_BOOL = (ast.And, ast.Or)
_ALLOWED_ARITH = (ast.Add, ast.Sub, ast.Mult, ast.Div)

def _attr_to_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name): return node.id
    if isinstance(node, ast.Attribute): return _attr_to_name(node.value) + "." + node.attr
    if isinstance(node, ast.Subscript):
        base = _attr_to_name(node.value)
        try:
            idx = ast.literal_eval(node.slice)
            if isinstance(idx, int) and idx < 0:
                return base + ".__prev__"
        except Exception:
            pass
        return base
    if isinstance(node, ast.Call):
        func = _attr_to_name(node.func)
        period = None
        if node.args:
            a0 = node.args[0]
            try: period = int(float(ast.literal_eval(a0)))
            except Exception: period = None
        return f"{func}({period})" if period is not None else func
    try:
        return ast.unparse(node)
    except Exception:
        return str(node)

def _eval_expr_numeric(node: ast.AST, get_value) -> float:
    if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript, ast.Call)):
        name = _attr_to_name(node)
        prev = False
        if name.endswith(".__prev__"):
            prev = True; name = name[:-9]
        val = get_value(name)
        if isinstance(val, (int, float)):
            return float(val)
        series = val
        try:
            return float(series[-1]) if prev else float(series[0])
        except Exception:
            return float(series[0])

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _eval_expr_numeric(node.operand, get_value)
        return +v if isinstance(node.op, ast.UAdd) else -v

    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_ARITH):
        L = _eval_expr_numeric(node.left, get_value)
        R = _eval_expr_numeric(node.right, get_value)
        if isinstance(node.op, ast.Add):  return L + R
        if isinstance(node.op, ast.Sub):  return L - R
        if isinstance(node.op, ast.Mult): return L * R
        if isinstance(node.op, ast.Div):  return L / (R if abs(R) > 1e-12 else 1e-12)

    return float(_eval_expr_bool(node, get_value))

def _eval_expr_bool(node: ast.AST, get_value) -> bool:
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError("complex compare not supported")
        op = node.ops[0]
        if not isinstance(op, _ALLOWED_OPS): raise ValueError("op not allowed")
        L = _eval_expr_numeric(node.left, get_value)
        R = _eval_expr_numeric(node.comparators[0], get_value)
        if isinstance(op, ast.Eq):   return L == R
        if isinstance(op, ast.NotEq):return L != R
        if isinstance(op, ast.Gt):   return L >  R
        if isinstance(op, ast.GtE):  return L >= R
        if isinstance(op, ast.Lt):   return L <  R
        if isinstance(op, ast.LtE):  return L <= R

    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, _ALLOWED_BOOL): raise ValueError("bool op not allowed")
        if isinstance(node.op, ast.And):
            return all(_eval_expr_bool(v, get_value) for v in node.values)
        else:
            return any(_eval_expr_bool(v, get_value) for v in node.values)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _eval_expr_bool(node.operand, get_value)

    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return bool(node.value)

    return _eval_expr_numeric(node, get_value) > 0.0

def _compile_condition(cond_str: str):
    src = _sanitize_cond_text(cond_str or "")
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as e:
        raise SyntaxError(f"Bad condition syntax: {cond_str!r} -> {e}") from e

    def evaluator(strategy) -> bool:
        obj_map = strategy._ind_objs

        def get_value_by_label(label: str):
            if label == "position":
                return float(getattr(strategy.position, "size", 0.0))
            return _series_for_label(strategy, obj_map, label)

        def adapter(node_or_label):
            if isinstance(node_or_label, ast.AST):
                name = _attr_to_name(node_or_label)
                return get_value_by_label(name)
            return get_value_by_label(node_or_label)

        return _eval_expr_bool(tree.body, adapter)

    return evaluator

# =========================
# Strategy (IR-driven)
# =========================
class IRStrategy(bt.Strategy):
    params = (("ir_obj", None), ("ir_cfg", None),)

    def __init__(self):
        self._ir = self.p.ir_obj or {}
        self._ir_cfg: BtExecConfig = self.p.ir_cfg or BtExecConfig()
        self._ind_objs: Dict[str, Any] = {}

        self._entry_cross = []
        self._exit_cross  = []
        self._short_entry_cross = []
        self._entry_conds = []
        self._exit_conds  = []
        self._short_entry_conds = []

        # Pre-warm indicators so they attach to the data before next()
        needed = set()

        # from explicit IR indicator list (if provided)
        for lbl in self._ir.get("indicators", []):
            needed.add(lbl)

        # from entry/exit rule texts
        for r in (self._ir.get("entry_rules", []) + self._ir.get("exit_rules", [])):
            when = r.get("when", "")
            needed |= _labels_from_cond_text(when)

        # Remove pseudo-variables that aren't indicators
        for bad in ("position",):
            if bad in needed:
                needed.remove(bad)

        # Instantiate once into self._ind_objs
        for lbl in sorted(needed):
            try:
                _ = _series_for_label(self, self._ind_objs, lbl)
            except Exception:
                # Best-effort: ignore unknown labels so one bad token doesn't block the run
                pass

        def build_cross(rule):
            when = (rule.get("when") or "").lower()
            if " crosses above " in when:
                a, b = when.split(" crosses above ")
                s1 = _series_for_label(self, self._ind_objs, a.strip())
                s2 = _series_for_label(self, self._ind_objs, b.strip())
                return ("above", bt.ind.CrossOver(s1, s2))
            if " crosses below " in when:
                a, b = when.split(" crosses below ")
                s1 = _series_for_label(self, self._ind_objs, a.strip())
                s2 = _series_for_label(self, self._ind_objs, b.strip())
                return ("below", bt.ind.CrossOver(s1, s2))
            return None

        def is_short_rule(r: dict) -> bool:
            act = (r.get("action") or "").strip().upper()
            if act in ("SELL","SHORT"): return True
            when = (r.get("when") or "").lower()
            return "short" in when or "sell short" in when

        for r in self._ir.get("entry_rules", []):
            c = build_cross(r)
            if not c: continue
            (self._short_entry_cross if is_short_rule(r) else self._entry_cross).append(c)

        for r in self._ir.get("entry_rules", []):
            w = r.get("when","")
            if "crosses above" in w.lower() or "crosses below" in w.lower():
                continue
            (self._short_entry_conds if is_short_rule(r) else self._entry_conds).append(_compile_condition(w))

        for r in self._ir.get("exit_rules", []):
            w = r.get("when","")
            if "crosses above" in w.lower() or "crosses below" in w.lower():
                c = build_cross(r)
                if c: self._exit_cross.append(c)
            else:
                self._exit_conds.append(_compile_condition(w))

        # Risk
        self._stop_loss = None
        self._take_profit = None
        for rr in self._ir.get("risk_rules", []):
            if "stop_loss" in rr:
                try: self._stop_loss = float(str(rr["stop_loss"]).strip("%"))/100.0
                except Exception: pass
            if "take_profit" in rr:
                try: self._take_profit = float(str(rr["take_profit"]).strip("%"))/100.0
                except Exception: pass

        self._entry_price = None

    def _should_enter(self) -> bool:
        for cond in self._entry_conds:
            try:
                if cond(self): return True
            except Exception: pass
        for direction, co in self._entry_cross:
            if direction == "above" and co[0] > 0: return True
            if direction == "below" and co[0] < 0: return True
        return False

    def _should_short(self) -> bool:
        for cond in self._short_entry_conds:
            try:
                if cond(self): return True
            except Exception: pass
        for direction, co in self._short_entry_cross:
            if direction == "below" and co[0] < 0: return True
            if direction == "above" and co[0] > 0: return True
        return False

    def _should_exit(self) -> bool:
        for cond in self._exit_conds:
            try:
                if cond(self): return True
            except Exception: pass
        for direction, co in self._exit_cross:
            if direction == "above" and co[0] > 0: return True
            if direction == "below" and co[0] < 0: return True
        return False

    def next(self):
        pos = self.position.size

        if pos > 0:  # LONG
            if self._should_exit():
                self.close(); self._entry_price = None; return
            if self._entry_price is not None:
                px = float(self.data.close[0])
                if self._stop_loss is not None and px <= self._entry_price * (1 - self._stop_loss):
                    self.close(); self._entry_price = None; return
                if self._take_profit is not None and px >= self._entry_price * (1 + self._take_profit):
                    self.close(); self._entry_price = None; return

        elif pos < 0:  # SHORT
            if self._should_exit():
                self.close(); self._entry_price = None; return
            if self._entry_price is not None:
                px = float(self.data.close[0])
                if self._stop_loss is not None and px >= self._entry_price * (1 + self._stop_loss):
                    self.close(); self._entry_price = None; return
                if self._take_profit is not None and px <= self._entry_price * (1 - self._take_profit):
                    self.close(); self._entry_price = None; return

        else:  # FLAT
            if self._should_enter():
                self.buy();  self._entry_price = float(self.data.close[0]); return
            if self._ir_cfg.allow_shorts and self._should_short():
                self.sell(); self._entry_price = float(self.data.close[0]); return

# =========================
# Full-notional sizer
# =========================
class FullNotionalSizer(bt.Sizer):
    params = (('target', 1.0),)  # fraction of available cash to deploy

    def _getsizing(self, comminfo, cash, data, isbuy):
        price = float(data.close[0])
        if not np.isfinite(price) or price <= 0:
            return 0
        # invest target% of current cash
        size = int((cash * self.p.target) / price)
        return max(size, 0)

# =========================
# Runner
# =========================
def run_backtrader_ir(strategy_ir: Dict[str,Any], cfg: BtExecConfig) -> Dict[str, Any]:
    # 1) Data
    end = cfg.end or dt.date.today().isoformat()
    try:
        raw = yf.download(cfg.symbol, start=cfg.start, end=end, auto_adjust=True, progress=False)
    except Exception as e:
        return {"error": f"Download failed for {cfg.symbol}: {e}"}
    if raw is None or raw.empty:
        return {"error": f"No data for {cfg.symbol} between {cfg.start} and {end}"}
    try:
        data = _extract_ohlcv(raw, cfg.symbol)
    except Exception as e:
        return {"error": f"Data shape error: {e}"}

    feed = bt.feeds.PandasData(dataname=data)

    # 2) Cerebro
    cerebro = bt.Cerebro()
    cerebro.adddata(feed, name=cfg.symbol)
    cerebro.broker.setcash(cfg.initial_cash)

    commission = float(cfg.commission_bps) / 10_000.0
    if cfg.slippage_bps and cfg.slippage_bps > 0:
        commission += float(cfg.slippage_bps) / 10_000.0
    cerebro.broker.setcommission(commission=commission)

    cerebro.addsizer(FullNotionalSizer, target=1.0)

    cerebro.addstrategy(IRStrategy, ir_obj=strategy_ir, ir_cfg=cfg)

    # analyzers
    cerebro.addanalyzer(EquityTracker, _name="equity")
    cerebro.addanalyzer(PositionTracker, _name="pos")
    cerebro.addanalyzer(SignalTracker, _name="sig")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")

    # 3) Run
    res = cerebro.run(stdstats=False)
    strat = res[0]

    eq_list  = strat.analyzers.equity.get_analysis()
    pos_list = strat.analyzers.pos.get_analysis()
    sigs     = strat.analyzers.sig.get_analysis()

    n = len(eq_list)
    equity_series = pd.Series(eq_list, index=data.index[-n:])
    position_series = pd.Series(pos_list[-n:], index=equity_series.index)

    rets = _to_returns(equity_series)
    cagr = _cagr(equity_series)
    try:
        sr_bt = strat.analyzers.sharpe.get_analysis().get("sharperatio", None)
        sharpe = float(sr_bt) if sr_bt is not None else _sharpe(rets)
    except Exception:
        sharpe = _sharpe(rets)

    try:
        maxdd = float(strat.analyzers.dd.get_analysis().get("max").get("drawdown")/100.0)
    except Exception:
        maxdd = _max_drawdown(equity_series)

    try:
        chart_url = _plot_equity(equity_series)
    except Exception:
        chart_url = None

    out = {
        "symbol": cfg.symbol,
        "start": cfg.start,
        "end": end,
        "initial_cash": float(cfg.initial_cash),
        "final_equity": float(equity_series.iloc[-1]) if len(equity_series) else float(cfg.initial_cash),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_drawdown": float(maxdd),
        "signals": {
            "enter_true": int(sum(1 for x in sigs["enter"] if x)),
            "exit_true":  int(sum(1 for x in sigs["exit"]  if x)),
            "short_true": int(sum(1 for x in sigs["short"] if x)),
        },
        "equity_curve": chart_url,
        "daily": {
            "dates": equity_series.index.strftime("%Y-%m-%d").tolist(),
            "equity": [float(x) for x in equity_series.values],
            "position": [int(x) for x in position_series.values],
            "returns": [float(x) for x in rets.values],
        }
    }
    return out
