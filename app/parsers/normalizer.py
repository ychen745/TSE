# parsers/normalizer.py
# Strict-allowlist normalizer utilities for trading strategies (Backtrader-centric).
# Converts AST nodes into a compact, framework-agnostic IR.

from __future__ import annotations
import ast
from typing import Dict, List, Tuple, Optional, Any

# =========================
# Configuration (strict allowlist + action blacklist)
# =========================
ACTION_CALLS = {
    "buy", "sell",
    "order_target_percent", "order_target_size", "order_target_value",
    "order_target", "cancel", "closeout"
}

# Common indicator names (Backtrader, TA-Lib/pandas-ta style). Extend as you see fit.
INDICATOR_NAMES = {
    # Moving averages
    "SMA", "EMA", "WMA", "HMA", "KAMA", "T3", "DEMA", "TEMA",
    # Momentum / oscillators
    "RSI", "MACD", "Stochastic", "MOM", "ROC", "CCI", "ADX", "RVI",
    # Volatility / bands
    "ATR", "BollingerBands", "BBANDS",
    # Volume-based
    "OBV", "ChaikinMoneyFlow",
    # Other
    "STDDEV",
}

DATA_STATE_LABELS = {
    "open", "close", "high", "low", "volume",
    "position", "position.size", "entry_price"
}

# Namespaces we will accept as indicator constructors (strict allowlist).
# E.g., bt.ind.SMA, bt.indicators.SMA, talib.RSI, ta.rsi (lowercase endings handled by name check).
INDICATOR_NAMESPACE_PREFIXES = [
    ("bt", "ind"),
    ("bt", "indicators"),
    ("talib",),
    ("ta",),
]

# =========================
# Helpers
# =========================
def _label_base(label: str) -> str:
    # "BollingerBands.top" -> "BollingerBands"
    # "SMA(20)" -> "SMA"
    base = label.split("(")[0]
    if "." in base:
        base = base.split(".")[0]
    return base

def is_indicator_label(label: str) -> bool:
    """True if this normalized label looks like an indicator, not data/state."""
    if not label:
        return False
    if label in DATA_STATE_LABELS:
        return False
    base = _label_base(label)
    return base in INDICATOR_NAMES

def _pretty_attr(node: ast.AST) -> str:
    """Turn Attribute/Subscript chains into a dotted name, dropping indices."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_pretty_attr(node.value)}.{node.attr}"
    if isinstance(node, ast.Subscript):
        return _pretty_attr(node.value)
    return ast.unparse(node)  # Py3.9+

def _func_name(call: ast.Call) -> str:
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return ast.unparse(f)

def _call_name_lower(call: ast.Call) -> str:
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr.lower()
    if isinstance(f, ast.Name):
        return f.id.lower()
    return ast.unparse(f).split("(")[0].strip().lower()

def _attr_chain(node: ast.AST) -> Tuple[str, ...]:
    """Collect attribute chain as tuple: bt.ind.SMA -> ('bt','ind','SMA')"""
    out = []
    while isinstance(node, ast.Attribute):
        out.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        out.append(node.id)
    return tuple(reversed(out))

def _kw_or_pos_period(call: ast.Call) -> Optional[str]:
    """Try to get a 'period'-like parameter for an indicator constructor."""
    for kw in call.keywords or []:
        if kw.arg in {"period", "timeperiod", "n", "length", "window"}:
            try:
                return ast.unparse(kw.value)
            except Exception:
                return None
    # crude positional fallback: first numeric literal
    for a in call.args or []:
        if isinstance(a, ast.Constant) and isinstance(a.value, (int, float)):
            return str(a.value)
    return None

def normalize_indicator_call(call: ast.Call) -> str:
    """Return a canonical indicator label, e.g., 'SMA(20)' or 'RSI(14)'."""
    name = _func_name(call)
    period = _kw_or_pos_period(call)
    return f"{name}({period})" if period else name

def is_indicator_call(call: ast.Call) -> bool:
    """Strict detection: namespace or terminal name must be allowlisted; actions excluded."""
    nm = _call_name_lower(call)
    if nm in ACTION_CALLS:
        return False

    # name match (case-sensitive for terminal names in allowlist)
    if isinstance(call.func, ast.Name):
        return call.func.id in INDICATOR_NAMES

    # attribute chain: namespace or terminal must match allowlist
    if isinstance(call.func, ast.Attribute):
        chain = _attr_chain(call.func)
        for pref in INDICATOR_NAMESPACE_PREFIXES:
            if chain[:len(pref)] == pref:
                # still require the terminal to look like a known indicator name (strict)
                return chain[-1] in INDICATOR_NAMES
        return chain and chain[-1] in INDICATOR_NAMES

    return False

# =========================
# Normalization of expressions
# =========================
def normalize_expr(node: ast.AST, symtab: Dict[str, str]) -> str:
    """Human-readable condition string using symtab mapping."""
    if isinstance(node, ast.Compare):
        left = normalize_expr(node.left, symtab)
        rights = [normalize_expr(c, symtab) for c in node.comparators]
        ops = [type(op).__name__ for op in node.ops]
        op_map = {"Gt": ">", "Lt": "<", "Eq": "==", "GtE": ">=", "LtE": "<="}
        s = left
        for op, r in zip(ops, rights):
            s += f" {op_map.get(op, '?')} {r}"
        return s

    if isinstance(node, ast.BoolOp):
        join = " AND " if isinstance(node.op, ast.And) else " OR "
        return join.join(normalize_expr(v, symtab) for v in node.values)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return f"NOT ({normalize_expr(node.operand, symtab)})"

    if isinstance(node, (ast.Attribute, ast.Name, ast.Subscript)):
        name = _pretty_attr(node)
        return symtab.get(name, name)

    if isinstance(node, ast.Constant):
        return str(node.value)

    # Fallback to python 3.9+ unparse
    return ast.unparse(node)

# =========================
# Symtab builder (Backtrader)
# =========================
def build_backtrader_symtab(tree: ast.AST) -> Dict[str, str]:
    """
    Scan __init__ for assignments like:
      self.sma = bt.ind.SMA(self.data.close, period=20)
    Keep only strict-allowlist indicator constructors.
    """
    symtab: Dict[str, str] = {}

    class InitVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef):
            if node.name != "__init__":
                return
            for stmt in node.body:
                # Support simple "self.xxx = Call(...)"
                if isinstance(stmt, ast.Assign) and stmt.targets and isinstance(stmt.targets[0], ast.Attribute):
                    target = stmt.targets[0]
                    if isinstance(stmt.value, ast.Call) and is_indicator_call(stmt.value):
                        tname = _pretty_attr(target)  # e.g., self.sma
                        label = normalize_indicator_call(stmt.value)  # e.g., SMA(20)
                        symtab[tname] = label

                        # Special-casing common line components (Bollinger)
                        if label.startswith("BollingerBands") or label.startswith("BBANDS"):
                            symtab[f"{tname}.lines.top"] = "BollingerBands.top"
                            symtab[f"{tname}.lines.mid"] = "BollingerBands.mid"
                            symtab[f"{tname}.lines.bot"] = "BollingerBands.bot"
            self.generic_visit(node)

    InitVisitor().visit(tree)

    # Common data + state aliases
    symtab.setdefault("self.data.close", "close")
    symtab.setdefault("self.datas[0].close", "close")
    symtab.setdefault("self.close", "close")
    symtab.setdefault("self.position", "position")
    symtab.setdefault("self.position.size", "position.size")
    symtab.setdefault("self.buyprice", "entry_price")
    return symtab

# =========================
# Pattern detectors
# =========================
def _cmp_tuple(c: ast.Compare, symtab: Dict[str, str]) -> Optional[Tuple[str, str, str]]:
    """Return (left, op, right) as strings if simple compare."""
    if len(c.ops) != 1 or len(c.comparators) != 1:
        return None
    op_map = {ast.Gt: ">", ast.Lt: "<", ast.GtE: ">=", ast.LtE: "<=", ast.Eq: "==",}
    op = op_map.get(type(c.ops[0]))
    if not op:
        return None
    L = normalize_expr(c.left, symtab)
    R = normalize_expr(c.comparators[0], symtab)
    return (L, op, R)

def detect_crossover_from_if(node_if: ast.If, symtab: Dict[str, str]) -> Optional[str]:
    """
    Heuristic: (A > B AND prevA <= prevB) => 'A crosses above B'
               (A < B AND prevA >= prevB) => 'A crosses below B'
    """
    terms: List[ast.AST] = []
    def collect(n: ast.AST):
        if isinstance(n, ast.BoolOp) and isinstance(n.op, ast.And):
            for v in n.values: collect(v)
        else:
            terms.append(n)
    collect(node_if.test)

    curr, prev = [], []
    for t in terms:
        if not isinstance(t, ast.Compare):
            continue
        txt = ast.unparse(t)
        tup = _cmp_tuple(t, symtab)
        if not tup:
            continue
        L, op, R = tup
        if "[-1]" in txt:
            prev.append((L.replace("[-1]", ""), op, R.replace("[-1]", "")))
        else:
            curr.append((L, op, R))

    for (L1, op1, R1) in curr:
        for (L0, op0, R0) in prev:
            if L1 == L0 and R1 == R0:
                if op1 in (">", ">=") and op0 in ("<", "<="):
                    return f"{L1} crosses above {R1}"
                if op1 in ("<", "<=") and op0 in (">", ">="):
                    return f"{L1} crosses below {R1}"
    return None

def _extract_percent_from_mul(node: ast.AST) -> Optional[float]:
    """
    Detect entry_price * 0.98  -> 0.02
           entry_price * (1 - 0.02) -> 0.02
           entry_price * (1 + 0.05) -> 0.05
    Very heuristic but robust for common patterns.
    """
    txt = ast.unparse(node)
    try:
        # case: * <float>
        if "*" in txt:
            k = float(txt.split("*")[-1].strip().strip("()"))
            return abs(1.0 - k)
    except Exception:
        pass
    if "1 -" in txt or "1+" in txt or "1 +" in txt:
        try:
            # naive extraction of the number after "1 +/-"
            tail = txt.split("1", 1)[1]
            num = tail.replace("-", " ").replace("+", " ").replace(")", " ").strip()
            return abs(float(num))
        except Exception:
            return None
    return None

def detect_stop_take_from_if(node_if: ast.If, symtab: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Find:
      close <= entry_price * (1 - x) -> {"stop_loss": "x%"}
      close >= entry_price * (1 + x) -> {"take_profit": "x%"}
    """
    found: List[Dict[str, str]] = []

    def scan(n: ast.AST):
        if isinstance(n, ast.BoolOp):
            for v in n.values: scan(v)
        elif isinstance(n, ast.Compare):
            tup = _cmp_tuple(n, symtab)
            if not tup:
                return
            L, op, R = tup
            left_is_close = "close" in L.lower()
            right_is_close = "close" in R.lower()

            # which side carries multiplier?
            pct = _extract_percent_from_mul(n.comparators[0]) if "entry_price" in R else _extract_percent_from_mul(n.left)
            if pct is None:
                return

            pct_str = f"{round(pct*100, 4)}%"
            # stop-loss
            if (left_is_close and op in ("<", "<=") and "entry_price" in R) or \
               (right_is_close and op in (">", ">=") and "entry_price" in L):
                found.append({"stop_loss": pct_str, "basis": "entry_price"})
            # take-profit
            if (left_is_close and op in (">", ">=") and "entry_price" in R) or \
               (right_is_close and op in ("<", "<=") and "entry_price" in L):
                found.append({"take_profit": pct_str, "basis": "entry_price"})

    scan(node_if.test)
    return found

def if_body_actions(node_if: ast.If) -> List[str]:
    """Collect trade action calls inside the if-body."""
    calls = []
    for n in node_if.body:
        for sub in ast.walk(n):
            if isinstance(sub, ast.Call):
                nm = _call_name_lower(sub)
                if nm in ACTION_CALLS:
                    calls.append(nm)
    return calls
