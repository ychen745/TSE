# parsers/strategy_parser.py
# Builds a compact, framework-agnostic IR for a strategy using strict-allowlist detection.

from __future__ import annotations
import ast
from typing import Dict, Any, Set, List

from .normalizer import (
    build_backtrader_symtab,
    is_indicator_call,
    normalize_indicator_call,
    normalize_expr,
    detect_crossover_from_if,
    detect_stop_take_from_if,
    if_body_actions,
    is_indicator_label
)

MAX_INDICATORS = 20
MAX_RULES = 40
MAX_RISK = 20

def _dedupe(seq):
    return list(dict.fromkeys(seq))

def parse_strategy(code: str) -> Dict[str, Any]:
    """
    Strict-allowlist parser:
      - Detect indicators created in Backtrader __init__ (primary source)
      - Optionally include other allowlisted indicator calls
      - Extract entry/exit/risk rules from If bodies in next()
      - Normalize conditions via symtab to human-readable strings
    """
    tree = ast.parse(code)

    # 1) Build Backtrader symtab (maps self.xxx -> 'SMA(20)' etc.)
    symtab = build_backtrader_symtab(tree)

    # 2) Collect indicators
    indicators: Set[str] = set()  # from __init__ (preferred)
    for k, v in symtab.items():
        if is_indicator_label(v):
            indicators.add(v)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and is_indicator_call(node):
            indicators.add(normalize_indicator_call(node))

    # 3) Walk If statements and classify rules
    entry_rules: List[Dict[str, Any]] = []
    exit_rules: List[Dict[str, Any]] = []
    risk_rules: List[Dict[str, Any]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            # Prefer crossover label if pattern found; else normalized condition
            cross = detect_crossover_from_if(node, symtab)
            cond_str = cross if cross else normalize_expr(node.test, symtab)

            # Detect risk controls
            risks = detect_stop_take_from_if(node, symtab)
            if risks:
                risk_rules.extend(risks)

            actions = [a.lower() for a in if_body_actions(node)]
            if any(a in ("buy", "order_target_percent", "order_target_size", "order_target_value", "order_target") for a in actions):
                entry_rules.append({"when": cond_str, "action": "BUY"})
            if any(a in ("sell", "close") for a in actions):
                exit_rules.append({"when": cond_str, "action": "SELL"})

    # 4) Trim / dedupe for token/latency control
    indicators = _dedupe(indicators)[:MAX_INDICATORS]
    entry_rules = entry_rules[:MAX_RULES]
    exit_rules = exit_rules[:MAX_RULES]
    risk_rules = risk_rules[:MAX_RISK]

    # 5) Heuristic framework tag (lightweight)
    framework = "backtrader" if "self.position" in symtab or any("bt." in n for n in _dedupe(indicators)) else "unknown"

    # 6) Final IR
    ir: Dict[str, Any] = {
        "framework": framework,
        "indicators": indicators,
        "entry_rules": entry_rules,
        "exit_rules": exit_rules,
        "risk_rules": risk_rules,
        "meta": {}
    }
    return ir
