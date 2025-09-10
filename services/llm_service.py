import os
import json
import hashlib
from dotenv import load_dotenv, find_dotenv
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from services.markdown_service import md_to_safe_html

_ = load_dotenv(find_dotenv())
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_llm = ChatOpenAI(model=MODEL, temperature=0.2, max_tokens=None)

_CACHE: Dict[str, Dict[str, str]] = {}  # key -> {"md":..., "html":...}

# --- Ultra-compact prompt ---
# Keep keys short, avoid chatty prose, push structure.
# Assumes your normalized IR already trims duplicates and size.
_PROMPT = ChatPromptTemplate.from_template(
"""System: You are a precise trading-strategy explainer. Use ONLY the JSON given. No invented metrics or unstated rules.

User JSON:
{j}

Return Markdown with EXACT sections and minimal wording:

# Overview
(4–6 sentences)

## Entry Rules
- bullets

## Exit Rules
- bullets

## Risk Controls
- bullets; say "None detected" if empty

## Market Regimes
- when likely to work/fail (generic, no performance claims)

## Pitfalls
- bullets (overfit, look-ahead, slippage, fill risk, etc.)

## Improvements
- 2–4 bullets
"""
)

def _key(d: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

def explain_strategy_html(strategy_ir: Dict[str, Any]) -> Dict[str, str]:
    k = _key(strategy_ir)
    if k in _CACHE:
        return _CACHE[k]

    # Very compact JSON string (short keys reduce tokens too)
    j = json.dumps(strategy_ir, separators=(",", ":"), ensure_ascii=False)

    msgs = _PROMPT.format_messages(j=j)
    resp = _llm.invoke(msgs)
    md = getattr(resp, "content", None)
    if md is None:
        # Some providers return dict/list messages
        if isinstance(resp, dict) and "content" in resp:
            md = resp["content"]
        else:
            md = str(resp)

    md = (md or "").strip()

    html = md_to_safe_html(md)
    out = {"markdown": md, "html": html}
    _CACHE[k] = out
    return out
