from __future__ import annotations

import os
from flask import Blueprint, request, jsonify, current_app

from app.parsers.strategy_parser import parse_strategy

# If you're using the server-side renderer version:
#   explain_strategy_html -> returns {"markdown": "...", "html": "..."}
try:
    from app.services.llm_service import explain_strategy_html as _explain
except Exception:
    try:
        from app.services.llm_service import explain_strategy_md as _explain
    except Exception:
        _explain = None


analyze_bp = Blueprint("analyze", __name__)

# ---- config knobs (can also live in config.py) ----
MAX_CODE_CHARS = int(os.getenv("MAX_CODE_CHARS", "200000"))  # 200k chars hard cap


def _get_code_from_request() -> str | None:
    """
    Accept JSON body: {"code": "..."} (preferred),
    or form body: code=<...> (fallback for simple HTML forms).
    """
    code = None
    if request.is_json:
        data = request.get_json(silent=True) or {}
        code = (data.get("code") or "").strip()
    else:
        # allow form-encoded fallback
        code = (request.form.get("code") or "").strip()
    return code or None


def _validate_code(code: str) -> tuple[bool, str | None]:
    if not code:
        return False, "No code provided."
    if len(code) > MAX_CODE_CHARS:
        return False, f"Code too large ({len(code)} chars > {MAX_CODE_CHARS})."
    return True, None


def _json_error(message: str, status: int = 400):
    # Always respond JSON so the frontend can res.json() safely.
    return jsonify({"error": message}), status


# ---------- Analyze: parser only ----------
@analyze_bp.route("", methods=["POST"])       # no trailing slash
@analyze_bp.route("/", methods=["POST"])      # with trailing slash
def analyze_code():
    code = _get_code_from_request()
    ok, err = _validate_code(code or "")
    if not ok:
        return _json_error(err, 400)

    try:
        parsed_ir = parse_strategy(code)
        return jsonify({"parsed_logic": parsed_ir}), 200
    except Exception as e:
        # Log full stack to server logs; return safe JSON error to client
        current_app.logger.exception("Analyze failed")
        return _json_error(f"Analyze failed: {e.__class__.__name__}", 500)


# ---------- Explain: parser + LLM ----------
@analyze_bp.route("/explain", methods=["POST"])   # no trailing slash
@analyze_bp.route("/explain/", methods=["POST"])  # with trailing slash
def analyze_and_explain():
    if _explain is None:
        return jsonify({"error": "LLM service not available."}), 501

    code = _get_code_from_request()
    ok, err = _validate_code(code or "")
    if not ok:
        return _json_error(err, 400)

    try:
        ir = parse_strategy(code)
    except Exception as e:
        current_app.logger.exception("Parse (pre-LLM) failed")
        return _json_error(f"Analyze failed: {e.__class__.__name__}", 500)

    try:
        rendered = _explain(ir)  # {"markdown": "...", "html": "..."}

        if isinstance(rendered, str):
            md = rendered
            try:
                from app.services.llm_service import md_to_safe_html
                html = md_to_safe_html(md)
            except Exception:
                html = ""
            payload = {"markdown": md, "html": html}
        elif isinstance(rendered, dict):
            payload = {"markdown": rendered.get("markdown", ""),
                       "html": rendered.get("html", "")}
        else:
            payload = {"markdown": "", "html": ""}

        return jsonify({"parsed_logic": ir, **payload}), 200

    except Exception as e:
        current_app.logger.exception("LLM explain failed")
        # 502 Bad Gateway is a reasonable choice for upstream/model failures
        return _json_error(f"Explain failed: {e.__class__.__name__}", 502)

