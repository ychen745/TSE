from __future__ import annotations
import datetime
import os
from flask import Blueprint, request, jsonify, render_template, session, flash, redirect, url_for
from flask_login import login_user, logout_user, current_user, login_required
from sqlalchemy.exc import IntegrityError
from werkzeug.security import generate_password_hash, check_password_hash
from app.extensions import db
from app.models import User

auth_bp = Blueprint("auth", __name__, template_folder="../templates")

@auth_bp.get("/login")
def login_page():
    # optional: render a simple form; your SPA can also hit /auth/login JSON
    return render_template("login.html")

@auth_bp.get("/signup")
def signup_page():
    return render_template("signup.html")

@auth_bp.post("/signup")
def signup_post():
    data = request.get_json(silent=True) or request.form
    # email = (data.get("email") or "").strip().lower()
    username = (data.get("username") or "").strip().lower()
    password = data.get("password") or ""
    if not username or not password:
        return _resp({"error": "Username and password are required."}, 400, redirect_to=url_for("auth.signup_page"))

    try:
        user = User(username=username, role="user")
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        return _resp({"error": "Email already registered."}, 409, redirect_to=url_for("auth.signup_page"))

    session["user_id"] = user.id
    session["user_email"] = user.email
    session["user_role"] = user.role
    return _resp({"ok": True, "user": {"id": user.id, "email": user.email, "role": user.role}}, 200, redirect_to=url_for("index"))


@auth_bp.post("/login")
def login_post():
    # admin_username = os.environ.get("ADMIN_USERNAME")
    # admin_password = os.environ.get("ADMIN_PASSWORD")
    #
    # admin_user = User(username=admin_username, role="admin")
    # admin_user.set_password(admin_password)
    # db.session.add(admin_user)
    # db.session.commit()

    data = request.get_json(silent=True) or request.form
    # email = (data.get("email") or "").strip().lower()
    username = (data.get("username") or "").strip().lower()
    password = data.get("password") or ""
    if not username or not password:
        return _resp({"error": "Username and password are required."}, 400)

    user = User.query.filter_by(username=username).first()

    if not user:
        return _resp({"error": "User doesn't exist."}, 401, redirect_to=url_for("auth.login_page"))
    if not user.check_password(password):
        return _resp({"error": "Wrong password"}, 401, redirect_to=url_for("auth.login_page"))

    login_user(user)
    user.last_login = datetime.datetime.now(datetime.timezone.utc)
    db.session.commit()
    flash('Logged in successfully!', 'success')

    session["user_id"] = user.id
    session["username"] = user.username
    session["user_role"] = user.role

    return _resp({"ok": True, "user": {"id": user.id, "email": user.email, "role": user.role}}, 200, redirect_to=url_for("index"))

@auth_bp.post("/logout")
# @login_required
def logout_post():
    session.pop("user_id", None)
    session.pop("username", None)
    session.pop("user_role", None)
    logout_user()
    return _resp({"ok": True}, 200, redirect_to=url_for("index"))

def _resp(payload: dict, status: int, redirect_to: str | None = None):
    # If the client sent JSON, reply JSON. Otherwise, do a browser redirect.
    if request.is_json:
        return jsonify(payload), status
    if "error" in payload:
        flash(payload["error"], "error")
    return redirect(redirect_to or url_for("index"))
