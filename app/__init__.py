import os
from flask import Flask, render_template, g, session
from app.routes.analyze_routes import analyze_bp
from app.routes.backtest_routes import backtest_bp
from app.routes.report_route import report_bp
from app.routes.upload_routes import upload_bp
from .config import Config
from .extensions import db, migrate, login_manager

# to avoid multithreading issue with MacOS matploglib backend
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)

def create_app():
    app = Flask(
        __name__,
        static_folder='static',
        template_folder='templates'
    )

    app.config.setdefault("SECRET_KEY", os.getenv("SECRET_KEY", "dev-secret-change-me"))
    app.config.setdefault("SQLALCHEMY_DATABASE_URI", os.getenv("DATABASE_URL", "sqlite:///tse.db"))
    app.config.setdefault("SQLALCHEMY_TRACK_MODIFICATIONS", False)

    app.url_map.strict_slashes = False
    app.config.from_object(Config)

    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        from app.models import User
        return User.query.get(int(user_id))

    # os.makedirs(app.config.get("UPLOAD_FOLDER", "uploads"), exist_ok=True)
    # os.makedirs(app.config.get("RESULTS_FOLDER", "results"), exist_ok=True)
    # os.makedirs(app.config.get("BACKTEST_RESULTS_FOLDER", "bt_results"), exist_ok=True)

    # register blueprints
    from .routes.auth_routes import auth_bp
    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(upload_bp, url_prefix='/upload')
    app.register_blueprint(analyze_bp, url_prefix='/analyze')
    app.register_blueprint(backtest_bp, url_prefix='/backtest')
    app.register_blueprint(report_bp, url_prefix='/report')

    from .cli import init_cli
    init_cli(app)

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route("/healthz")
    def healthz():
        return {"status": "ok"}, 200

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=8080)

    # for rule in app.url_map.iter_rules():
    #     print("ROUTE:", rule.rule, "->", list(rule.methods))
