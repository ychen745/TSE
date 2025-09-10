import os
from flask import Flask, render_template
from routes.upload_routes import upload_bp
from routes.analyze_routes import analyze_bp
from routes.backtest_routes import backtest_bp
from routes.report_route import report_bp

# to avoid multithreading issue with MacOS matploglib backend
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)

def create_app():
    app = Flask(__name__)
    app.url_map.strict_slashes = False
    app.config.from_object('config.Config')

    # register blueprints
    app.register_blueprint(upload_bp, url_prefix='/upload')
    app.register_blueprint(analyze_bp, url_prefix='/analyze')
    app.register_blueprint(backtest_bp, url_prefix='/backtest')
    app.register_blueprint(report_bp, url_prefix='/report')

    @app.route('/')
    def index():
        return render_template('index.html')

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=8080)

    # for rule in app.url_map.iter_rules():
    #     print("ROUTE:", rule.rule, "->", list(rule.methods))
