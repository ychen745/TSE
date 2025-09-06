from flask import Flask
from routes.upload_routes import upload_bp
from routes.analyze_routes import analyze_bp

def create_app():
    app = Flask(__name__)
    app.config.from_object('config.Config')

    # register blueprints
    app.register_blueprint(upload_bp, url_prefix='/upload')
    app.register_blueprint(analyze_bp, url_prefix='/analyze')

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5050)

