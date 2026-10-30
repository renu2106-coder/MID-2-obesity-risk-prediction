"""
Obesity Risk and Level Prediction System
Module: app.py
Description: Flask Application Factory, configuration, blueprint registration,
             and error handling.
"""

import os
import sys
from flask import Flask, jsonify

# Set root directory paths
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
FRONTEND_DIR = os.path.join(PROJECT_ROOT, 'frontend')

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def create_app(config: dict = None) -> Flask:
    """Create and configure the Flask healthcare application instance."""
    app = Flask(
        __name__,
        template_folder=os.path.join(FRONTEND_DIR, 'templates'),
        static_folder=os.path.join(FRONTEND_DIR, 'static')
    )

    # Configuration defaults
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'health-analytics-key-2026')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
    app.config['JSON_SORT_KEYS'] = False

    if config:
        app.config.update(config)

    # Register blueprints
    from backend.routes import api_bp, main_bp
    app.register_blueprint(api_bp)
    app.register_blueprint(main_bp)

    # Global HTTP error handlers
    @app.errorhandler(404)
    def handle_not_found(e):
        return jsonify({
            'success': False,
            'error': 'The requested resource or endpoint was not found.',
            'status_code': 404
        }), 404

    @app.errorhandler(500)
    def handle_server_error(e):
        return jsonify({
            'success': False,
            'error': 'An internal server error occurred. Please verify backend logs.',
            'status_code': 500
        }), 500

    @app.errorhandler(413)
    def handle_file_too_large(e):
        return jsonify({
            'success': False,
            'error': 'File payload exceeds maximum allowed size limit (16MB).',
            'status_code': 413
        }), 413

    return app


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    app = create_app()
    print("=" * 70)
    print(" OBESITY RISK AND LEVEL PREDICTION SYSTEM")
    print(" Healthcare Machine Learning Practical Demonstration")
    print(f" Server starting on: http://127.0.0.1:{port}")
    print("=" * 70)
    app.run(host='0.0.0.0', port=port, debug=debug)

