"""
Launcher for Obesity Risk and Level Prediction System
Usage: python run.py
"""

import os
import sys

# Add project root and src to path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from backend.app import create_app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app = create_app()
    print("=" * 75)
    print("  OBESITY RISK AND LEVEL PREDICTION SYSTEM")
    print("  Healthcare Machine Learning Practical Demonstration")
    print("=" * 75)
    print(f"  Access Web Dashboard at: http://127.0.0.1:{port}")
    print("=" * 75)
    app.run(host='127.0.0.1', port=port, debug=debug)

