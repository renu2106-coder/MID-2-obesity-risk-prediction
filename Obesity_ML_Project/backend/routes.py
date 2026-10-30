"""
Obesity Risk and Level Prediction System
Module: routes.py
Description: Flask blueprint and REST API endpoints for dataset management,
             preprocessing, EDA, model training, evaluation, and patient inference.
"""

import os
import sys
import json
from typing import Dict, Any
from flask import Blueprint, request, jsonify, render_template, send_from_directory, current_app
from werkzeug.utils import secure_filename

# Ensure local src directory is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_loader import (
    load_dataset,
    get_dataset_summary,
    get_default_dataset_path,
    SYNTHETIC_DATA_STATS
)
from data_preprocessing import get_preprocessing_report
from eda import generate_eda_visualizations
from model_training import train_and_save_all_models
from model_evaluation import evaluate_models
from prediction import predict_obesity_level, is_model_trained

api_bp = Blueprint('api', __name__, url_prefix='/api')
main_bp = Blueprint('main', __name__)

ALLOWED_EXTENSIONS = {'csv'}


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --- UI Pages Route ---
@main_bp.route('/')
def index():
    return render_template('index.html')


# --- Static Reports Images Serving ---
@main_bp.route('/reports/eda/<path:filename>')
def serve_eda_report(filename):
    eda_dir = os.path.join(PROJECT_ROOT, 'reports', 'eda')
    return send_from_directory(eda_dir, filename)


@main_bp.route('/reports/evaluation/<path:filename>')
def serve_eval_report(filename):
    eval_dir = os.path.join(PROJECT_ROOT, 'reports', 'evaluation')
    return send_from_directory(eval_dir, filename)


# --- REST API Endpoints ---

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint reporting API and artifact availability."""
    dataset_exists = os.path.exists(get_default_dataset_path())
    model_trained = is_model_trained()
    return jsonify({
        'status': 'healthy',
        'application': 'Obesity Risk and Level Prediction System',
        'purpose': 'Healthcare Machine Learning Practical Demonstration',
        'environment': 'Production-Ready Educational API',
        'dataset_available': dataset_exists,
        'model_trained': model_trained,
        'python_version': sys.version.split()[0]
    }), 200


@api_bp.route('/dataset-summary', methods=['GET'])
def dataset_summary():
    """Return dataset overview, dimensions, column types, and distributions."""
    try:
        summary = get_dataset_summary()
        return jsonify({'success': True, 'data': summary}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': f"Failed to load dataset summary: {str(e)}"}), 500


@api_bp.route('/upload-dataset', methods=['POST'])
def upload_dataset():
    """Upload and validate a new dataset CSV file."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part in request.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file format. Only CSV files (.csv) are accepted.'}), 400

    try:
        filename = secure_filename(file.filename)
        dest_path = os.path.join(PROJECT_ROOT, 'dataset', 'ObesityDataSet.csv')
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        file.save(dest_path)

        # Validate schema by attempting to load
        summary = get_dataset_summary(file_path=dest_path)
        return jsonify({
            'success': True,
            'message': f"Dataset '{filename}' successfully uploaded and validated.",
            'data': summary
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': f"Uploaded CSV validation failed: {str(e)}"}), 400


@api_bp.route('/preprocessing-report', methods=['GET'])
def preprocessing_report():
    """Return data quality report before and after preprocessing."""
    try:
        report = get_preprocessing_report()
        return jsonify({'success': True, 'data': report}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': f"Failed to generate preprocessing report: {str(e)}"}), 500


@api_bp.route('/eda-data', methods=['GET'])
def eda_data():
    """Return 10 EDA visualizations with metadata and clinical interpretations."""
    try:
        eda_results = generate_eda_visualizations()
        # Ensure image paths are relative URLs for web client
        for chart in eda_results['charts']:
            chart['image_url'] = f"/reports/eda/{chart['file']}"
        return jsonify({'success': True, 'data': eda_results}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': f"Failed to retrieve EDA data: {str(e)}"}), 500


@api_bp.route('/train-model', methods=['POST'])
def train_model():
    """Execute model training with stratified CV and hyperparameter tuning."""
    try:
        metadata = train_and_save_all_models()
        # Also run evaluation immediately after training to update evaluation artifacts
        evaluate_models()
        return jsonify({
            'success': True,
            'message': 'Model training and validation completed successfully.',
            'data': metadata
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': f"Model training failed: {str(e)}"}), 500


@api_bp.route('/model-metrics', methods=['GET'])
def model_metrics():
    """Return comprehensive evaluation metrics and comparison table."""
    try:
        eval_json_path = os.path.join(PROJECT_ROOT, 'reports', 'evaluation', 'evaluation_metrics.json')
        if os.path.exists(eval_json_path):
            with open(eval_json_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
        else:
            report = evaluate_models()

        # Add web URLs for evaluation plots
        report['image_urls'] = {
            'confusion_matrix': '/reports/evaluation/confusion_matrix.png',
            'roc_curves': '/reports/evaluation/roc_curves.png',
            'pr_curves': '/reports/evaluation/pr_curves.png'
        }

        return jsonify({'success': True, 'data': report}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': f"Failed to retrieve model metrics: {str(e)}"}), 500


@api_bp.route('/predict', methods=['POST'])
def predict():
    """Process patient input, validate, compute BMI, and execute ML inference."""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'success': False, 'error': 'Missing JSON payload.'}), 400

        result = predict_obesity_level(data)
        if not result['success']:
            return jsonify(result), 400

        return jsonify(result), 200
    except Exception as e:
        return jsonify({'success': False, 'error': f"Prediction error: {str(e)}"}), 500


@api_bp.route('/reports/download/<report_type>', methods=['GET'])
def download_report(report_type):
    """Generate and return downloadable reports in JSON format."""
    try:
        if report_type == 'evaluation':
            eval_path = os.path.join(PROJECT_ROOT, 'reports', 'evaluation', 'evaluation_metrics.json')
            if os.path.exists(eval_path):
                with open(eval_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return jsonify({'report_type': 'Model Evaluation Report', 'data': data})
            return jsonify({'error': 'Evaluation report not yet generated.'}), 404

        elif report_type == 'quality':
            report = get_preprocessing_report()
            return jsonify({'report_type': 'Dataset Quality & Preprocessing Report', 'data': report})

        elif report_type == 'dataset':
            summary = get_dataset_summary()
            return jsonify({'report_type': 'Dataset Summary Report', 'data': summary})

        else:
            return jsonify({'error': f"Unknown report type: '{report_type}'."}), 400
    except Exception as e:
        return jsonify({'error': f"Error generating report: {str(e)}"}), 500

