"""
Obesity Risk and Level Prediction System
Module: model_evaluation.py
Description: Evaluates trained models on held-out test data. Computes Accuracy, Precision, Recall,
             F1-score, ROC-AUC, generates Confusion Matrix, OvR ROC curves, PR curves,
             and provides plain-language metric interpretations.
"""

import os
import sys
import json
from typing import Dict, Any, List
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve,
    classification_report
)
from sklearn.preprocessing import label_binarize

# Ensure local src directory is on sys.path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from .data_preprocessing import preprocess_data
    from .data_loader import TARGET_CLASSES
except (ImportError, ValueError):
    from data_preprocessing import preprocess_data
    from data_loader import TARGET_CLASSES

MODELS_DIR = os.path.join(os.path.dirname(SRC_DIR), 'models')
EVAL_DIR = os.path.join(os.path.dirname(SRC_DIR), 'reports', 'evaluation')
os.makedirs(EVAL_DIR, exist_ok=True)

PRIMARY_BLUE = "#163B65"
SECONDARY_BLUE = "#2878B5"
ACCENT_TEAL = "#159A9C"

METRIC_EXPLANATIONS = {
    'accuracy': (
        "Accuracy measures the overall fraction of correct predictions across all obesity categories. "
        "Formula: Correct Predictions / Total Predictions. While intuitive, it can be deceptive if classes "
        "are imbalanced."
    ),
    'precision_macro': (
        "Macro Precision measures the average purity of positive predictions across each class independently, "
        "giving equal weight to rare and frequent categories. Answering: 'When the model predicts a specific obesity "
        "tier, how often is it truly correct?'"
    ),
    'precision_weighted': (
        "Weighted Precision calculates the average class precision, weighted by the prevalence of each class "
        "in the test set."
    ),
    'recall_macro': (
        "Macro Recall measures the model's ability to detect all individuals belonging to each specific obesity class. "
        "Answering: 'Out of all individuals who truly belong to an obesity category, what percentage did the model find?'"
    ),
    'recall_weighted': (
        "Weighted Recall computes the sensitivity averaged according to each class's sample support."
    ),
    'f1_macro': (
        "Macro F1-Score is the harmonic mean of precision and recall computed equally across all 7 categories. "
        "It serves as the gold-standard validation criterion in multiclass healthcare evaluation because it penalizes "
        "poor performance on any single clinical tier."
    ),
    'f1_weighted': (
        "Weighted F1-Score is the prevalence-weighted harmonic mean of precision and recall."
    ),
    'roc_auc_ovr_macro': (
        "Multiclass ROC-AUC (One-vs-Rest) evaluates how effectively the model discriminates each obesity category "
        "from the other six classes across all possible probability thresholds. 1.0 represents perfect discrimination."
    )
}


def evaluate_single_pipeline(
    pipeline,
    X_test_raw: pd.DataFrame,
    y_test: np.ndarray,
    target_classes: List[str]
) -> Dict[str, Any]:
    """
    Compute full suite of classification metrics for a single pipeline on held-out test data.
    """
    y_pred = pipeline.predict(X_test_raw)
    y_proba = pipeline.predict_proba(X_test_raw)

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    prec_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_mac = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_wt = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Multiclass ROC-AUC (OvR)
    n_classes = len(target_classes)
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
    try:
        roc_auc_macro = roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='macro')
        roc_auc_weighted = roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='weighted')
    except Exception:
        roc_auc_macro = None
        roc_auc_weighted = None

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Per-class classification report
    report_dict = classification_report(
        y_test, y_pred,
        target_names=target_classes,
        output_dict=True,
        zero_division=0
    )

    return {
        'accuracy': round(float(acc), 4),
        'precision_macro': round(float(prec_macro), 4),
        'precision_weighted': round(float(prec_weighted), 4),
        'recall_macro': round(float(rec_macro), 4),
        'recall_weighted': round(float(rec_weighted), 4),
        'f1_macro': round(float(f1_mac), 4),
        'f1_weighted': round(float(f1_wt), 4),
        'roc_auc_macro': round(float(roc_auc_macro), 4) if roc_auc_macro is not None else "N/A",
        'roc_auc_weighted': round(float(roc_auc_weighted), 4) if roc_auc_weighted is not None else "N/A",
        'confusion_matrix': cm.tolist(),
        'per_class_report': report_dict,
        'y_pred': y_pred.tolist(),
        'y_proba': y_proba.tolist()
    }


def generate_evaluation_visualizations(
    best_name: str,
    cm: np.ndarray,
    y_test: np.ndarray,
    y_proba: np.ndarray,
    target_classes: List[str]
) -> Dict[str, str]:
    """
    Generate Confusion Matrix, One-vs-Rest ROC curves, and Precision-Recall curves.
    """
    clean_labels = [c.replace('_', ' ') for c in target_classes]
    n_classes = len(target_classes)
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))

    # 1. Confusion Matrix
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clean_labels, yticklabels=clean_labels, ax=ax)
    ax.set_title(f"Confusion Matrix: {best_name} (Test Set)", fontsize=13, fontweight='bold', color=PRIMARY_BLUE)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight='bold')
    ax.set_ylabel("True Class", fontsize=11, fontweight='bold')
    plt.xticks(rotation=30, ha='right')
    fig.tight_layout()
    cm_path = os.path.join(EVAL_DIR, 'confusion_matrix.png')
    fig.savefig(cm_path, dpi=200)
    plt.close(fig)

    # 2. Multiclass One-vs-Rest ROC Curves
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#2878B5", "#159A9C", "#48A9A6", "#E3B23C", "#E27D60", "#C34A36", "#8E2800"]
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        try:
            auc_val = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
            label_text = f"{clean_labels[i]} (AUC = {auc_val:.3f})"
        except Exception:
            label_text = f"{clean_labels[i]}"
        ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2, label=label_text)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='Chance Line (AUC = 0.50)')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight='bold')
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight='bold')
    ax.set_title(f"One-vs-Rest ROC Curves: {best_name}", fontsize=13, fontweight='bold', color=PRIMARY_BLUE)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)
    fig.tight_layout()
    roc_path = os.path.join(EVAL_DIR, 'roc_curves.png')
    fig.savefig(roc_path, dpi=200)
    plt.close(fig)

    # 3. Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(9, 7))
    for i in range(n_classes):
        prec, rec, _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
        ax.plot(rec, prec, color=colors[i % len(colors)], linewidth=2, label=clean_labels[i])

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight='bold')
    ax.set_title(f"Precision-Recall Curves: {best_name}", fontsize=13, fontweight='bold', color=PRIMARY_BLUE)
    ax.legend(loc="lower left", fontsize=8.5, frameon=True)
    fig.tight_layout()
    pr_path = os.path.join(EVAL_DIR, 'pr_curves.png')
    fig.savefig(pr_path, dpi=200)
    plt.close(fig)

    return {
        'confusion_matrix': cm_path,
        'roc_curves': roc_path,
        'pr_curves': pr_path
    }


def evaluate_models(file_path: str = None) -> Dict[str, Any]:
    """
    Load saved models, evaluate both on test data, generate visual artifacts,
    and save full evaluation report.
    """
    lr_path = os.path.join(MODELS_DIR, 'logistic_regression.pkl')
    rf_path = os.path.join(MODELS_DIR, 'random_forest.pkl')
    best_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    meta_path = os.path.join(MODELS_DIR, 'model_metadata.json')

    if not (os.path.exists(lr_path) and os.path.exists(rf_path)):
        raise FileNotFoundError("Trained models not found. Please run model_training.py first.")

    lr_pipeline = joblib.load(lr_path)
    rf_pipeline = joblib.load(rf_path)
    best_pipeline = joblib.load(best_path)

    # Load test split
    prep_data = preprocess_data(file_path=file_path, include_bmi=True)
    X_test_raw = prep_data['X_test_raw']
    y_test = prep_data['y_test']
    target_classes = prep_data['target_classes']

    # Evaluate both algorithms
    lr_metrics = evaluate_single_pipeline(lr_pipeline, X_test_raw, y_test, target_classes)
    rf_metrics = evaluate_single_pipeline(rf_pipeline, X_test_raw, y_test, target_classes)

    # Read training metadata if available
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

    # Comparison summary
    best_name = metadata.get('best_model', 'Random Forest Classifier')
    best_metrics = rf_metrics if 'Forest' in best_name else lr_metrics
    best_proba = np.array(best_metrics['y_proba'])
    best_cm = np.array(best_metrics['confusion_matrix'])

    # Visualizations
    img_paths = generate_evaluation_visualizations(best_name, best_cm, y_test, best_proba, target_classes)

    comparison_table = [
        {
            'metric': 'Accuracy',
            'logistic_regression': f"{lr_metrics['accuracy'] * 100:.2f}%",
            'random_forest': f"{rf_metrics['accuracy'] * 100:.2f}%",
            'interpretation': 'Overall rate of correct classification across all 7 classes.'
        },
        {
            'metric': 'Macro F1-Score',
            'logistic_regression': f"{lr_metrics['f1_macro']:.4f}",
            'random_forest': f"{rf_metrics['f1_macro']:.4f}",
            'interpretation': 'Unweighted mean F1; penalizes models that fail on minority classes.'
        },
        {
            'metric': 'Weighted F1-Score',
            'logistic_regression': f"{lr_metrics['f1_weighted']:.4f}",
            'random_forest': f"{rf_metrics['f1_weighted']:.4f}",
            'interpretation': 'Prevalence-weighted harmonic mean of precision and recall.'
        },
        {
            'metric': 'Macro Precision',
            'logistic_regression': f"{lr_metrics['precision_macro']:.4f}",
            'random_forest': f"{rf_metrics['precision_macro']:.4f}",
            'interpretation': 'Average positive predictive value across classes.'
        },
        {
            'metric': 'Macro Recall',
            'logistic_regression': f"{lr_metrics['recall_macro']:.4f}",
            'random_forest': f"{rf_metrics['recall_macro']:.4f}",
            'interpretation': 'Average sensitivity across classes.'
        },
        {
            'metric': 'Multiclass ROC-AUC (OvR)',
            'logistic_regression': f"{lr_metrics['roc_auc_macro']:.4f}" if isinstance(lr_metrics['roc_auc_macro'], float) else lr_metrics['roc_auc_macro'],
            'random_forest': f"{rf_metrics['roc_auc_macro']:.4f}" if isinstance(rf_metrics['roc_auc_macro'], float) else rf_metrics['roc_auc_macro'],
            'interpretation': 'One-vs-Rest discrimination ability across all thresholds.'
        }
    ]

    evaluation_report = {
        'selected_model': best_name,
        'selection_reason': (
            f"{best_name} was selected as the optimal production pipeline due to its superior "
            f"Macro F1-score ({best_metrics['f1_macro']:.4f}) and high multi-class discrimination "
            f"(ROC-AUC: {best_metrics['roc_auc_macro']}) on held-out test data."
        ),
        'comparison_table': comparison_table,
        'logistic_regression': {
            'accuracy': lr_metrics['accuracy'],
            'f1_macro': lr_metrics['f1_macro'],
            'f1_weighted': lr_metrics['f1_weighted'],
            'precision_macro': lr_metrics['precision_macro'],
            'recall_macro': lr_metrics['recall_macro'],
            'roc_auc_macro': lr_metrics['roc_auc_macro'],
            'per_class': lr_metrics['per_class_report']
        },
        'random_forest': {
            'accuracy': rf_metrics['accuracy'],
            'f1_macro': rf_metrics['f1_macro'],
            'f1_weighted': rf_metrics['f1_weighted'],
            'precision_macro': rf_metrics['precision_macro'],
            'recall_macro': rf_metrics['recall_macro'],
            'roc_auc_macro': rf_metrics['roc_auc_macro'],
            'per_class': rf_metrics['per_class_report']
        },
        'experiments_comparison': metadata.get('experiments_comparison', {}),
        'metric_explanations': METRIC_EXPLANATIONS,
        'image_paths': img_paths
    }

    eval_json_path = os.path.join(EVAL_DIR, 'evaluation_metrics.json')
    with open(eval_json_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_report, f, indent=4)

    return evaluation_report


if __name__ == '__main__':
    rep = evaluate_models()
    print("Model evaluation complete.")
    print("Selected Model:", rep['selected_model'])
    for row in rep['comparison_table']:
        print(f" - {row['metric']}: LR = {row['logistic_regression']} | RF = {row['random_forest']}")

