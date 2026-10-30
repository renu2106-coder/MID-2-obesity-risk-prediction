"""
Obesity Risk and Level Prediction System
Module: model_training.py
Description: Trains Logistic Regression and Random Forest models with stratified 5-fold CV,
             hyperparameter grid search, and Scikit-learn Pipelines.
             Compares Experiment A (Lifestyle-only) and Experiment B (Full features).
             Saves trained models with Joblib.
"""

import os
import sys
import json
import time
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Ensure local src directory is on sys.path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from .data_loader import load_dataset, TARGET_COLUMN, TARGET_CLASSES
    from .data_preprocessing import preprocess_data, build_preprocessor
    from .feature_selection import EXPERIMENT_A_FEATURES, EXPERIMENT_B_FEATURES
except (ImportError, ValueError):
    from data_loader import load_dataset, TARGET_COLUMN, TARGET_CLASSES
    from data_preprocessing import preprocess_data, build_preprocessor
    from feature_selection import EXPERIMENT_A_FEATURES, EXPERIMENT_B_FEATURES

MODELS_DIR = os.path.join(os.path.dirname(SRC_DIR), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def train_logistic_regression(X_train_raw: pd.DataFrame, y_train: np.ndarray, preprocessor) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train Multinomial Logistic Regression within a leak-free Pipeline using GridSearchCV.
    """
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        ))
    ])

    param_grid = {
        'classifier__C': [0.1, 1.0, 10.0]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring='f1_macro',
        n_jobs=-1,
        refit=True
    )

    start_time = time.time()
    grid.fit(X_train_raw, y_train)
    training_time = round(time.time() - start_time, 2)

    best_pipe = grid.best_estimator_
    cv_results = {
        'best_params': grid.best_params_,
        'best_cv_f1_macro': round(float(grid.best_score_), 4),
        'training_time_seconds': training_time
    }
    return best_pipe, cv_results


def train_random_forest(X_train_raw: pd.DataFrame, y_train: np.ndarray, preprocessor) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train Random Forest Classifier within a leak-free Pipeline using GridSearchCV.
    """
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [15, 25, None],
        'classifier__min_samples_split': [2, 5]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring='f1_macro',
        n_jobs=-1,
        refit=True
    )

    start_time = time.time()
    grid.fit(X_train_raw, y_train)
    training_time = round(time.time() - start_time, 2)

    best_pipe = grid.best_estimator_
    cv_results = {
        'best_params': grid.best_params_,
        'best_cv_f1_macro': round(float(grid.best_score_), 4),
        'training_time_seconds': training_time
    }
    return best_pipe, cv_results


def run_experiment_comparison(df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Train models on:
    Experiment A: Lifestyle and Demographic features (Excluding Height, Weight, and BMI).
    Experiment B: Full features (Including Height, Weight, and BMI).
    """
    if df is None:
        df = load_dataset()
    else:
        df = df.copy()

    # Drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    label_enc = LabelEncoder()
    label_enc.fit(TARGET_CLASSES)
    y = label_enc.transform(df[TARGET_COLUMN])

    results = {}

    for exp_key, exp_info in [('Experiment_A', EXPERIMENT_A_FEATURES), ('Experiment_B', EXPERIMENT_B_FEATURES)]:
        cols = exp_info['numerical'] + exp_info['categorical']
        X = df[cols].copy()

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        prep = build_preprocessor(exp_info['numerical'], exp_info['categorical'])
        
        # Logistic Regression
        lr_pipe = Pipeline([
            ('preprocessor', prep),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42, C=1.0))
        ])
        lr_pipe.fit(X_tr, y_tr)
        lr_preds = lr_pipe.predict(X_te)
        lr_f1 = f1_score(y_te, lr_preds, average='macro')
        lr_acc = accuracy_score(y_te, lr_preds)

        # Random Forest
        rf_pipe = Pipeline([
            ('preprocessor', prep),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42))
        ])
        rf_pipe.fit(X_tr, y_tr)
        rf_preds = rf_pipe.predict(X_te)
        rf_f1 = f1_score(y_te, rf_preds, average='macro')
        rf_acc = accuracy_score(y_te, rf_preds)

        results[exp_key] = {
            'name': exp_info['name'],
            'description': exp_info['description'],
            'features_used': cols,
            'has_target_leakage': exp_info['has_target_leakage'],
            'logistic_regression': {
                'accuracy': round(float(lr_acc), 4),
                'macro_f1': round(float(lr_f1), 4)
            },
            'random_forest': {
                'accuracy': round(float(rf_acc), 4),
                'macro_f1': round(float(rf_f1), 4)
            }
        }

    return results


def train_and_save_all_models(file_path: str = None) -> Dict[str, Any]:
    """
    Execute full training pipeline for both algorithms, save artifacts, and generate metadata.
    """
    print("Initiating leak-free data preprocessing...")
    prep_data = preprocess_data(file_path=file_path, include_bmi=True)
    X_train_raw = prep_data['X_train_raw']
    y_train = prep_data['y_train']
    num_cols = prep_data['numerical_features']
    cat_cols = prep_data['categorical_features']
    target_classes = prep_data['target_classes']

    # Fresh preprocessor instances for each pipeline
    prep_lr = build_preprocessor(num_cols, cat_cols)
    prep_rf = build_preprocessor(num_cols, cat_cols)

    print("Training Logistic Regression with 5-fold Stratified CV...")
    lr_pipeline, lr_cv = train_logistic_regression(X_train_raw, y_train, prep_lr)

    print("Training Random Forest with 5-fold Stratified CV & Hyperparameter Grid...")
    rf_pipeline, rf_cv = train_random_forest(X_train_raw, y_train, prep_rf)

    # Determine best model based on CV Macro-F1 score
    if rf_cv['best_cv_f1_macro'] >= lr_cv['best_cv_f1_macro']:
        best_name = 'Random Forest Classifier'
        best_pipeline = rf_pipeline
        best_cv_score = rf_cv['best_cv_f1_macro']
    else:
        best_name = 'Logistic Regression'
        best_pipeline = lr_pipeline
        best_cv_score = lr_cv['best_cv_f1_macro']

    # Save models using Joblib
    lr_path = os.path.join(MODELS_DIR, 'logistic_regression.pkl')
    rf_path = os.path.join(MODELS_DIR, 'random_forest.pkl')
    best_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    label_enc_path = os.path.join(MODELS_DIR, 'label_encoder.pkl')

    joblib.dump(lr_pipeline, lr_path)
    joblib.dump(rf_pipeline, rf_path)
    joblib.dump(best_pipeline, best_path)
    joblib.dump(prep_data['label_encoder'], label_enc_path)

    # Run comparative experiments (A vs B)
    print("Running Experiment A vs B target leakage comparison...")
    experiments = run_experiment_comparison()

    metadata = {
        'training_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'best_model': best_name,
        'best_cv_f1_macro': best_cv_score,
        'target_classes': target_classes,
        'feature_names': prep_data['feature_names'],
        'logistic_regression': {
            'cv_results': lr_cv,
            'model_file': 'logistic_regression.pkl'
        },
        'random_forest': {
            'cv_results': rf_cv,
            'model_file': 'random_forest.pkl'
        },
        'experiments_comparison': experiments
    }

    meta_path = os.path.join(MODELS_DIR, 'model_metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

    print(f"All models saved successfully in {MODELS_DIR}.")
    print(f"Selected Best Model: {best_name} (CV Macro-F1: {best_cv_score:.4f})")
    return metadata


if __name__ == '__main__':
    res = train_and_save_all_models()
    print("Training finished. Summary:", res['best_model'], "CV F1:", res['best_cv_f1_macro'])
