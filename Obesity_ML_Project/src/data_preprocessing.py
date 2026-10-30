"""
Obesity Risk and Level Prediction System
Module: data_preprocessing.py
Description: Data preprocessing pipeline with duplicate handling, missing value imputation,
             OneHotEncoding, StandardScaler, ColumnTransformer, and stratified train-test splitting.
             Strictly enforces prevention of data leakage by fitting transformers on train data only.
"""

import os
import sys
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np

# Ensure local src directory is on sys.path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    from .data_loader import (
        load_dataset,
        NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES,
        TARGET_COLUMN,
        TARGET_CLASSES
    )
except (ImportError, ValueError):
    from data_loader import (
        load_dataset,
        NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES,
        TARGET_COLUMN,
        TARGET_CLASSES
    )

NUMERICAL_RANGES = {
    'Age': (10.0, 100.0),
    'Height': (1.0, 2.5),
    'Weight': (30.0, 300.0),
    'FCVC': (1.0, 3.0),
    'NCP': (1.0, 4.0),
    'CH2O': (1.0, 3.0),
    'FAF': (0.0, 3.0),
    'TUE': (0.0, 2.0)
}


def validate_numerical_ranges(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate that numerical values fall within clinically and physiologically plausible boundaries.
    """
    range_report = {}
    valid = True
    for col, (min_val, max_val) in NUMERICAL_RANGES.items():
        if col in df.columns:
            out_of_bounds = df[(df[col] < min_val) | (df[col] > max_val)]
            count_invalid = len(out_of_bounds)
            if count_invalid > 0:
                valid = False
            range_report[col] = {
                'expected_range': [min_val, max_val],
                'actual_min': float(df[col].min()),
                'actual_max': float(df[col].max()),
                'out_of_range_count': count_invalid,
                'status': 'Valid' if count_invalid == 0 else f'{count_invalid} out of range'
            }
    return {'is_valid': valid, 'details': range_report}


def build_preprocessor(
    numerical_cols: List[str] = None,
    categorical_cols: List[str] = None
) -> ColumnTransformer:
    """
    Construct a Scikit-learn ColumnTransformer for preprocessing.
    - Numerical: SimpleImputer(median) -> StandardScaler
    - Categorical: SimpleImputer(most_frequent) -> OneHotEncoder(drop='first' or ignore)
    """
    if numerical_cols is None:
        numerical_cols = NUMERICAL_FEATURES
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_FEATURES

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ],
        remainder='drop'
    )
    return preprocessor


def preprocess_data(
    df: Optional[pd.DataFrame] = None,
    file_path: Optional[str] = None,
    remove_duplicates: bool = True,
    test_size: float = 0.20,
    random_state: int = 42,
    include_bmi: bool = True
) -> Dict[str, Any]:
    """
    Execute full data preprocessing workflow with leakage-free train-test split.
    
    Returns:
        Dictionary containing:
        - X_train, X_test, y_train, y_test
        - X_train_raw, X_test_raw (untransformed DataFrames for transparency)
        - preprocessor (fitted on train set only)
        - label_encoder (fitted on target classes)
        - quality_report_before
        - quality_report_after
        - feature_names (transformed column names)
    """
    if df is None:
        df = load_dataset(file_path)
    else:
        df = df.copy()

    # --- Data Quality Before Preprocessing ---
    before_records = len(df)
    before_missing = df.isnull().sum().to_dict()
    before_duplicates = int(df.duplicated().sum())
    range_validation = validate_numerical_ranges(df)

    quality_before = {
        'total_records': before_records,
        'total_columns': len(df.columns),
        'missing_values': before_missing,
        'total_missing': int(sum(before_missing.values())),
        'duplicate_rows': before_duplicates,
        'range_validation': range_validation
    }

    # --- Step 1: Remove Duplicates ---
    if remove_duplicates and before_duplicates > 0:
        df = df.drop_duplicates().reset_index(drop=True)
    after_dedup_records = len(df)

    # --- Step 2: Feature Engineering (Calculate BMI) ---
    if include_bmi and 'Height' in df.columns and 'Weight' in df.columns:
        # BMI = Weight (kg) / Height (m)^2
        df['BMI'] = df['Weight'] / (df['Height'] ** 2)

    # Determine numerical and categorical feature lists
    current_num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    if include_bmi and 'BMI' in df.columns and 'BMI' not in current_num_cols:
        current_num_cols.append('BMI')
    current_cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    # --- Step 3: Separate X and y ---
    X = df[current_num_cols + current_cat_cols].copy()
    y_raw = df[TARGET_COLUMN].copy()

    # Encode target labels consistently
    label_encoder = LabelEncoder()
    # Fit on all canonical classes to preserve standard indexing 0..6
    label_encoder.fit(TARGET_CLASSES)
    y = label_encoder.transform(y_raw)

    # --- Step 4: Stratified Train-Test Split (80:20) ---
    # Stratification ensures each of the 7 obesity classes is proportionately represented
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # --- Step 5: Fit Preprocessor strictly on Train Data to Prevent Data Leakage ---
    preprocessor = build_preprocessor(current_num_cols, current_cat_cols)
    preprocessor.fit(X_train_raw)

    X_train_transformed = preprocessor.transform(X_train_raw)
    X_test_transformed = preprocessor.transform(X_test_raw)

    # Extract transformed feature names
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    encoded_cat_names = list(cat_encoder.get_feature_names_out(current_cat_cols))
    all_feature_names = current_num_cols + encoded_cat_names

    # --- Data Quality After Preprocessing ---
    quality_after = {
        'total_records': len(df),
        'records_retained': len(df),
        'duplicates_removed': before_duplicates if remove_duplicates else 0,
        'train_samples': len(X_train_raw),
        'test_samples': len(X_test_raw),
        'split_ratio': f"{(1 - test_size)*100:.0f}:{(test_size)*100:.0f}",
        'missing_values_after': 0,
        'input_features_count': len(all_feature_names),
        'numerical_features': current_num_cols,
        'categorical_features': current_cat_cols,
        'encoded_features_count': len(encoded_cat_names),
        'target_classes': list(label_encoder.classes_)
    }

    return {
        'X_train': X_train_transformed,
        'X_test': X_test_transformed,
        'X_train_raw': X_train_raw,
        'X_test_raw': X_test_raw,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_raw': label_encoder.inverse_transform(y_train),
        'y_test_raw': label_encoder.inverse_transform(y_test),
        'preprocessor': preprocessor,
        'label_encoder': label_encoder,
        'target_classes': list(label_encoder.classes_),
        'feature_names': all_feature_names,
        'numerical_features': current_num_cols,
        'categorical_features': current_cat_cols,
        'quality_report_before': quality_before,
        'quality_report_after': quality_after
    }


def get_preprocessing_report(file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate serializable report for Flask API and UI display.
    """
    prep_results = preprocess_data(file_path=file_path)
    return {
        'before': prep_results['quality_report_before'],
        'after': prep_results['quality_report_after'],
        'feature_names': prep_results['feature_names'],
        'target_classes': [str(c) for c in prep_results['target_classes']]
    }


if __name__ == '__main__':
    rep = get_preprocessing_report()
    print("Preprocessing completed successfully.")
    print("Before:", rep['before']['total_records'], "records, Duplicates:", rep['before']['duplicate_rows'])
    print("After:", rep['after']['total_records'], "records (Train:", rep['after']['train_samples'], ", Test:", rep['after']['test_samples'], ")")
    print(f"Total Transformed Features: {len(rep['feature_names'])}")
