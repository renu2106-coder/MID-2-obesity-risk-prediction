"""
Obesity Risk and Level Prediction System
Module: data_loader.py
Description: Data loading, schema validation, and dataset statistics for the UCI Obesity dataset.
"""

import os
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

# Canonical feature lists
NUMERICAL_FEATURES = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
CATEGORICAL_FEATURES = [
    'Gender', 'family_history_with_overweight', 'FAVC', 'CAEC',
    'SMOKE', 'SCC', 'CALC', 'MTRANS'
]
TARGET_COLUMN = 'NObeyesdad'

TARGET_CLASSES = [
    'Insufficient_Weight',
    'Normal_Weight',
    'Overweight_Level_I',
    'Overweight_Level_II',
    'Obesity_Type_I',
    'Obesity_Type_II',
    'Obesity_Type_III'
]

FEATURE_DESCRIPTIONS = {
    'Gender': 'Demographic biological sex (Female / Male)',
    'Age': 'Individual age in years (Range: 14 - 61)',
    'Height': 'Height in meters (e.g., 1.75 m)',
    'Weight': 'Body mass in kilograms (Range: 39 - 173 kg)',
    'family_history_with_overweight': 'Family member diagnosed with overweight or obesity (yes / no)',
    'FAVC': 'Frequent consumption of high-calorie foods (yes / no)',
    'FCVC': 'Frequency of vegetable consumption (1 = Never, 2 = Sometimes, 3 = Always)',
    'NCP': 'Number of main daily meals (1 to 4)',
    'CAEC': 'Consumption of food between meals (no, Sometimes, Frequently, Always)',
    'SMOKE': 'Current tobacco smoker status (yes / no)',
    'CH2O': 'Daily water intake in litres (1 = <1L, 2 = 1-2L, 3 = >2L)',
    'SCC': 'Calorie consumption monitoring behavior (yes / no)',
    'FAF': 'Physical activity frequency (0 = None, 1 = 1-2 days, 2 = 2-4 days, 3 = 4-5+ days)',
    'TUE': 'Time spent using technology devices daily (0 = 0-2 hours, 1 = 3-5 hours, 2 = >5 hours)',
    'CALC': 'Alcohol intake frequency (no, Sometimes, Frequently, Always)',
    'MTRANS': 'Primary mode of transportation (Automobile, Bike, Motorbike, Public_Transportation, Walking)',
    'NObeyesdad': 'Obesity risk category target variable (7 distinct WHO-aligned classes)'
}

SYNTHETIC_DATA_STATS = {
    'total_records': 2111,
    'user_collected_records': 498,
    'user_collected_percentage': 23.59,
    'synthetic_records': 1613,
    'synthetic_percentage': 76.41,
    'synthesis_method': 'Weka SMOTE filter (Synthetic Minority Over-sampling Technique)',
    'collection_countries': ['Mexico', 'Peru', 'Colombia'],
    'data_source': 'UCI Machine Learning Repository: Estimation of Obesity Levels Based on Eating Habits and Physical Condition'
}


def get_default_dataset_path() -> str:
    """Resolve default dataset path relative to project structure."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Try multiple standard candidate paths
    candidates = [
        os.path.join(project_root, 'dataset', 'ObesityDataSet.csv'),
        os.path.join(project_root, 'dataset', 'ObesityDataSet_raw_and_data_sinthetic.csv'),
        os.path.join(os.path.dirname(project_root), 'dataset', 'ObesityDataSet.csv'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


def load_dataset(file_path: str = None) -> pd.DataFrame:
    """
    Load CSV dataset and validate basic schema.
    
    Args:
        file_path: Absolute or relative path to CSV. If None, uses default path.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if file_path is None:
        file_path = get_default_dataset_path()
        
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}. Please verify file path or upload dataset.")
        
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file: {str(e)}")
        
    # Validate required columns
    expected_cols = set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN])
    actual_cols = set(df.columns)
    missing = expected_cols - actual_cols
    if missing:
        raise ValueError(f"Uploaded CSV is missing mandatory columns: {list(missing)}")
        
    return df


def get_dataset_summary(df: pd.DataFrame = None, file_path: str = None) -> Dict[str, Any]:
    """
    Generate comprehensive dataset summary statistics for UI dashboard and API.
    """
    if df is None:
        df = load_dataset(file_path)
        
    total_records, total_columns = df.shape
    missing_counts = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    
    # Target distribution
    target_counts = df[TARGET_COLUMN].value_counts().to_dict()
    target_proportions = (df[TARGET_COLUMN].value_counts(normalize=True) * 100).round(2).to_dict()
    
    # Column details
    columns_info = []
    for col in df.columns:
        col_type = str(df[col].dtype)
        is_num = col in NUMERICAL_FEATURES
        is_target = (col == TARGET_COLUMN)
        
        info = {
            'name': col,
            'type': 'Numerical' if is_num else ('Target' if is_target else 'Categorical'),
            'dtype': col_type,
            'missing': int(missing_counts.get(col, 0)),
            'description': FEATURE_DESCRIPTIONS.get(col, '')
        }
        if is_num:
            info['min'] = float(df[col].min())
            info['max'] = float(df[col].max())
            info['mean'] = round(float(df[col].mean()), 2)
            info['std'] = round(float(df[col].std()), 2)
            info['median'] = round(float(df[col].median()), 2)
        else:
            info['unique_values'] = [str(x) for x in df[col].unique().tolist()]
            info['n_unique'] = int(df[col].nunique())
            
        columns_info.append(info)
        
    # Preview top 10 rows formatted as dict
    preview = df.head(10).round(2).to_dict(orient='records')
    
    return {
        'total_records': total_records,
        'total_features': total_columns - 1, # excluding target
        'total_columns': total_columns,
        'target_classes_count': len(target_counts),
        'missing_values_count': total_missing,
        'duplicate_rows_count': duplicate_count,
        'target_distribution': target_counts,
        'target_proportions': target_proportions,
        'columns': columns_info,
        'preview': preview,
        'synthetic_data_info': SYNTHETIC_DATA_STATS
    }


if __name__ == '__main__':
    summary = get_dataset_summary()
    print(f"Loaded {summary['total_records']} rows, {summary['total_features']} features.")
    print("Target distribution:", summary['target_distribution'])

