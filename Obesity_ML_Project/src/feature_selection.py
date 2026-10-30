"""
Obesity Risk and Level Prediction System
Module: feature_selection.py
Description: Feature subset definitions for Experiment A (Lifestyle & Demographics Only)
             and Experiment B (Full Anthropometric & Lifestyle Model).
"""

from typing import Dict, List, Any


# Experiment A: Behavioral, lifestyle, and non-anthropometric demographic features
EXPERIMENT_A_FEATURES = {
    'name': 'Experiment A: Lifestyle & Demographic Model',
    'description': (
        'Excludes Height, Weight, and calculated BMI. Evaluates the independent predictive power '
        'of eating habits, physical activity, hydration, screen time, alcohol consumption, '
        'transportation modes, and family history without direct anthropometric cues.'
    ),
    'numerical': ['Age', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE'],
    'categorical': [
        'Gender', 'family_history_with_overweight', 'FAVC', 'CAEC',
        'SMOKE', 'SCC', 'CALC', 'MTRANS'
    ],
    'has_target_leakage': False
}

# Experiment B: Full features including anthropometrics and calculated BMI
EXPERIMENT_B_FEATURES = {
    'name': 'Experiment B: Full-Feature Model (Anthropometrics + Lifestyle)',
    'description': (
        'Includes Height, Weight, and calculated BMI alongside all behavioral attributes. '
        'Demonstrates the mathematical effect of anthropometric features on WHO obesity categories.'
    ),
    'numerical': ['Age', 'Height', 'Weight', 'BMI', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE'],
    'categorical': [
        'Gender', 'family_history_with_overweight', 'FAVC', 'CAEC',
        'SMOKE', 'SCC', 'CALC', 'MTRANS'
    ],
    'has_target_leakage': True
}


def get_feature_sets() -> Dict[str, Any]:
    """Return formal specification of both experimental feature sets."""
    return {
        'experiment_a': EXPERIMENT_A_FEATURES,
        'experiment_b': EXPERIMENT_B_FEATURES
    }


def get_feature_list_for_experiment(experiment_key: str = 'experiment_b') -> Dict[str, List[str]]:
    """Retrieve numerical and categorical columns for a given experiment."""
    if experiment_key.lower() in ['experiment_a', 'lifestyle_only', 'a']:
        return {
            'numerical': EXPERIMENT_A_FEATURES['numerical'],
            'categorical': EXPERIMENT_A_FEATURES['categorical']
        }
    else:
        return {
            'numerical': EXPERIMENT_B_FEATURES['numerical'],
            'categorical': EXPERIMENT_B_FEATURES['categorical']
        }


if __name__ == '__main__':
    feats = get_feature_sets()
    print("Experiment A features:", feats['experiment_a']['numerical'] + feats['experiment_a']['categorical'])
    print("Experiment B features:", feats['experiment_b']['numerical'] + feats['experiment_b']['categorical'])

