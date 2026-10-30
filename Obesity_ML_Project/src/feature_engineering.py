"""
Obesity Risk and Level Prediction System
Module: feature_engineering.py
Description: Feature engineering, BMI calculation, WHO category mapping,
             and target leakage analysis.
"""

from typing import Dict, Any, Union
import pandas as pd
import numpy as np


def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """
    Calculate Body Mass Index (BMI).
    Formula: BMI = weight (kg) / (height (m))^2
    
    Args:
        weight_kg: Weight in kilograms (e.g. 70.0)
        height_m: Height in meters (e.g. 1.75)
        
    Returns:
        float: Calculated BMI rounded to 2 decimal places.
    """
    if height_m <= 0 or weight_kg <= 0:
        raise ValueError("Height and weight must be strictly positive values.")
    bmi = weight_kg / (height_m ** 2)
    return round(float(bmi), 2)


def classify_who_bmi(bmi: float) -> str:
    """
    Map calculated BMI to clinical World Health Organization (WHO) reference categories.
    Note: Educational clinical reference only, not an automated individual clinical diagnosis.
    """
    if bmi < 18.5:
        return "Insufficient Weight (< 18.5)"
    elif bmi < 25.0:
        return "Normal Weight (18.5 - 24.9)"
    elif bmi < 30.0:
        return "Overweight (25.0 - 29.9)"
    elif bmi < 35.0:
        return "Obesity Class I (30.0 - 34.9)"
    elif bmi < 40.0:
        return "Obesity Class II (35.0 - 39.9)"
    else:
        return "Obesity Class III (>= 40.0)"


def analyze_target_leakage() -> Dict[str, Any]:
    """
    Provide academic and methodological analysis of target leakage in obesity classification.
    
    Academic Context:
    The target variable 'NObeyesdad' consists of classes (Insufficient Weight, Normal Weight,
    Overweight Level I & II, Obesity Type I, II, III) that are clinically defined by WHO cutoffs
    directly calculated from Weight and Height (BMI = kg/m^2).
    
    When Height, Weight, and derived BMI are included as predictors, the machine learning model
    is not performing behavioral or clinical risk estimation; rather, it is approximating
    a mathematical partition function. This constitutes severe 'surrogate target leakage'
    or definition-dependent leakage.
    
    In contrast, a Lifestyle-Only model assesses how predictive dietary habits, physical activity,
    hydration, screen time, and family history are of obesity risk in the absence of direct
    anthropometric measurements.
    """
    return {
        'leakage_detected': True,
        'leakage_feature_group': ['Height', 'Weight', 'BMI'],
        'mechanism': 'The target variable NObeyesdad is partitioned directly based on WHO BMI cutoffs.',
        'clinical_implication': (
            'A model trained with Height and Weight exhibits near-perfect accuracy (>96%) because '
            'it reconstructs the mathematical formula for BMI rather than learning biological '
            'or lifestyle risk dynamics. High accuracy in this setting does NOT prove lifestyle causality.'
        ),
        'methodological_solution': (
            'We conduct two controlled experiments: '
            'Experiment A (Lifestyle and Demographic Only) excluding Height, Weight, and BMI, and '
            'Experiment B (Full-Feature Model) including anthropometric measures. '
            'This enables students and healthcare analysts to contrast mechanistic mapping with lifestyle predictive power.'
        )
    }


def add_engineered_features(df: pd.DataFrame, include_ratios: bool = False) -> pd.DataFrame:
    """
    Enrich dataset with clinical and lifestyle engineered features.
    """
    df = df.copy()
    if 'Height' in df.columns and 'Weight' in df.columns:
        df['BMI'] = df['Weight'] / (df['Height'] ** 2)

    if include_ratios:
        # Lifestyle activity-to-screen ratio (avoid division by zero with small epsilon)
        if 'FAF' in df.columns and 'TUE' in df.columns:
            df['Activity_to_Screen_Ratio'] = (df['FAF'] + 0.1) / (df['TUE'] + 0.1)
        # Daily water relative to body mass (litres per 10 kg)
        if 'CH2O' in df.columns and 'Weight' in df.columns:
            df['Hydration_Per_Weight'] = df['CH2O'] / (df['Weight'] / 10.0)

    return df


if __name__ == '__main__':
    bmi_sample = calculate_bmi(85, 1.75)
    print(f"Sample BMI for 85kg, 1.75m: {bmi_sample} -> {classify_who_bmi(bmi_sample)}")
    analysis = analyze_target_leakage()
    print("Target Leakage Analysis:\n", analysis['clinical_implication'])

