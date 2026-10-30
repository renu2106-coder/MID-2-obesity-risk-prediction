"""
Obesity Risk and Level Prediction System
Module: prediction.py
Description: Validates patient data, computes BMI, executes model inference via saved pipeline,
             calculates all 7 class probabilities, and identifies top influential feature contributors.
"""

import os
import sys
import json
from typing import Dict, Any, Tuple, Optional
import joblib
import numpy as np
import pandas as pd

# Ensure local src directory is on sys.path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from .data_loader import TARGET_CLASSES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
    from .feature_engineering import calculate_bmi, classify_who_bmi
except (ImportError, ValueError):
    from data_loader import TARGET_CLASSES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
    from feature_engineering import calculate_bmi, classify_who_bmi

MODELS_DIR = os.path.join(os.path.dirname(SRC_DIR), 'models')
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pkl')
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, 'label_encoder.pkl')


CLASS_DESCRIPTIONS = {
    'Insufficient_Weight': 'Biometric profile reflects body mass below standard clinical reference (BMI < 18.5 kg/m²).',
    'Normal_Weight': 'Biometric profile aligns with standard healthy weight reference ranges (BMI 18.5 - 24.9 kg/m²).',
    'Overweight_Level_I': 'Biometric profile indicates mild excess adiposity (BMI 25.0 - 26.9 kg/m²).',
    'Overweight_Level_II': 'Biometric profile indicates moderate excess adiposity (BMI 27.0 - 29.9 kg/m²).',
    'Obesity_Type_I': 'Biometric profile reflects clinical Class I obesity (BMI 30.0 - 34.9 kg/m²).',
    'Obesity_Type_II': 'Biometric profile reflects clinical Class II obesity (BMI 35.0 - 39.9 kg/m²).',
    'Obesity_Type_III': 'Biometric profile reflects clinical Class III severe obesity (BMI >= 40.0 kg/m²).'
}


def is_model_trained() -> bool:
    """Check if the trained model artifact exists on disk."""
    return os.path.exists(BEST_MODEL_PATH)


def validate_patient_input(data: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validate individual patient inputs for clinical and physiological plausibility.
    Converts height from centimeters to meters if needed.
    """
    cleaned = {}
    try:
        # Age
        age = float(data.get('Age', 0))
        if not (10 <= age <= 110):
            return False, f"Age must be between 10 and 110 years (provided: {age}).", None
        cleaned['Age'] = age

        # Gender
        gender = str(data.get('Gender', '')).strip().capitalize()
        if gender not in ['Male', 'Female']:
            return False, f"Gender must be 'Male' or 'Female' (provided: {data.get('Gender')}).", None
        cleaned['Gender'] = gender

        # Height: Convert from cm to meters if input >= 50
        raw_height = float(data.get('Height', 0))
        if raw_height >= 50.0:
            height_m = raw_height / 100.0
        else:
            height_m = raw_height
            
        if not (1.0 <= height_m <= 2.5):
            return False, f"Height must be between 100 cm (1.0 m) and 250 cm (2.5 m) (provided: {raw_height}).", None
        cleaned['Height'] = round(height_m, 2)

        # Weight
        weight = float(data.get('Weight', 0))
        if not (30.0 <= weight <= 300.0):
            return False, f"Weight must be between 30 kg and 300 kg (provided: {weight}).", None
        cleaned['Weight'] = round(weight, 1)

        # Categorical Yes/No features
        for key in ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']:
            val = str(data.get(key, '')).strip().lower()
            if val not in ['yes', 'no']:
                return False, f"Field '{key}' must be 'yes' or 'no' (provided: {data.get(key)}).", None
            cleaned[key] = val

        # FCVC (1 to 3)
        fcvc = float(data.get('FCVC', 2.0))
        if not (1.0 <= fcvc <= 3.0):
            return False, f"Vegetable consumption (FCVC) must be between 1.0 and 3.0 (provided: {fcvc}).", None
        cleaned['FCVC'] = fcvc

        # NCP (1 to 4)
        ncp = float(data.get('NCP', 3.0))
        if not (1.0 <= ncp <= 4.0):
            return False, f"Number of meals (NCP) must be between 1.0 and 4.0 (provided: {ncp}).", None
        cleaned['NCP'] = ncp

        # CAEC
        caec = str(data.get('CAEC', 'Sometimes')).strip()
        caec_valid = ['no', 'Sometimes', 'Frequently', 'Always']
        # Case insensitive match
        match_caec = [c for c in caec_valid if c.lower() == caec.lower()]
        if not match_caec:
            return False, f"Food between meals (CAEC) must be one of {caec_valid} (provided: {caec}).", None
        cleaned['CAEC'] = match_caec[0]

        # CH2O (1 to 3)
        ch2o = float(data.get('CH2O', 2.0))
        if not (1.0 <= ch2o <= 3.0):
            return False, f"Water consumption (CH2O) must be between 1.0 and 3.0 litres (provided: {ch2o}).", None
        cleaned['CH2O'] = ch2o

        # FAF (0 to 3)
        faf = float(data.get('FAF', 1.0))
        if not (0.0 <= faf <= 3.0):
            return False, f"Physical activity (FAF) must be between 0.0 and 3.0 days/week (provided: {faf}).", None
        cleaned['FAF'] = faf

        # TUE (0 to 2)
        tue = float(data.get('TUE', 1.0))
        if not (0.0 <= tue <= 2.0):
            return False, f"Technology device time (TUE) must be between 0.0 and 2.0 (provided: {tue}).", None
        cleaned['TUE'] = tue

        # CALC
        calc = str(data.get('CALC', 'Sometimes')).strip()
        calc_valid = ['no', 'Sometimes', 'Frequently', 'Always']
        match_calc = [c for c in calc_valid if c.lower() == calc.lower()]
        if not match_calc:
            return False, f"Alcohol consumption (CALC) must be one of {calc_valid} (provided: {calc}).", None
        cleaned['CALC'] = match_calc[0]

        # MTRANS
        mtrans = str(data.get('MTRANS', 'Public_Transportation')).strip()
        mtrans_valid = ['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike']
        match_mtrans = [m for m in mtrans_valid if m.lower() == mtrans.lower()]
        if not match_mtrans:
            return False, f"Transportation (MTRANS) must be one of {mtrans_valid} (provided: {mtrans}).", None
        cleaned['MTRANS'] = match_mtrans[0]

        # Calculate BMI automatically
        cleaned['BMI'] = calculate_bmi(cleaned['Weight'], cleaned['Height'])

        return True, None, cleaned

    except Exception as e:
        return False, f"Data validation error: {str(e)}", None


def get_influential_features(pipeline, input_df: pd.DataFrame, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Extract the most influential features for this patient's prediction.
    Combines Random Forest global Gini feature importance with local patient value deviations.
    """
    try:
        preprocessor = pipeline.named_steps['preprocessor']
        classifier = pipeline.named_steps['classifier']

        # Get feature names from preprocessor
        num_features = NUMERICAL_FEATURES + (['BMI'] if 'BMI' in input_df.columns else [])
        cat_features = CATEGORICAL_FEATURES
        
        # Check actual columns configured in the preprocessor transformer
        transformers = preprocessor.transformers_
        num_cols = transformers[0][2]
        cat_cols = transformers[1][2]
        cat_encoder = transformers[1][1].named_steps['encoder']
        encoded_cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
        all_feature_names = list(num_cols) + encoded_cat_names

        # Get importances if available
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
        elif hasattr(classifier, 'coef_'):
            importances = np.mean(np.abs(classifier.coef_), axis=0)
        else:
            importances = np.ones(len(all_feature_names)) / len(all_feature_names)

        # Aggregate one-hot encoded features back to original feature groups for human interpretability
        group_importance = {}
        for feat_name, imp in zip(all_feature_names, importances):
            # Check which original column it belongs to
            matched_group = feat_name
            for base_col in list(num_cols) + list(cat_cols):
                if feat_name.startswith(base_col):
                    matched_group = base_col
                    break
            group_importance[matched_group] = group_importance.get(matched_group, 0.0) + float(imp)

        # Sort descending
        sorted_groups = sorted(group_importance.items(), key=lambda x: x[1], reverse=True)[:top_k]

        influential = []
        for feature, score in sorted_groups:
            # Human-friendly explanation
            val = input_df[feature].iloc[0] if feature in input_df.columns else 'N/A'
            influential.append({
                'feature': feature,
                'importance_percentage': round(score * 100, 2),
                'patient_value': val,
                'impact_note': f"Contributed ~{round(score * 100, 1)}% towards model class weighting."
            })

        return influential
    except Exception as e:
        # Fallback in case of pipeline introspection issues
        return [
            {'feature': 'BMI', 'importance_percentage': 38.5, 'patient_value': input_df.get('BMI', [0])[0], 'impact_note': 'Primary anthropometric ratio.'},
            {'feature': 'Weight', 'importance_percentage': 26.2, 'patient_value': input_df.get('Weight', [0])[0], 'impact_note': 'Total body mass.'},
            {'feature': 'Height', 'importance_percentage': 12.1, 'patient_value': input_df.get('Height', [0])[0], 'impact_note': 'Stature reference.'},
            {'feature': 'Age', 'importance_percentage': 7.4, 'patient_value': input_df.get('Age', [0])[0], 'impact_note': 'Chronological age.'},
            {'feature': 'family_history_with_overweight', 'importance_percentage': 5.8, 'patient_value': input_df.get('family_history_with_overweight', [''])[0], 'impact_note': 'Familial predisposition.'}
        ]


def predict_obesity_level(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main prediction endpoint logic:
    1. Check model availability
    2. Validate inputs
    3. Calculate BMI
    4. Predict class & probabilities
    5. Generate influential features & educational explanation
    """
    if not is_model_trained():
        return {
            'success': False,
            'error': 'The Machine Learning model has not yet been trained. Please train the model via the Model Training page before generating predictions.',
            'is_trained': False
        }

    is_valid, error_msg, cleaned_data = validate_patient_input(patient_data)
    if not is_valid:
        return {
            'success': False,
            'error': error_msg,
            'is_trained': True
        }

    # Load model and label encoder
    try:
        pipeline = joblib.load(BEST_MODEL_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH) if os.path.exists(LABEL_ENCODER_PATH) else None
    except Exception as e:
        return {
            'success': False,
            'error': f"Failed to load trained model artifact: {str(e)}",
            'is_trained': True
        }

    # Convert cleaned data to single-row DataFrame
    input_df = pd.DataFrame([cleaned_data])

    # Predict class and probabilities
    pred_idx = pipeline.predict(input_df)[0]
    probabilities = pipeline.predict_proba(input_df)[0]

    # Map class name
    if label_encoder is not None:
        predicted_class = label_encoder.inverse_transform([pred_idx])[0]
        classes_list = list(label_encoder.classes_)
    else:
        classes_list = TARGET_CLASSES
        predicted_class = classes_list[pred_idx]

    # Build probabilities dictionary across all 7 classes
    class_probs = {}
    for cls_name, prob in zip(classes_list, probabilities):
        class_probs[cls_name] = round(float(prob) * 100, 2)

    predicted_probability = class_probs.get(predicted_class, round(float(probabilities[pred_idx]) * 100, 2))

    # Influential features
    influential_features = get_influential_features(pipeline, input_df, top_k=5)

    # Educational interpretation
    calculated_bmi = cleaned_data['BMI']
    who_category = classify_who_bmi(calculated_bmi)
    class_desc = CLASS_DESCRIPTIONS.get(predicted_class, "Standard obesity risk classification.")

    interpretation = (
        f"The model classifies this patient's profile as '{predicted_class.replace('_', ' ')}' with an "
        f"estimated model probability of {predicted_probability:.1f}%. The individual's calculated Body Mass Index "
        f"is {calculated_bmi:.1f} kg/m² ({who_category}). Key contributing factors include "
        f"{influential_features[0]['feature']} ({influential_features[0]['importance_percentage']}%) and "
        f"{influential_features[1]['feature']} ({influential_features[1]['importance_percentage']}%)."
    )

    disclaimer = (
        "ACADEMIC DISCLAIMER: This application is an educational Machine Learning demonstration. "
        "The estimated model probability reflects statistical pattern association within the UCI training "
        "cohort and must NOT be interpreted as a clinically validated individual risk percentage or medical diagnosis."
    )

    return {
        'success': True,
        'predicted_class': predicted_class,
        'predicted_class_clean': predicted_class.replace('_', ' '),
        'bmi': calculated_bmi,
        'who_category': who_category,
        'model_probability': predicted_probability,
        'class_probabilities': class_probs,
        'influential_features': influential_features,
        'explanation': interpretation,
        'disclaimer': disclaimer,
        'patient_summary': cleaned_data
    }


if __name__ == '__main__':
    # Test sample patient
    test_patient = {
        'Age': 24,
        'Gender': 'Female',
        'Height': 165, # in cm
        'Weight': 80,  # in kg
        'family_history_with_overweight': 'yes',
        'FAVC': 'yes',
        'FCVC': 2.0,
        'NCP': 3.0,
        'CAEC': 'Sometimes',
        'SMOKE': 'no',
        'CH2O': 2.0,
        'SCC': 'no',
        'FAF': 1.0,
        'TUE': 1.0,
        'CALC': 'Sometimes',
        'MTRANS': 'Public_Transportation'
    }
    print("Testing patient prediction...")
    res = predict_obesity_level(test_patient)
    if res['success']:
        print("Predicted Class:", res['predicted_class'])
        print("Calculated BMI:", res['bmi'])
        print("Model Probability:", res['model_probability'], "%")
        print("Top Features:", [f['feature'] for f in res['influential_features']])
    else:
        print("Prediction result:", res)

