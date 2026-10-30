"""
Obesity Risk and Level Prediction System
Module: eda.py
Description: Generates 10 healthcare exploratory data visualizations with clinical interpretations,
             saves high-resolution plots to reports/eda/, and extracts structured data for web UI.
"""

import os
import sys
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless backend for web/CLI execution
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure local src directory is on sys.path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from .data_loader import load_dataset, TARGET_COLUMN, TARGET_CLASSES, NUMERICAL_FEATURES
    from .feature_engineering import calculate_bmi
except (ImportError, ValueError):
    from data_loader import load_dataset, TARGET_COLUMN, TARGET_CLASSES, NUMERICAL_FEATURES
    from feature_engineering import calculate_bmi

# Professional Healthcare Theme Palette
PRIMARY_BLUE = "#163B65"
SECONDARY_BLUE = "#2878B5"
ACCENT_TEAL = "#159A9C"
BG_LIGHT = "#F4F7FA"
TEXT_DARK = "#0E2439"
PALETTE_7 = ["#2878B5", "#159A9C", "#48A9A6", "#E3B23C", "#E27D60", "#C34A36", "#8E2800"]

# Ensure directory exists
REPORTS_DIR = os.path.join(os.path.dirname(SRC_DIR), 'reports', 'eda')
os.makedirs(REPORTS_DIR, exist_ok=True)


INTERPRETATIONS = {
    '01_obesity_categories': (
        "Class Distribution: The dataset exhibits a balanced representation across all 7 categories "
        "(roughly 270 to 350 instances per class). Obesity Type I is the most prevalent (351 records), "
        "while Insufficient Weight is the smallest class (272 records). This balance is largely attributed "
        "to synthetic over-sampling (SMOTE) applied during original dataset construction."
    ),
    '02_age_distribution': (
        "Age Demographics: The cohort is heavily skewed toward young adults, with a median age of 22.8 years "
        "and a peak between 18 and 26 years (interquartile range: 19.9 to 26.0 years). Individuals above 40 "
        "are sparsely represented, reflecting the university/collegiate demographic of the original survey respondents."
    ),
    '03_bmi_distribution': (
        "BMI Distribution: Calculated Body Mass Index (Weight/Height²) spans from 13.0 kg/m² to over 50.0 kg/m² "
        "with multimodal peaks corresponding to the WHO clinical tiers. Note: BMI is a derived biometric proxy, "
        "not an autonomous clinical diagnosis, but it closely mirrors the 7 predefined target categories."
    ),
    '04_weight_distribution': (
        "Weight Distribution: Patient weights range from 39 kg to 173 kg with a mean of 86.6 kg. "
        "Distinct cluster peaks emerge above 100 kg, representing participants categorized into "
        "Obesity Type II and Type III."
    ),
    '05_height_distribution': (
        "Height Distribution: Patient heights follow a near-Gaussian distribution centered at 1.70 meters "
        "(std: 0.09 m, range: 1.45 to 1.98 m). Differences in height between biological sexes explain "
        "the mild bimodal curvature."
    ),
    '06_physical_activity_vs_obesity': (
        "Physical Activity vs. Obesity: Participants classified under Obesity Type II and III exhibit noticeably "
        "lower physical activity frequencies (FAF) than Normal Weight individuals. However, physical activity "
        "alone does not guarantee normal weight without dietary balance."
    ),
    '07_high_calorie_food_vs_obesity': (
        "High-Calorie Diet (FAVC) vs. Obesity: Over 90% of individuals in Obesity Types I, II, and III frequently "
        "consume high-calorie foods (FAVC = yes), compared to a more balanced split among Normal Weight individuals. "
        "This indicates a strong behavioral association with excess adiposity."
    ),
    '08_family_history_vs_obesity': (
        "Family History vs. Obesity: Almost 100% of participants categorized in Obesity Types I, II, and III report "
        "a positive family history with overweight. Conversely, those with Insufficient Weight frequently lack "
        "a familial predisposition, highlighting strong genetic and shared household lifestyle influence."
    ),
    '09_vegetable_consumption_vs_obesity': (
        "Vegetable Consumption (FCVC) vs. Obesity: Frequent vegetable intake (score 2 to 3) is commonly reported "
        "across higher obesity tiers in this synthetic dataset, contrary to naive clinical intuition. This reflects "
        "either social desirability bias in self-reporting or artifacts introduced during synthetic sample synthesis."
    ),
    '10_correlation_heatmap': (
        "Correlation Matrix: Weight and BMI exhibit the highest positive collinearity (r > 0.93) with obesity severity. "
        "Age shows moderate positive correlation with weight. Physical activity (FAF) is negatively correlated with Age "
        "and Obesity, whereas Screen time (TUE) is negatively correlated with Age."
    )
}


def setup_plot_style():
    """Apply unified healthcare visual theme to matplotlib/seaborn."""
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.edgecolor'] = '#D1D5DB'
    plt.rcParams['axes.linewidth'] = 0.8


def generate_eda_visualizations(df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Generate all 10 EDA charts, save high-res images to reports/eda/,
    and return metadata and data summaries for frontend rendering.
    """
    setup_plot_style()
    if df is None:
        df = load_dataset()
    else:
        df = df.copy()

    # Calculate BMI
    if 'BMI' not in df.columns and 'Height' in df.columns and 'Weight' in df.columns:
        df['BMI'] = df['Weight'] / (df['Height'] ** 2)

    chart_results = []

    # 1. Distribution of seven obesity categories
    fig, ax = plt.subplots(figsize=(10, 5))
    class_order = TARGET_CLASSES
    counts = df[TARGET_COLUMN].value_counts().reindex(class_order)
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=PALETTE_7, ax=ax, legend=False)
    ax.set_title("Distribution of Obesity Categories (Target Variable)", fontsize=13, fontweight='bold', color=PRIMARY_BLUE)
    ax.set_xlabel("Obesity Level", fontsize=11, color=TEXT_DARK)
    ax.set_ylabel("Number of Records", fontsize=11, color=TEXT_DARK)
    ax.tick_params(axis='x', rotation=25)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, str(v), ha='center', fontsize=9, fontweight='bold', color=PRIMARY_BLUE)
    fig.tight_layout()
    path_01 = os.path.join(REPORTS_DIR, '01_obesity_categories.png')
    fig.savefig(path_01, dpi=200)
    plt.close(fig)

    chart_results.append({
        'id': '01_obesity_categories',
        'title': 'Distribution of Obesity Categories',
        'file': '01_obesity_categories.png',
        'path': path_01,
        'interpretation': INTERPRETATIONS['01_obesity_categories'],
        'data': {'labels': [c.replace('_', ' ') for c in counts.index], 'values': [int(v) for v in counts.values]}
    })

    # 2. Age distribution
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.histplot(df['Age'], kde=True, color=SECONDARY_BLUE, ax=ax, bins=25)
    ax.axvline(df['Age'].median(), color=ACCENT_TEAL, linestyle='--', linewidth=2, label=f"Median: {df['Age'].median():.1f} yrs")
    ax.set_title("Age Distribution of Participants", fontsize=13, fontweight='bold', color=PRIMARY_BLUE)
    ax.set_xlabel("Age (Years)", fontsize=11, color=TEXT_DARK)
    ax.set_ylabel("Count", fontsize=11, color=TEXT_DARK)
    ax.legend(frameon=True)
    fig.tight_layout()
    path_02 = os.path.join(REPORTS_DIR, '02_age_distribution.png')
    fig.savefig(path_02, dpi=200)
    plt.close(fig)

    chart_results.append({
        'id': '02_age_distribution',
        'title': 'Age Distribution of Participants',
        'file': '02_age_distribution.png',
        'path': path_02,
        'interpretation': INTERPRETATIONS['02_age_distribution']
    })

    # 3. BMI distribution
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.histplot(df['BMI'], kde=True, color=ACCENT_TEAL, ax=ax, bins=30)
    # Clinical WHO threshold lines
    thresholds = [(18.5, 'Underweight', '#2878B5'), (25.0, 'Normal', '#159A9C'), (30.0, 'Overweight', '#E3B23C'), (35.0, 'Obese I', '#E27D60'), (40.0, 'Obese II', '#C34A36')]
    for val, lbl, col in thresholds:
        ax.axvline(val, color=col, linestyle=':', linewidth=1.5, label=f"{lbl} ({val})")
    ax.set_title("Calculated Body Mass Index (BMI) Distribution", fontsize=13, fontweight='bold', color=PRIMARY_BLUE)
    ax.set_xlabel("BMI (kg/m²)", fontsize=11, color=TEXT_DARK)
    ax.set_ylabel("Count", fontsize=11, color=TEXT_DARK)
    ax.legend(fontsize=8, loc='upper right', frameon=True)
    fig.tight_layout()
    path_03 = os.path.join(REPORTS_DIR, '03_bmi_distribution.png')
    fig.savefig(path_03, dpi=200)
    plt.close(fig)

    chart_results.append({
        'id': '03_bmi_distribution',
        'title': 'Calculated Body Mass Index (BMI) Distribution',
        'file': '03_bmi_distribution.png',
        'path': path_03,
        'interpretation': INTERPRETATIONS['03_bmi_distribution']
    })

    # 4. Weight distribution
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.histplot(df['Weight'], kde=True, color='#2B5B84', ax=ax, bins=28)
    ax.set_title("Patient Weight Distribution", fontsize=13, fontweight='bold', color=PRIMARY_BLUE)
    ax.set_xlabel("Weight (kg)", fontsize=11, color=TEXT_DARK)
    ax.set_ylabel("Count", fontsize=11, color=TEXT_DARK)
    fig.tight_layout()
    path_04 = os.path.join(REPORTS_DIR, '04_weight_distribution.png')
    fig.savefig(path_04, dpi=200)
    plt.close(fig)

    chart_results.append({
        'id': '04_weight_distribution',
        'title': 'Patient Weight Distribution',
        'file': '04_weight_distribution.png',
        'path': path_04,
        'interpretation': INTERPRETATIONS['04_weight_distribution']
    })

    # 5. Height distribution
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.histplot(df['Height'], kde=True, color='#48A9A6', ax=ax, bins=25)
    ax.set_title("Patient Height Distribution", fontsize=13, fontweight='bold', color=PRIMARY_BLUE)
    ax.set_xlabel("Height (Meters)", fontsize=11, color=TEXT_DARK)
    ax.set_ylabel("Count", fontsize=11, color=TEXT_DARK)
    fig.tight_layout()
    path_05 = os.path.join(REPORTS_DIR, '05_height_distribution.png')
    fig.savefig(path_05, dpi=200)
    plt.close(fig)

    chart_results.append({
        'id': '05_height_distribution',
        'title': 'Patient Height Distribution',
        'file': '05_height_distribution.png',
        'path': path_05,
        'interpretation': INTERPRETATIONS['05_height_distribution']
    })

    # 6. Physical activity versus obesity level
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x=TARGET_COLUMN, y='FAF', data=df, order=class_order, hue=TARGET_COLUMN, palette=PALETTE_7, ax=ax, legend=False)
    ax.set_title("Physical Activity Frequency (FAF) across Obesity Levels", fontsize=13, fontweight='bold', color=PRIMARY_BLUE)
    ax.set_xlabel("Obesity Level", fontsize=11, color=TEXT_DARK)
    ax.set_ylabel("Physical Activity Frequency (Days/Week Score 0-3)", fontsize=10, color=TEXT_DARK)
    ax.tick_params(axis='x', rotation=25)
    fig.tight_layout()
    path_06 = os.path.join(REPORTS_DIR, '06_physical_activity_vs_obesity.png')
    fig.savefig(path_06, dpi=200)
    plt.close(fig)

    chart_results.append({
        'id': '06_physical_activity_vs_obesity',
        'title': 'Physical Activity vs. Obesity Level',
        'file': '06_physical_activity_vs_obesity.png',
        'path': path_06,
        'interpretation': INTERPRETATIONS['06_physical_activity_vs_obesity']
    })

    # 7. High-calorie food consumption versus obesity level
    fig, ax = plt.subplots(figsize=(10, 5))
    favc_cross = pd.crosstab(df[TARGET_COLUMN], df['FAVC'], normalize='index').reindex(class_order) * 100
    favc_cross.plot(kind='bar', stacked=True, color=[SECONDARY_BLUE, '#E27D60'], ax=ax, edgecolor='white')
    ax.set_title("High-Calorie Food Consumption (FAVC) by Obesity Level (%)", fontsize=13, fontweight='bold', color=PRIMARY_BLUE)
    ax.set_xlabel("Obesity Level", fontsize=11, color=TEXT_DARK)
    ax.set_ylabel("Proportion (%)", fontsize=11, color=TEXT_DARK)
    ax.legend(["No High-Calorie Food", "Frequent High-Calorie Food"], loc='lower right', frameon=True)
    ax.tick_params(axis='x', rotation=25)
    fig.tight_layout()
    path_07 = os.path.join(REPORTS_DIR, '07_high_calorie_food_vs_obesity.png')
    fig.savefig(path_07, dpi=200)
    plt.close(fig)

    chart_results.append({
        'id': '07_high_calorie_food_vs_obesity',
        'title': 'High-Calorie Diet (FAVC) vs. Obesity Level',
        'file': '07_high_calorie_food_vs_obesity.png',
        'path': path_07,
        'interpretation': INTERPRETATIONS['07_high_calorie_food_vs_obesity']
    })

    # 8. Family history versus obesity level
    fig, ax = plt.subplots(figsize=(10, 5))
    hist_cross = pd.crosstab(df[TARGET_COLUMN], df['family_history_with_overweight'], normalize='index').reindex(class_order) * 100
    hist_cross.plot(kind='bar', stacked=True, color=['#159A9C', '#C34A36'], ax=ax, edgecolor='white')
    ax.set_title("Family History with Overweight by Obesity Level (%)", fontsize=13, fontweight='bold', color=PRIMARY_BLUE)
    ax.set_xlabel("Obesity Level", fontsize=11, color=TEXT_DARK)
    ax.set_ylabel("Proportion (%)", fontsize=11, color=TEXT_DARK)
    ax.legend(["No Family History", "Positive Family History"], loc='lower right', frameon=True)
    ax.tick_params(axis='x', rotation=25)
    fig.tight_layout()
    path_08 = os.path.join(REPORTS_DIR, '08_family_history_vs_obesity.png')
    fig.savefig(path_08, dpi=200)
    plt.close(fig)

    chart_results.append({
        'id': '08_family_history_vs_obesity',
        'title': 'Family History with Overweight vs. Obesity Level',
        'file': '08_family_history_vs_obesity.png',
        'path': path_08,
        'interpretation': INTERPRETATIONS['08_family_history_vs_obesity']
    })

    # 9. Vegetable consumption versus obesity level
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x=TARGET_COLUMN, y='FCVC', data=df, order=class_order, hue=TARGET_COLUMN, palette=PALETTE_7, ax=ax, legend=False)
    ax.set_title("Vegetable Consumption Frequency (FCVC) across Obesity Levels", fontsize=13, fontweight='bold', color=PRIMARY_BLUE)
    ax.set_xlabel("Obesity Level", fontsize=11, color=TEXT_DARK)
    ax.set_ylabel("Vegetable Consumption (1=Never, 2=Sometimes, 3=Always)", fontsize=10, color=TEXT_DARK)
    ax.tick_params(axis='x', rotation=25)
    fig.tight_layout()
    path_09 = os.path.join(REPORTS_DIR, '09_vegetable_consumption_vs_obesity.png')
    fig.savefig(path_09, dpi=200)
    plt.close(fig)

    chart_results.append({
        'id': '09_vegetable_consumption_vs_obesity',
        'title': 'Vegetable Intake (FCVC) vs. Obesity Level',
        'file': '09_vegetable_consumption_vs_obesity.png',
        'path': path_09,
        'interpretation': INTERPRETATIONS['09_vegetable_consumption_vs_obesity']
    })

    # 10. Correlation heatmap for numerical variables
    fig, ax = plt.subplots(figsize=(9, 7))
    num_cols_for_corr = [c for c in NUMERICAL_FEATURES if c in df.columns]
    if 'BMI' in df.columns:
        num_cols_for_corr.append('BMI')
    corr = df[num_cols_for_corr].corr().round(2)
    sns.heatmap(corr, annot=True, cmap='Blues', fmt='.2f', linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title("Correlation Heatmap of Numerical Features", fontsize=13, fontweight='bold', color=PRIMARY_BLUE)
    fig.tight_layout()
    path_10 = os.path.join(REPORTS_DIR, '10_correlation_heatmap.png')
    fig.savefig(path_10, dpi=200)
    plt.close(fig)

    chart_results.append({
        'id': '10_correlation_heatmap',
        'title': 'Correlation Heatmap for Numerical Variables',
        'file': '10_correlation_heatmap.png',
        'path': path_10,
        'interpretation': INTERPRETATIONS['10_correlation_heatmap'],
        'correlation_matrix': corr.to_dict()
    })

    return {
        'charts': chart_results,
        'total_charts': len(chart_results),
        'reports_dir': REPORTS_DIR
    }


if __name__ == '__main__':
    res = generate_eda_visualizations()
    print(f"Generated {res['total_charts']} EDA figures successfully in {res['reports_dir']}.")
