# Obesity Risk and Level Prediction Using Machine Learning
### An End-to-End Healthcare Analytics Machine Learning System

> **ACADEMIC & CLINICAL DISCLAIMER:**
> *This application is an educational Machine Learning demonstration developed for healthcare analytics. Predictions are statistical estimates derived from patterns within the UCI training dataset and must **NOT** be used for clinical diagnosis, treatment planning, or individual patient care decisions.*

---

## 1. Project Overview & Introduction

Obesity has emerged as a major global public health epidemic, linked as a primary risk factor to cardiometabolic disorders, type-2 diabetes mellitus, cardiovascular conditions, and musculoskeletal complications. While traditional clinical evaluation relies primarily on anthropometric indices such as Body Mass Index (BMI), comprehensive epidemiological and behavioral informatics recognize that obesity is fundamentally influenced by complex interactions between dietary habits, physical activity levels, hydration, screen time, familial predispositions, and socioeconomic transportation factors.

This project delivers an end-to-end Machine Learning healthcare analytics web system capable of evaluating an individual's demographic and lifestyle profile to predict their obesity level across seven World Health Organization (WHO) aligned categories.

---

## 2. Problem Statement

Healthcare systems often face challenges in early risk stratification and preventative public health screening due to reliance on sporadic in-clinic visits. Clinical data science can provide supportive informatics tools to explore behavioral risk profiles. However, a major methodological risk in applying machine learning to obesity datasets is **target leakage**—where derived anthropometric variables (such as Height and Weight) deterministically reconstruct the target variable rather than allowing algorithms to evaluate true lifestyle predictability. This project directly addresses and investigates this phenomenon.

---

## 3. Aim and Objectives

The overarching aim is to design, validate, and deploy an educational, leakage-free multiclass prediction system. Specific objectives include:

1. **Data Auditing & Preprocessing:** Implement automated quality checks, duplicate removal (24 instances), range validation, median and mode imputation, OneHotEncoding, and StandardScaler transformations without test or target leakage.
2. **Exploratory Data Analysis (EDA):** Engineer 10 publication-quality healthcare visualizations dissecting demographic age clusters, BMI multimodal peaks, physical activity levels, dietary habits, and collinearity.
3. **Controlled Target Leakage Investigation:** Build and evaluate two rigorous experiments:
   - **Experiment A (Lifestyle & Demographics Only):** Excludes Height, Weight, and derived BMI to measure the true predictive capacity of behavior.
   - **Experiment B (Full Anthropometrics + Lifestyle):** Measures the deterministic mathematical partitioning of WHO BMI thresholds.
4. **Model Optimization & Benchmarking:** Train Multinomial Logistic Regression and Random Forest ensembles with 5-fold Stratified Cross-Validation and hyperparameter grid searches.
5. **Rigorous Multiclass Evaluation:** Validate models on an isolated 20% held-out test cohort across Accuracy, Macro/Weighted F1, Macro/Weighted Precision and Recall, Multiclass ROC-AUC (One-vs-Rest), Confusion Matrices, and PR curves.
6. **Deployment & Explainability:** Deliver a modern Flask backend and responsive healthcare analytics dashboard featuring live BMI calculation, multiclass probability distributions, and top feature contribution insights.

---

## 4. Dataset Source & Provenance

The system utilizes the authoritative dataset from the **UCI Machine Learning Repository**:
- **Dataset Title:** *Estimation of Obesity Levels Based on Eating Habits and Physical Condition*
- **UCI Repository ID:** 544
- **Original Authors:** Fabio Mendoza Palechor and Alexis de la Hoz Manotas (University of the North, Colombia)
- **Reference Publication:** *Data in Brief*, 26, 104344 (2019)
- **Total Records:** 2,111
- **Input Features:** 16 attributes (demographic, dietary, and lifestyle)
- **Target Variable:** `NObeyesdad` (7 categorical classes)

### Synthetic Sampling Breakdown
It is vital to document dataset provenance transparently:
- **Direct Survey Records:** **498 instances (23.59%)** were directly collected from respondents across Mexico, Peru, and Colombia via an online survey instrument.
- **Synthetically Generated Records:** **1,613 instances (76.41%)** were synthetically generated using the Weka SMOTE (Synthetic Minority Over-sampling Technique) filter to balance category prevalence.

### Seven Target Classes (Multiclass Classification)
The target variable is strictly evaluated as a 7-class multiclass problem:
1. `Insufficient_Weight` (WHO BMI < 18.5)
2. `Normal_Weight` (WHO BMI 18.5 – 24.9)
3. `Overweight_Level_I` (WHO BMI 25.0 – 26.9)
4. `Overweight_Level_II` (WHO BMI 27.0 – 29.9)
5. `Obesity_Type_I` (WHO BMI 30.0 – 34.9)
6. `Obesity_Type_II` (WHO BMI 35.0 – 39.9)
7. `Obesity_Type_III` (WHO BMI &ge; 40.0)

---

## 5. Dataset Limitations

1. **Synthetic Data Artifacts:** Approximately 76.4% of rows were synthesized via interpolation algorithms. This generates continuous floating-point values for originally discrete ordinal responses (e.g. `FCVC = 2.45`, `NCP = 3.21`) and can introduce subtle correlation artifacts.
2. **Age Truncation:** The demographic cohort is concentrated in young collegiate adults (median age: 22.8 years), limiting broad generalizability to pediatric or geriatric populations.
3. **Self-Report Bias:** Dietary intakes (such as vegetable consumption and high-calorie snacking) were self-reported, susceptible to recall and social desirability biases.
4. **Non-Causal Association:** Machine learning classification identifies correlative statistical associations; it does not establish medical etiology or biological causality.

---

## 6. Machine Learning Workflow & Architecture

The system enforces modular separation between ML processing, backend APIs, and the presentation layer:

```
+-------------------------------------------------------------+
|                      UCI CSV Dataset                        |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| 1. Data Loader & Quality Audit (Duplicates, Ranges, Schema) |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| 2. Stratified 80:20 Train-Test Split (1669 Train / 418 Test)|
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| 3. Leak-Free Preprocessing Pipeline (Fitted on Train Only)  |
|    - ColumnTransformer                                      |
|    - Numerical: SimpleImputer(median) -> StandardScaler     |
|    - Categorical: SimpleImputer(mode) -> OneHotEncoder      |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| 4. Controlled Target Leakage Study (Experiment A vs. B)     |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| 5. Stratified 5-Fold Cross-Validation & Grid Hyperparam Tuning
|    - Multinomial Logistic Regression (L-BFGS)               |
|    - Random Forest Ensemble (100-200 trees, depth tuning)   |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| 6. Final Evaluation on Held-Out Test Cohort (418 Instances) |
|    - Confusion Matrix, OvR ROC Curves, Precision-Recall     |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| 7. Flask REST API & Modern Analytics Frontend Dashboard    |
+-------------------------------------------------------------+
```

---

## 7. Data Preprocessing & Leakage Prevention

- **Duplicate Handling:** Exactly 24 duplicate records were detected and purged, leaving 2,087 unique patient cases.
- **Missing Value Handling:** Although the raw dataset was populated, real-world deployment requires defensive imputation. `SimpleImputer` is integrated with median strategy for numerical columns and most-frequent strategy for categorical columns.
- **Range Validation:** Physiological checks ensure variables satisfy acceptable ranges (e.g. Age: 10–100, Height: 1.0–2.5 m, Weight: 30–300 kg).
- **Encoding & Scaling:** `OneHotEncoder(handle_unknown='ignore')` transforms 8 categorical variables into 23 binary indicators. `StandardScaler` standardizes continuous variables.
- **Prevention of Data Leakage:** All transformer fits are strictly localized to the training partition (`X_train`), guaranteeing test data isolation.

---

## 8. Exploratory Data Analysis (EDA)

Ten high-resolution healthcare visual charts were produced and saved to `reports/eda/`:

1. `01_obesity_categories.png`: Target class distribution verifying balanced representation.
2. `02_age_distribution.png`: Histogram and KDE highlighting the young-adult demographic peak.
3. `03_bmi_distribution.png`: Multimodal curve annotated with clinical WHO BMI thresholds.
4. `04_weight_distribution.png`: Patient body mass distribution with severe obesity clusters.
5. `05_height_distribution.png`: Normal distribution centered at 1.70 meters.
6. `06_physical_activity_vs_obesity.png`: Boxplots illustrating declining activity frequency in high obesity tiers.
7. `07_high_calorie_food_vs_obesity.png`: Stacked bar chart showing over 90% high-calorie consumption in obese cohorts.
8. `08_family_history_vs_obesity.png`: Demonstration of strong familial predisposition in Class II and III obesity.
9. `09_vegetable_consumption_vs_obesity.png`: Self-reported vegetable frequency analysis.
10. `10_correlation_heatmap.png`: Collinearity matrix highlighting Weight and BMI correlation ($r > 0.93$).

---

## 9. Feature Engineering & Target Leakage Analysis

Body Mass Index (BMI) is calculated as:
$$\text{BMI} = \frac{\text{Weight (kg)}}{(\text{Height (m)})^2}$$

### Controlled Target Leakage Experiment

| Experiment | Features Included | Target Leakage Status | Logistic Regression (Macro F1) | Random Forest (Macro F1) |
| :--- | :--- | :--- | :---: | :---: |
| **Experiment A (Lifestyle Only)** | Age, Gender, Diet, Physical Activity, Water, Screen Time, Smoking, Alcohol, Transport, Family History | **No Leakage** (Pure Behavioral Signals) | `0.5866` (Acc: 62.7%) | **`0.8366` (Acc: 84.0%)** |
| **Experiment B (Full Features)** | All Experiment A features + **Height, Weight, and BMI** | **Deterministic Leakage** (Surrogate Target) | `0.9230` (Acc: 92.6%) | **`0.9801` (Acc: 98.1%)** |

**Clinical & Academic Conclusion:**
In Experiment A, lifestyle and demographic predictors alone achieve an impressive **84.0% accuracy** with Random Forest, demonstrating substantial predictive signal in behavioral patterns. The leap to **98.1%** in Experiment B occurs because the target classes are defined mathematically by WHO BMI thresholds, meaning height and weight provide near-deterministic classification. High accuracy in Experiment B is therefore mechanistic rather than proof that lifestyle causally drives the diagnosis.

---

## 10. Model Training & Hyperparameter Tuning

Training was performed using 5-fold Stratified Cross-Validation via `GridSearchCV`:

1. **Multinomial Logistic Regression:**
   - Solver: `lbfgs`, `max_iter=1000`
   - Parameter Grid: `C: [0.1, 1.0, 10.0]`
   - Best Regularization: `C = 10.0`
   - 5-Fold CV Macro F1: `0.9512`
2. **Random Forest Classifier:**
   - Parameter Grid: `n_estimators: [100, 200]`, `max_depth: [15, 25, None]`, `min_samples_split: [2, 5]`
   - Best Configuration: `n_estimators: 100`, `max_depth: 25`, `min_samples_split: 2`
   - 5-Fold CV Macro F1: **`0.9804` (98.04%)**

Selected optimal pipeline: **Random Forest Classifier**.

---

## 11. Model Evaluation Results (Held-Out Test Set)

Both trained models were evaluated on the held-out test cohort of **418 records** (20% partition):

| Evaluation Metric | Logistic Regression (Multinomial) | Random Forest Classifier | Metric Meaning in Healthcare |
| :--- | :---: | :---: | :--- |
| **Accuracy** | 94.98% | **97.85%** | Overall proportion of correct classifications |
| **Macro F1-Score** | 0.9487 | **0.9779** | Unweighted mean F1; penalizes poor minority performance |
| **Weighted F1-Score** | 0.9495 | **0.9786** | Prevalence-weighted harmonic mean of precision & recall |
| **Macro Precision** | 0.9488 | **0.9788** | Positive predictive value averaged across all 7 tiers |
| **Macro Recall** | 0.9494 | **0.9780** | Diagnostic sensitivity averaged across all 7 tiers |
| **Multiclass ROC-AUC (OvR)** | 0.9974 | **0.9998** | Multi-threshold discrimination across One-vs-Rest curves |

Visual evaluation artifacts generated:
- `reports/evaluation/confusion_matrix.png`
- `reports/evaluation/roc_curves.png`
- `reports/evaluation/pr_curves.png`
- `reports/evaluation/evaluation_metrics.json`

---

## 12. Obesity Level Prediction & Explainability

The prediction module (`src/prediction.py`) processes patient clinical data through:
1. Input bounds checking (Age, Stature, Weight, Ordinal habits).
2. Automated BMI calculation and WHO reference classification.
3. Inference execution through the serialized Scikit-learn Pipeline.
4. Computation of model probabilities across all 7 classes.
5. Local feature influence estimation combining ensemble importance with individual patient deviations.
6. Transparent reporting with clear academic disclaimers.

---

## 13. Technology Stack

- **Core Language:** Python 3.10+ (Tested on Python 3.14)
- **Data Engineering:** Pandas, NumPy
- **Visual Analytics:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, Joblib
- **Web Application Backend:** Flask 3.x, Werkzeug
- **Frontend Presentation:** HTML5, CSS3, Vanilla JavaScript (Fetch API)

---

## 14. Project Directory Structure

```
Obesity_ML_Project/
│
├── dataset/
│   ├── ObesityDataSet.csv
│   └── ObesityDataSet_raw_and_data_sinthetic.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── best_model.pkl
│   ├── label_encoder.pkl
│   └── model_metadata.json
│
├── reports/
│   ├── eda/
│   │   ├── 01_obesity_categories.png
│   │   ├── 02_age_distribution.png
│   │   ├── 03_bmi_distribution.png
│   │   ├── ... (all 10 charts)
│   │   └── 10_correlation_heatmap.png
│   └── evaluation/
│       ├── confusion_matrix.png
│       ├── roc_curves.png
│       ├── pr_curves.png
│       └── evaluation_metrics.json
│
├── src/
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── feature_selection.py
│   ├── eda.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── prediction.py
│
├── backend/
│   ├── app.py
│   └── routes.py
│
├── frontend/
│   ├── templates/
│   │   ├── index.html
│   │   ├── dashboard.html
│   │   ├── dataset.html
│   │   ├── preprocessing.html
│   │   ├── eda.html
│   │   ├── training.html
│   │   ├── evaluation.html
│   │   ├── prediction.html
│   │   └── reports.html
│   └── static/
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── app.js
│
├── run.py
├── requirements.txt
├── README.md
└── .env.example
```

---

## 15. Execution Instructions (Windows / VS Code Terminal)

### Step 1: Open Terminal in Workspace Directory
In Visual Studio Code or PowerShell, navigate to the project directory:
```powershell
cd "c:\Users\HI\OneDrive\Desktop\Obesity risk"
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: (Optional) Re-execute ML Pipeline & Evaluation
To regenerate all models, metrics, and EDA plots from scratch:
```powershell
python Obesity_ML_Project/src/eda.py
python Obesity_ML_Project/src/model_training.py
python Obesity_ML_Project/src/model_evaluation.py
```

### Step 4: Launch the Web Dashboard
```powershell
python run.py
```

### Step 5: Access the Web Dashboard
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 16. Future Improvements

1. **Longitudinal Clinical Integration:** Expand models to process time-series health monitoring data (e.g. continuous wearable accelerometry, continuous glucose monitoring).
2. **SHAP & LIME Deep Explainability:** Integrate TreeSHAP to compute exact additive feature attribution values for each individual patient prediction.
3. **Clinical EHR Interoperability:** Build HL7 / FHIR (Fast Healthcare Interoperability Resources) JSON adapters to enable standards-based electronic health record integration.
4. **Subgroup Fairness & Demographic Parity Auditing:** Perform disparate impact audits across sex, age brackets, and transport modalities to avoid demographic prediction bias.

