# DevelopersHub_ChurnPipeline
**DevelopersHub Corporation — AI/ML Engineering Advanced Internship | Task 2**

## Objective
Build a reusable, production-ready ML pipeline to predict customer churn using the Telco dataset.

## Dataset
- **Name:** Telco Customer Churn Dataset (IBM)
- **Source:** Embedded directly in notebook — no download required
- **Size:** 85 customers, 20 features
- **Target:** Churn (Yes/No)

## Approach / Methodology
| Step | Description |
|------|-------------|
| Preprocessing | ColumnTransformer: StandardScaler (numeric) + OneHotEncoder (categorical) |
| Models | Logistic Regression & Random Forest Classifier |
| Tuning | GridSearchCV with StratifiedKFold (5-fold CV) |
| Export | Complete pipeline serialized with joblib |
| Evaluation | Accuracy, F1, AUC-ROC, Confusion Matrix, ROC Curve |

## Pipeline Architecture
```
Raw Input Data
      ↓
ColumnTransformer
  ├── Numeric (4 cols)  → StandardScaler
  └── Categorical (15 cols) → OneHotEncoder
      ↓
RandomForestClassifier (GridSearchCV tuned)
      ↓
Churn Prediction + Probability
```

## Key Results
| Model | Accuracy | F1 | AUC-ROC |
|-------|----------|-----|---------|
| Logistic Regression | ~85% | ~0.84 | ~0.91 |
| Random Forest (tuned) | ~90% | ~0.89 | ~0.95 |

**Top Churn Drivers:**
1. Contract type (month-to-month = highest risk)
2. Short tenure (< 6 months)
3. High monthly charges
4. Fiber optic internet service

## Tools & Libraries
Python 3.10 | scikit-learn | pandas | numpy | matplotlib | seaborn | joblib

## How to Run
```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib jupyter
jupyter notebook Task2_Churn_Pipeline.ipynb
```
Run all cells top to bottom. Pipeline is exported as `churn_pipeline.joblib`.
