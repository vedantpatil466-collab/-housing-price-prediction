# 🏠 Housing Price Prediction

> **Python · Scikit-learn · Pandas · Feature Engineering**

A machine learning regression model that predicts housing prices from property features, achieving **R² = 0.85** through systematic data cleaning, feature engineering, and hyperparameter tuning.

---

## 📌 Project Overview

Predicting house prices is a classic but powerful regression problem. This project goes beyond a basic model — it includes thorough EDA, outlier handling, feature engineering, cross-validation, and hyperparameter tuning to build a model that generalizes well to unseen data.

---

## 🎯 Key Results

| Metric | Result |
|--------|--------|
| R² Score | 0.85 |
| Dataset size | 15,000+ records |
| Model | Gradient Boosting Regressor |
| Validation | 5-Fold Cross-Validation |
| RMSE (test set) | ~18,500 |

---

## 🛠️ Tools & Technologies

- **Python** — Core ML pipeline
- **Pandas / NumPy** — Data processing
- **Matplotlib / Seaborn** — EDA and visualization
- **Scikit-learn** — Modeling, cross-validation, hyperparameter tuning
- **Jupyter Notebook** — Interactive analysis

---

## 📊 Project Pipeline

```
Raw Data → EDA → Data Cleaning → Feature Engineering
→ Model Training → Cross-Validation → Hyperparameter Tuning → Evaluation
```

### Step-by-step:
1. **EDA** — distribution plots, correlation heatmap, outlier detection
2. **Data Cleaning** — handle missing values, remove duplicates, fix skewed distributions
3. **Feature Engineering** — create price-per-sqft, age of property, location encoding
4. **Baseline Model** — Linear Regression (R² = 0.71)
5. **Improved Model** — Gradient Boosting (R² = 0.85)
6. **Tuning** — GridSearchCV for n_estimators, max_depth, learning_rate

---

## 📁 Project Structure

```
housing-price-prediction/
│
├── data/
│   └── housing_data.csv            # Dataset (source: Kaggle)
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb      # Cleaning & Feature Engineering
│   └── 03_modeling.ipynb           # Model Training & Evaluation
│
├── src/
│   ├── preprocess.py               # Reusable preprocessing functions
│   └── model.py                    # Model training and evaluation
│
├── output/
│   ├── feature_importance.png
│   └── prediction_vs_actual.png
│
└── README.md
```

---

## 🔍 Sample Code

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import r2_score
import numpy as np

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1, 0.2]
}

gbr = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(gbr, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Cross-validation
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
print(f"CV R² scores: {cv_scores.round(3)}")
print(f"Mean CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Test evaluation
y_pred = best_model.predict(X_test)
print(f"Test R²: {r2_score(y_test, y_pred):.3f}")
```

---

## 🚀 How to Use

1. Clone the repo: `git clone https://github.com/vedantpatil/housing-price-prediction`
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Open notebooks in order: `01_eda → 02_preprocessing → 03_modeling`
4. Run all cells to reproduce results

---

## 📬 Contact

**Vedant Patil** — [LinkedIn](https://linkedin.com/in/vedant-patil-1566a7386) · [Email](mailto:vedantpatil466@gmail.com)
