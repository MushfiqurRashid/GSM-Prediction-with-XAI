# Physics-Guided Machine Learning for Accurate GSM Prediction  
### Single Jersey Knit Fabric | Comparative Study with Explainable AI

---

## 📌 Overview

This project presents a **Physics-Guided Machine Learning (PGML)** framework for accurate prediction of **GSM (Grams per Square Meter)** in Single Jersey knit fabric.

The study integrates classical textile engineering theory (Pierce’s tightness factor) with modern ensemble learning and explainable AI techniques to develop a robust, interpretable, and industrially deployable GSM prediction pipeline.

---

## 🎯 Objectives

- Incorporate textile physics into machine learning models  
- Benchmark 22 regression algorithms  
- Perform systematic hyperparameter tuning  
- Evaluate accuracy–interpretability trade-off  
- Validate model behavior using SHAP & LIME  
- Develop a production-ready prediction pipeline  

---

## 🏭 Industrial Motivation

Accurate GSM prediction enables:

- Reduced fabric rejection rates  
- Lower material wastage  
- Improved process control  
- Data-driven quality monitoring  

Best model achieved:

- **Test RMSE:** ~4.10 g/m²  
- **Test R²:** ~0.95  
- **Test MAPE:** <1%  

---

## 🧠 Physics-Guided Feature Engineering

Instead of purely empirical modeling, we integrate textile domain knowledge using:

\[
K = \frac{\sqrt{tex}}{stitch\ length}
\]

Where:
- `tex` = Yarn linear density  
- `stitch length` = Loop geometry parameter  
- `K` = Tightness factor (physics-derived feature)

SHAP analysis confirms that:

- Yarn count  
- Tightness factor  
- Stitch length  

Together contribute **>86% of total feature importance**, validating the physics-guided approach.

---

## 📊 Models Evaluated

### Interpretable Models
- Linear Regression (OLS)
- Ridge, Lasso, ElasticNet
- Bayesian Ridge
- Huber Regressor
- Polynomial Ridge (Degree 2 & 3)
- Decision Tree
- Generalized Additive Model (GAM)

### Black-Box Models
- Random Forest
- Extra Trees (Best Model)
- XGBoost
- LightGBM
- CatBoost
- Gradient Boosting
- AdaBoost
- SVR (RBF & Polynomial)
- KNN
- MLP (Neural Networks)

---

## 🏆 Best Model: Extra Trees

| Metric | Value |
|--------|-------|
| Test RMSE | 4.10 g/m² |
| Test R² | 0.952 |
| Test MAPE | 0.47% |
| CV RMSE | 2.81 ± 0.47 |

Extra Trees provided the best balance between predictive accuracy and stability.

---

## 🔬 Explainability

### SHAP (Global & Local)
- Quantifies feature contributions
- Validates alignment with textile physics
- Highlights dominant nonlinear interactions

### LIME (Instance-Level)
- Explains individual predictions
- Improves trust in black-box models

---

## 📂 Project Structure
