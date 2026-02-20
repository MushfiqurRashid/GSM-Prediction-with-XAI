# 🧵 Physics-Guided Machine Learning for Accurate GSM Prediction  
### Single Jersey Knit Fabric | Comparative Study with Explainable AI

---

## 📌 Overview

This project presents a **Physics-Guided Machine Learning (PGML)** framework for accurate prediction of **GSM (Grams per Square Meter)** in Single Jersey knit fabric.

Instead of purely data-driven modeling, this work integrates textile engineering physics (Pierce’s tightness factor) with advanced ensemble machine learning models and explainable AI techniques.

The result is a high-accuracy, interpretable, and industry-ready GSM prediction system.

---

## 🎯 Key Objectives

- Integrate textile physics into ML models  
- Benchmark 22 regression algorithms  
- Perform systematic hyperparameter tuning  
- Evaluate accuracy–interpretability trade-off  
- Apply SHAP & LIME for explainability  
- Develop a reproducible ML pipeline  

---

## 🏭 Industrial Impact

Accurate GSM prediction enables:

- Reduced fabric rejection  
- Lower material wastage  
- Improved production control  
- Data-driven quality assurance  

### 🏆 Best Model Performance (Extra Trees)

| Metric | Value |
|--------|-------|
| Test RMSE | ~4.10 g/m² |
| Test R² | ~0.95 |
| Test MAPE | <1% |
| CV RMSE | 2.81 ± 0.47 |

---

## 🧠 Physics-Guided Feature Engineering

We incorporate domain knowledge using Pierce’s Tightness Factor:

K = sqrt(tex) / stitch_length

Where:
- `tex` = Yarn linear density  
- `stitch_length` = Loop geometry parameter  
- `K` = Tightness factor  

SHAP analysis confirms:

- Yarn Count  
- Tightness Factor  
- Stitch Length  

Together contribute >86% of total feature importance, validating the physics-guided approach.

---

## 📊 Models Evaluated

### 🔹 Interpretable Models
- Linear Regression
- Ridge
- Lasso
- ElasticNet
- Bayesian Ridge
- Huber Regressor
- Polynomial Ridge (Deg 2 & 3)
- Decision Tree
- Generalized Additive Model (GAM)

### 🔹 Ensemble / Black-Box Models
- Random Forest
- Extra Trees (Best)
- XGBoost
- LightGBM
- CatBoost
- Gradient Boosting
- AdaBoost
- SVR (RBF & Poly)
- KNN
- MLP Regressor

---

## 🔬 Explainable AI

### ✅ SHAP
- Global & local feature importance
- Validates textile theory alignment
- Identifies nonlinear interactions

### ✅ LIME
- Instance-level prediction explanation
- Enhances model transparency

---

## 📂 Project Structure

```
physics-guided-gsm-prediction/
│
├── README.md
├── GSM_Prediction_Report.pdf
├── dataset/
│   └── single_jersey_data.csv
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── hyperparameter_tuning.py
│   ├── evaluation.py
│   └── explainability.py
├── notebooks/
│   └── exploratory_analysis.ipynb
├── requirements.txt
└── LICENSE
```

---

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/your-username/physics-guided-gsm-prediction.git
cd physics-guided-gsm-prediction
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ Usage

Train baseline models:

```
python src/model_training.py
```

Run hyperparameter optimization:

```
python src/hyperparameter_tuning.py
```

Generate SHAP explanations:

```
python src/explainability.py
```

---

## 📦 Requirements

```
numpy
pandas
scikit-learn
xgboost
lightgbm
catboost
matplotlib
seaborn
shap
lime
scipy
```

---

## 📈 Evaluation Metrics

- RMSE (Primary Metric)
- MAE
- R² Score
- MAPE
- Cross-validation Stability

---

## 📚 Research Contributions

- Large-scale benchmarking of 22 ML models  
- Systematic hyperparameter optimization  
- Physics-guided feature validation  
- SHAP-based interpretability verification  
- Construction-specific modeling (Single Jersey Knit)  

---

## 👨‍🎓 Author

**Md. Mushfiqur Rashid Marmo**  
Email: mushfiqur.tech@gmail.com  
Contact: +880 1720-097317  

Department of Electrical and Computer Engineering  
North South University  
Fall 2025  

---

## 📄 Citation

If you use this work, please cite:

Physics-Guided Machine Learning for Accurate GSM Prediction in Single Jersey Knit Fabric:  
A Comprehensive Comparative Study with Explainable AI, 2025.

---

## 📜 License

This project is released under the MIT License.

---

## 🚀 Future Work

- Extend to multiple knit constructions  
- Real-time industrial deployment  
- Economic optimization modeling  
- Physics-Informed Neural Networks (PINNs) integration  

---

⭐ If you find this project useful, consider giving it a star!
