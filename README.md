# 🏥 ICU Mortality Prediction & Patient Clustering – MIMIC-III Final Project

This project applies machine learning techniques to ICU data from the MIMIC-III database in order to:

- 🔍 Identify clinically meaningful **patient subgroups** using unsupervised clustering  
- ⚰️ Predict **30-day mortality** using supervised learning  
- 🧠 Explain model predictions using SHAP  
- 🧪 Integrate phenotypic clusters into prediction models for better stratification

> Final project submitted by Group 8 – Industrial Engineering & Management  
> Technion, Course: Machine Learning in Medicine  
> Instructor: Dr. Orit Raphaeli

---

## 📁 Project Structure

```
.
├── EDA.ipynb                     # Exploratory data analysis (univariate, bivariate, trivariate)
├── Clustering.ipynb             # K-means & hierarchical clustering to identify patient phenotypes
├── Clustering_&_Prediction.ipynb# Integration of cluster membership into predictive models
├── Models.ipynb                 # Supervised models: Logistic Regression, Random Forest, XGBoost
├── Preprocessing.py             # Data cleaning, feature engineering, scaling pipeline
├── MIMIC_data_sample_mortality.csv       # Raw input dataset (subset of MIMIC-III)
├── ICU_data_with_kmeans_clusters.csv     # Final dataset with added cluster labels
├── logistic_regression_model.pkl         # Saved best-performing model
```

---

## 🧠 Methods Overview

### 📊 1. Data Preprocessing
- Filtered patients (<18 y/o), handled missing values via MICE
- Simplified categorical variables (e.g. ethnicity)
- Clipped outliers, log-transformed skewed features
- Standardized data using z-score
- Removed highly correlated features

### 🔎 2. Exploratory Analysis
- Identified key mortality indicators: lactate, SpO2, SOFA score, creatinine, BUN
- Detected interactions between lab values and outcomes
- Used violin plots and correlation matrices for insight

### 🤖 3. Clustering
- Applied **K-means (k=4)** and **hierarchical clustering (k=5)**
- Identified distinct ICU subgroups with different mortality rates
- Labeled clusters used as features in downstream models

### ⚰️ 4. Mortality Prediction
- Compared **Logistic Regression**, **Random Forest**, and **XGBoost**
- Evaluated using AUC, F1, recall (for mortality class), and calibration
- Visualized:
  - ROC curves
  - Calibration curves
  - SHAP summary and dependence plots

---

## ✅ Results & Recommendations

| Model               | AUC   | Recall (Deaths) | Calibration | Interpretability | Best For                    |
|---------------------|-------|------------------|--------------|------------------|-----------------------------|
| Logistic Regression | 0.837 | 0.37             | ✅ Excellent  | ✅✅ High         | Clinical use & risk scores |
| Random Forest       | 0.861 | 0.35             | ✅ Good       | ✅ Moderate       | Balanced performance        |
| XGBoost             | 0.854 | ✅ 0.41           | ❌ Poor       | ⚠️ Low           | High-risk case detection    |

> **Final Recommendation**: Use **Logistic Regression** for calibrated, explainable decision support.  
> Use **XGBoost** when recall is critical for catching at-risk patients.

---

## 🧩 Clustering Integration

- Appending cluster membership as a feature **improved XGBoost’s performance**
- Helped models recognize phenotypic patterns
- Demonstrated promise for **hybrid approaches** in clinical ML

---

## 📦 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/ML-in-medicine.git
   cd ML-in-medicine
   ```

2. Open in Google Colab or Jupyter:
   - Run `EDA.ipynb` → `Clustering.ipynb` → `Models.ipynb` in sequence
   - Or use `Clustering_&_Prediction.ipynb` for an end-to-end run

3. Requirements:
   - Python 3.8+
   - scikit-learn, pandas, numpy, matplotlib, seaborn, shap, xgboost

---

## 📜 License

This project is for educational and research use only.

---

## 👩‍⚕️ Citation

Data source: [MIMIC-III Clinical Database (v1.4)](https://physionet.org/content/mimiciii/1.4/)  
SHAP: Lundberg & Lee, NIPS 2017 – "A Unified Approach to Interpreting Model Predictions"
