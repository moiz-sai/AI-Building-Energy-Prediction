<div align="center">

# 🏗️ AI Building Energy Prediction

### Predictive Modelling of Building Heating Load using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)]()

> End-to-end ML pipeline that predicts the **heating load** of residential buildings from 8 architectural features — enabling energy-efficient design decisions before construction begins.

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [ML Pipeline](#-ml-pipeline)
- [Models & Results](#-models--results)
- [Key Findings](#-key-findings)
- [Installation](#-installation)
- [Usage](#-usage)
- [GUI Application](#-gui-application)
- [Visualizations](#-visualizations)
- [Tech Stack](#-tech-stack)
- [Future Work](#-future-work)
- [Authors](#-authors)

---

## 🔍 Overview

Buildings account for approximately **40% of global energy consumption**, with heating being one of the largest contributors in colder climates. Accurately predicting a building's heating load from its physical characteristics enables architects and engineers to make data-driven design decisions *before* construction — dramatically reducing long-term energy waste and carbon emissions.

This project applies supervised machine learning regression techniques to the **UCI Energy Efficiency dataset (ENB2012)**, comparing Linear Regression, Polynomial Regression, and Decision Tree Regressor models to determine which architectural features most strongly influence heating load.

---

## 🎯 Problem Statement

> **Given 8 architectural features of a building, predict its heating load (kWh/m²) with minimum prediction error.**

The ability to forecast heating demand from building design parameters allows:
- Early-stage energy performance assessment
- Comparison of design alternatives without simulation software
- Integration into building permit and rating systems (e.g. LEED, BREEAM)

---

## 📊 Dataset

**Source:** [UCI Machine Learning Repository — Energy Efficiency Dataset (ENB2012)](https://archive.ics.uci.edu/ml/datasets/energy+efficiency)

| Property | Value |
|---|---|
| Samples | 768 |
| Features | 8 input + 2 targets |
| Task | Regression |
| Target used | Heating Load (Y1) |
| Missing values | None |

### Feature Description

| Feature | Symbol | Description | Unit |
|---|---|---|---|
| Relative Compactness | X1 | Volume-to-surface ratio of the building | — |
| Surface Area | X2 | Total exposed surface area | m² |
| Wall Area | X3 | Total wall surface area | m² |
| Roof Area | X4 | Total roof surface area | m² |
| Overall Height | X5 | Height of the building | m |
| Orientation | X6 | Cardinal orientation (2–5) | — |
| Glazing Area | X7 | Ratio of glazing to floor area | — |
| Glazing Area Distribution | X8 | Distribution pattern of glazing | — |
| **Heating Load** | **Y1** | **Target: heating energy demand** | **kWh/m²** |

---

## 📁 Project Structure

```
AI-Building-Energy-Prediction/
│
├── data/
│   ├── raw/
│   │   └── ENB2012_data.xlsx           # Original UCI dataset
│   └── processed/
│       └── energy_efficiency_clean.csv # Cleaned, feature-engineered data
│
├── src/
│   ├── data_processing.py              # Data loading, cleaning, feature engineering
│   ├── eda.py                          # Exploratory data analysis & visualizations
│   ├── models.py                       # Model definitions: LR, Poly, Decision Tree
│   ├── evaluate.py                     # Metrics: MAE, RMSE, R², learning curves
│   ├── gui.py                          # Tkinter GUI for interactive prediction
│   └── main.py                         # Pipeline entry point — runs end-to-end
│
├── notebooks/
│   ├── 01_eda.ipynb                    # Interactive EDA walkthrough
│   └── 02_modelling.ipynb              # Model training & comparison
│
├── outputs/
│   ├── plots/                          # All generated charts (PNG)
│   └── predictions/
│       ├── heating_load_prediction.xlsx
│       └── learned_coefficients.xlsx
│
├── docs/
│   ├── proposal/
│   │   ├── AI-Project-Proposal.docx
│   │   └── PIPELINE.png
│   ├── report/
│   │   └── documentation.docx
│   └── Predictive_Building_Energy_Modeling.pptx
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ ML Pipeline

```
Raw Data (ENB2012_data.xlsx)
        │
        ▼
┌─────────────────────┐
│   Data Processing   │  ← Cleaning, outlier detection, feature engineering
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│        EDA          │  ← Correlation heatmap, distributions, scatter plots
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Train/Test Split   │  ← 80/20 stratified split
└─────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│               Model Training                  │
│  ┌─────────────┐  ┌──────────┐  ┌─────────┐  │
│  │   Linear    │  │  Poly    │  │Decision │  │
│  │ Regression  │  │ Regress. │  │  Tree   │  │
│  └─────────────┘  └──────────┘  └─────────┘  │
└──────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────┐
│    Evaluation       │  ← MAE, RMSE, R², Learning Curves, Residual Analysis
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Feature Importance │  ← Before & after feature engineering comparison
└─────────────────────┘
        │
        ▼
   GUI Application / Exported Predictions
```

---

## 📈 Models & Results

### Performance Comparison

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Linear Regression | ~2.95 | ~3.84 | ~0.89 |
| Polynomial Regression (deg=2) | ~1.12 | ~1.63 | ~0.97 |
| **Decision Tree Regressor** | **~0.61** | **~1.02** | **~0.99** |

> ✅ **Decision Tree Regressor** achieved the best performance across all metrics.

### Learning Curves

Learning curves were generated for all three models to diagnose bias-variance tradeoff and confirm models generalize well to unseen data without significant overfitting.

---

## 💡 Key Findings

1. **Relative Compactness** (X1) and **Overall Height** (X5) are the strongest predictors of heating load — compact, taller buildings retain heat more efficiently.

2. **Glazing Area** (X7) has a non-linear relationship with heating load — moderate glazing increases solar gain but large glazing increases heat loss in winter.

3. **Orientation** (X6) showed the weakest correlation with heating load among all features — confirmed by feature importance analysis both before and after feature engineering.

4. **Feature engineering** (interaction terms between compactness and surface area) improved model R² from 0.89 → 0.97 for Linear Regression.

5. **Outlier removal** via IQR filtering on heating load improved Decision Tree RMSE by approximately 14%.

---

## 🛠️ Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/moiz-sai/AI-Building-Energy-Prediction.git
cd AI-Building-Energy-Prediction

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.4.0
matplotlib>=3.7.0
seaborn>=0.12.0
openpyxl>=3.1.0
```

---

## 🚀 Usage

### Run the full pipeline

```bash
python src/main.py
```

This will:
1. Load and clean the raw dataset
2. Run EDA and save all plots to `outputs/plots/`
3. Train all three models
4. Evaluate and print metrics to console
5. Save predictions to `outputs/predictions/`

### Run individual modules

```bash
# EDA only
python src/eda.py

# Train and evaluate models only
python src/models.py

# Evaluate a saved model
python src/evaluate.py
```

### Jupyter Notebooks

```bash
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_modelling.ipynb
```

---

## 🖥️ GUI Application

An interactive desktop application built with **Tkinter** allows you to input building parameters and receive an instant heating load prediction.

```bash
python src/gui.py
```

**Input fields:**
- Relative Compactness, Surface Area, Wall Area, Roof Area
- Overall Height, Orientation, Glazing Area, Glazing Area Distribution

**Output:**
- Predicted Heating Load (kWh/m²)
- Model confidence indicator

---

## 📊 Visualizations

All plots are auto-generated by the pipeline and saved to `outputs/plots/`.

### Exploratory Data Analysis

<table>
  <tr>
    <td align="center">
      <img src="outputs/plots/heatmap_correlation.png" width="400" alt="Correlation Heatmap"/>
      <br/><sub><b>Correlation Heatmap</b> — Pearson correlation of all features vs heating load</sub>
    </td>
    <td align="center">
      <img src="outputs/plots/heating_load_distribution.png" width="400" alt="Heating Load Distribution"/>
      <br/><sub><b>Heating Load Distribution</b> — Target variable skewness check</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="outputs/plots/detecting_outliers.png" width="400" alt="Outlier Detection"/>
      <br/><sub><b>Outlier Detection</b> — IQR-based boxplots across all features</sub>
    </td>
    <td align="center">
      <img src="outputs/plots/relative_compactness_vs_heating_load.png" width="400" alt="Relative Compactness vs Heating Load"/>
      <br/><sub><b>Compactness vs Heating Load</b> — Strongest single-feature relationship</sub>
    </td>
  </tr>
</table>

### Feature Importance

<table>
  <tr>
    <td align="center">
      <img src="outputs/plots/feature_importance_before.png" width="400" alt="Feature Importance Before Engineering"/>
      <br/><sub><b>Before Feature Engineering</b></sub>
    </td>
    <td align="center">
      <img src="outputs/plots/feature_importance_after.png" width="400" alt="Feature Importance After Engineering"/>
      <br/><sub><b>After Feature Engineering</b></sub>
    </td>
  </tr>
</table>

### Model Evaluation

<table>
  <tr>
    <td align="center">
      <img src="outputs/plots/linear_regression_scatter.png" width="400" alt="Linear Regression Scatter"/>
      <br/><sub><b>Linear Regression</b> — Predicted vs Actual</sub>
    </td>
    <td align="center">
      <img src="outputs/plots/decision_tree_scatter.png" width="400" alt="Decision Tree Scatter"/>
      <br/><sub><b>Decision Tree</b> — Predicted vs Actual</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="outputs/plots/residual_plot_linear_regression.png" width="400" alt="Residual Plot Linear Regression"/>
      <br/><sub><b>Residuals — Linear Regression</b></sub>
    </td>
    <td align="center">
      <img src="outputs/plots/residual_plot_decision_tree.png" width="400" alt="Residual Plot Decision Tree"/>
      <br/><sub><b>Residuals — Decision Tree</b></sub>
    </td>
  </tr>
</table>

---

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data manipulation | Pandas, NumPy |
| Machine learning | scikit-learn |
| Visualization | Matplotlib, Seaborn |
| GUI | Tkinter |
| Notebook environment | Jupyter |
| Data format | Excel (openpyxl), CSV |
| IDE | PyCharm |
| Version control | Git, GitHub |

---

## 🔮 Future Work

- [ ] Add **Random Forest** and **Gradient Boosting (XGBoost/LightGBM)** regressors for performance comparison
- [ ] Implement **cross-validation** (k-fold) instead of single train/test split
- [ ] Add **hyperparameter tuning** via GridSearchCV for Decision Tree depth, min_samples
- [ ] Extend target to also predict **Cooling Load (Y2)** as a multi-output regression task
- [ ] Deploy the GUI as a **web application** using Streamlit or Flask
- [ ] Add **SHAP values** for model explainability on individual predictions
- [ ] Integrate **CI/CD** with GitHub Actions for automated testing of the pipeline

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Authors

**Moiz Sai**
- GitHub: [@moiz-sai](https://github.com/moiz-sai)

---

## 🙏 Acknowledgements

- Dataset: [Angeliki Xifara and Athanasios Tsanas, Oxford Centre for Industrial and Applied Mathematics, University of Oxford, UK](https://archive.ics.uci.edu/ml/datasets/energy+efficiency)
- Reference paper: *Tsanas, A., Xifara, A. (2012). Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools. Energy and Buildings.*

---

<div align="center">

⭐ If you found this project useful, please consider giving it a star!

</div>