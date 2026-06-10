# Precision Farming for Crop Yield Optimisation
### Predicting District-Level Agricultural Output from Soil, Weather, and Irrigation Data Using K-Means Soil Profiling and Random Forest Regression

---

## 1. Project Overview

This project implements a complete machine learning system designed to predict district-level agricultural output (crop yield) across India. It integrates historical crop production statistics, soil health parameters, weather conditions, and irrigation indicators using a two-stage analytical pipeline:
1. **Unsupervised K-Means Soil Profiling**: Groups districts into homogeneous soil clusters based on macronutrients (Nitrogen, Phosphorus, Potassium), pH, average temperature, and humidity.
2. **Supervised Random Forest Regression**: Predicts crop yield (tonnes/ha) utilizing crop type, season, state, environmental factors, fertilizer usage, and the soil cluster label.

The project is structured as a modular Python package adhering to best software engineering practices.

---

## 2. Key Features

- **Data Merging and Aggregation**: Automatically aligns and merges district crop yield tables with soil recommendation profiles on standardized keys.
- **Outlier Capping**: Includes robust, IQR-based cap-and-clip outlier treatment to remove anomalies and improve regression stability.
- **Unsupervised Clustering**: Employs K-Means with Silhouette and Elbow analysis to identify optimal soil groupings (automatically selected $k=10$).
- **Supervised ML Regressor**: Features a Random Forest Regressor optimized via `GridSearchCV` hyperparameter tuning (target: R² ≥ 0.75).
- **Diagnostics and Visualisation**: Produces publication-quality plots including heatmaps, scatter plots, residual distributions, and actual vs. predicted bar charts.

---

## 3. Directory Structure

```
HCL Project Group !9/
├── README.md                        — Project documentation
├── requirements.txt                 — Python package dependencies
├── requirement_analysis.md          — Requirements analysis specification
├── main.py                          — Orchestrator script to run the full pipeline
├── data/
│   ├── raw/                         — Input raw Kaggle CSV files
│   │   ├── crop_yield.csv
│   │   └── crop_recommendation.csv
│   └── processed/                   — Cleaned and merged master datasets
│       ├── master_dataset.csv
│       └── clustered_master_dataset.csv
├── src/                             — Modular source code
│   ├── __init__.py
│   ├── data_preprocessing.py        — Loading, cleaning, and feature engineering
│   ├── eda.py                       — Exploratory data analysis graphs
│   ├── clustering.py                — K-Means soil clustering & PCA plots
│   ├── model.py                     — RandomForest model preparation & training
│   ├── evaluation.py                — Metrics calculation & model comparison
│   └── visualization.py            — Final visual summary charts
└── outputs/
    ├── figures/                     — Output visualisations (.png) and reports (.csv)
    └── models/                      — Serialized joblib model package (.pkl)
```

---

## 4. Installation & Setup

1. **Python Environment**: Ensure Python 3.10+ is installed on your system.
2. **Install Dependencies**: Install the required libraries using pip:
   ```bash
   pip install -r requirements.txt
   ```

---

## 5. Usage

### A. Run the Machine Learning Pipeline
To run the complete data cleaning, soil clustering, model training, baseline evaluation, and graph generation pipeline:
```bash
python main.py
```

*Configuration Options:* You can modify options in [main.py](file:///c:/Users/parth/Desktop/HCL%20Project%20Group%20!9/main.py):
- `RUN_EDA`: Toggle Exploratory Data Analysis (Default: `True`).
- `TUNE_HYPERPARAMS`: Toggle GridSearch hyperparameter tuning (Default: `True`).
- `CLUSTERING_K`: Force a specific number of soil clusters, or set to `None` for automatic optimal selection (Default: `None`).

### B. Run the Interactive Web Dashboard
To start the Flask-based web application:
```bash
python app.py
```
Once the server starts, open your web browser and navigate to:
**`http://127.0.0.1:5000/`**

*Dashboard Features:*
- **Harvest Predictor**: Interactive single predictions using dropdown menus (State, Season, Crop) and sliders, showing live soil profiles and estimated total yields.
- **Bulk Harvest Predictor**: Upload a spreadsheet (.csv) of multiple farm fields to calculate yield forecasts in bulk and download the results sheet.
- **Teach the AI with New Data**: Upload fresh historical yield or soil CSV files to trigger the retraining pipeline and hot-reload the newly trained AI model dynamically.

---

## 6. Output Deliverables

Upon running the pipeline, the following outputs are generated in the `outputs/` folder:
- **`outputs/models/rf_model.pkl`**: A serialized dictionary containing:
  - `model`: The trained and tuned Random Forest Regressor.
  - `scaler`: Fitted preprocessing scalers (if any).
  - `encoders`: Dictionary of fitted label encoders for categorical variables.
- **`outputs/figures/`**:
  - `descriptive_statistics.csv`: Summary statistics table.
  - `baseline_comparison.csv`: Performance metrics for RF vs. Decision Tree and Linear Regression.
  - `cluster_summary.csv`: Average soil nutrients and sizes per cluster.
  - `eda_*.png`: EDA plots (correlation heatmaps, distributions, and rainfall-yield trends).
  - `cluster_*.png`: Soil clustering diagnostic plots (Elbow/Silhouette graph, 2D PCA scatter).
  - `eval_*.png`: Model diagnostics (actual vs. predicted, residual distributions).
  - `viz_*.png`: Feature importance rank and actual vs. predicted yield grouped bar charts.

---

## 7. Model Performance

The Random Forest Regressor is validated against a 20% holdout test split of the historical datasets. 
- **Target Performance**: R² ≥ 0.75
- **Achieved Performance**: **R² = 0.9177** (using the default merged datasets)
- **Comparisons**: Outperforms Linear Regression (R² = 0.2127) and a standard Decision Tree Regressor (R² = 0.8099).

