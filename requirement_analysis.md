# Requirement Analysis

## Precision Farming for Crop Yield Optimisation
**Predicting District-Level Agricultural Output from Soil, Weather, and Irrigation Data Using K-Means Soil Profiling and Random Forest Regression**

---

## 1. Project Overview

This project aims to develop a machine learning–based decision-support system that predicts **district-level crop yield** by integrating heterogeneous agricultural data sources — soil characteristics, weather patterns, and irrigation parameters. The system employs a two-stage analytical pipeline:

1. **K-Means Clustering** for unsupervised soil profiling (grouping districts by soil similarity)
2. **Random Forest Regression** for supervised crop yield prediction

The solution targets agricultural planners, policy makers, and farming communities to enable data-driven decisions for crop selection, resource allocation, and yield optimisation.

---

## 2. Problem Statement

Indian agriculture faces significant yield variability across districts due to diverse soil types, erratic weather patterns, and uneven irrigation infrastructure. Traditional yield estimation methods rely on manual surveys and historical averages, which are:

- **Inaccurate** — unable to capture micro-level variations
- **Delayed** — available only post-harvest
- **Non-actionable** — lacking prescriptive insights

There is a need for a **predictive analytics system** that can forecast district-level crop output ahead of harvest by leveraging soil, weather, and irrigation data with modern machine learning techniques.

---

## 3. Objectives

| # | Objective | Type |
|---|-----------|------|
| O1 | Collect and integrate multi-source agricultural datasets (soil, weather, irrigation, historical yield) | Data Engineering |
| O2 | Perform exploratory data analysis (EDA) to understand feature distributions, correlations, and anomalies | Analysis |
| O3 | Apply K-Means clustering to create distinct soil profile groups across districts | Unsupervised ML |
| O4 | Build a Random Forest Regression model to predict crop yield per district | Supervised ML |
| O5 | Evaluate model performance using standard regression metrics (R², RMSE, MAE) | Validation |
| O6 | Visualise results through interactive charts, maps, and dashboards | Visualisation |
| O7 | Provide actionable insights for crop planning and resource optimisation | Decision Support |

---

## 4. Scope

### 4.1 In-Scope
- District-level yield prediction for major crops (rice, wheat, maize, pulses, oilseeds)
- Historical data analysis (minimum 5–10 years)
- Soil profiling using clustering algorithms
- Weather feature engineering (rainfall, temperature, humidity)
- Irrigation coverage as a predictive feature
- Model training, validation, and evaluation
- Static and interactive visualisations
- Final report and presentation

### 4.2 Out-of-Scope
- Real-time IoT sensor integration
- Farm-level (individual farmer) predictions
- Mobile or web application deployment
- Satellite/remote sensing imagery processing
- Economic or market price prediction
- Pest and disease modelling

---

## 5. Functional Requirements

### FR-01: Data Collection & Integration
| ID | Requirement | Priority |
|----|------------|----------|
| FR-01.1 | System shall ingest crop production data (area, production, yield) from government datasets (e.g., data.gov.in, ICRISAT) | High |
| FR-01.2 | System shall ingest soil data (pH, nitrogen, phosphorus, potassium, organic carbon, soil type) per district | High |
| FR-01.3 | System shall ingest weather data (annual/seasonal rainfall, temperature min/max/avg, humidity) per district | High |
| FR-01.4 | System shall ingest irrigation data (% area irrigated, irrigation source type) per district | High |
| FR-01.5 | System shall merge all datasets on common keys (district, state, year, crop) | High |

### FR-02: Data Preprocessing
| ID | Requirement | Priority |
|----|------------|----------|
| FR-02.1 | System shall handle missing values using appropriate imputation strategies (mean, median, KNN, or domain-based) | High |
| FR-02.2 | System shall detect and treat outliers using IQR or Z-score methods | High |
| FR-02.3 | System shall encode categorical variables (crop type, soil type, season) using label or one-hot encoding | High |
| FR-02.4 | System shall normalise/standardise numerical features where required | Medium |
| FR-02.5 | System shall perform feature engineering to derive new features (e.g., rainfall-to-irrigation ratio, NPK index) | Medium |

### FR-03: Exploratory Data Analysis (EDA)
| ID | Requirement | Priority |
|----|------------|----------|
| FR-03.1 | System shall generate descriptive statistics for all numerical features | High |
| FR-03.2 | System shall produce correlation heatmaps to identify feature relationships | High |
| FR-03.3 | System shall visualise yield distributions across states, districts, and crops | High |
| FR-03.4 | System shall plot time-series trends for yield, rainfall, and production | Medium |
| FR-03.5 | System shall create box plots and violin plots for soil parameter distributions | Medium |

### FR-04: K-Means Soil Profiling (Clustering)
| ID | Requirement | Priority |
|----|------------|----------|
| FR-04.1 | System shall select relevant soil features for clustering (pH, N, P, K, organic carbon, moisture) | High |
| FR-04.2 | System shall determine optimal number of clusters using the Elbow Method and Silhouette Score | High |
| FR-04.3 | System shall apply K-Means clustering to group districts into soil profile categories | High |
| FR-04.4 | System shall visualise clusters using 2D/3D PCA or t-SNE scatter plots | High |
| FR-04.5 | System shall assign cluster labels as a new feature for downstream modelling | High |
| FR-04.6 | System shall generate cluster-wise summary statistics (centroid values, size, dominant soil type) | Medium |

### FR-05: Random Forest Regression (Yield Prediction)
| ID | Requirement | Priority |
|----|------------|----------|
| FR-05.1 | System shall define the target variable as crop yield (production/area in tonnes/hectare) | High |
| FR-05.2 | System shall use soil cluster label, weather features, irrigation data, and crop type as input features | High |
| FR-05.3 | System shall split data into training (80%) and testing (20%) sets with stratification | High |
| FR-05.4 | System shall train a Random Forest Regressor with configurable hyperparameters (n_estimators, max_depth, min_samples_split) | High |
| FR-05.5 | System shall perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV | Medium |
| FR-05.6 | System shall extract and rank feature importances from the trained model | High |
| FR-05.7 | System shall support cross-validation (k-fold) for robust performance estimation | Medium |

### FR-06: Model Evaluation
| ID | Requirement | Priority |
|----|------------|----------|
| FR-06.1 | System shall compute R² (coefficient of determination) on test data | High |
| FR-06.2 | System shall compute RMSE (Root Mean Squared Error) on test data | High |
| FR-06.3 | System shall compute MAE (Mean Absolute Error) on test data | High |
| FR-06.4 | System shall generate Actual vs. Predicted scatter plots | High |
| FR-06.5 | System shall generate residual plots to assess prediction error distribution | Medium |
| FR-06.6 | System shall compare model performance with a baseline (e.g., Linear Regression, Decision Tree) | Medium |

### FR-07: Visualisation & Reporting
| ID | Requirement | Priority |
|----|------------|----------|
| FR-07.1 | System shall generate feature importance bar charts | High |
| FR-07.2 | System shall produce district-wise predicted yield heatmaps/choropleth maps | Medium |
| FR-07.3 | System shall create cluster distribution pie/bar charts | Medium |
| FR-07.4 | System shall generate a comprehensive PDF/HTML report summarising all findings | Medium |
| FR-07.5 | System shall support interactive visualisations (Plotly/Streamlit dashboard) | Low |

---

## 6. Non-Functional Requirements

| ID | Requirement | Category | Priority |
|----|------------|----------|----------|
| NFR-01 | Model training shall complete within 5 minutes on standard hardware (8 GB RAM, 4-core CPU) | Performance | High |
| NFR-02 | Prediction for a single district shall return within 2 seconds | Performance | Medium |
| NFR-03 | Code shall be modular, well-documented, and follow PEP 8 standards | Maintainability | High |
| NFR-04 | System shall run on Python 3.8+ with standard data science libraries | Portability | High |
| NFR-05 | All data processing steps shall be reproducible with fixed random seeds | Reproducibility | High |
| NFR-06 | System shall handle datasets up to 100,000 records without memory errors | Scalability | Medium |
| NFR-07 | Source code shall be version-controlled using Git | Maintainability | High |
| NFR-08 | Model shall achieve minimum R² ≥ 0.75 on test data | Accuracy | High |
| NFR-09 | System shall provide clear error messages for invalid inputs or missing data | Usability | Medium |

---

## 7. Data Requirements

### 7.1 Primary Datasets

| Dataset | Source | Key Fields | Format |
|---------|--------|------------|--------|
| Crop Production Data | data.gov.in / ICRISAT / Kaggle | State, District, Crop, Season, Area (ha), Production (tonnes), Yield | CSV |
| Soil Health Data | Soil Health Card Portal / NBSS&LUP | District, pH, N, P, K, OC, EC, Soil Type, Texture | CSV |
| Weather/Rainfall Data | IMD / Open-Meteo / NASA POWER | District, Year, Annual Rainfall (mm), Avg Temp (°C), Humidity (%) | CSV |
| Irrigation Data | Minor Irrigation Census / data.gov.in | District, Net Irrigated Area, Gross Irrigated Area, Source (canal, well, tube well) | CSV |

### 7.2 Data Volume Estimates

| Aspect | Estimate |
|--------|----------|
| Districts covered | 500–700 districts (India) |
| Time span | 10–15 years (2005–2020) |
| Crops covered | 10–15 major crops |
| Total records (after merging) | 50,000–100,000 rows |
| Feature count (after engineering) | 20–30 features |

### 7.3 Data Quality Requirements
- Missing values: ≤ 15% per feature (otherwise feature is dropped or heavily imputed)
- Temporal consistency: Continuous year-wise data with no gaps > 2 years
- Spatial consistency: District names normalised across datasets
- No duplicate records after merging

---

## 8. Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Programming Language | Python 3.10+ | Core development |
| Data Manipulation | Pandas, NumPy | Data loading, cleaning, transformation |
| Visualisation | Matplotlib, Seaborn, Plotly | Charts, heatmaps, interactive plots |
| Machine Learning | Scikit-learn | K-Means, Random Forest, evaluation metrics |
| Geospatial (Optional) | Geopandas, Folium | Choropleth maps |
| Notebook Environment | Jupyter Notebook / Google Colab | Development and presentation |
| Dashboard (Optional) | Streamlit | Interactive web dashboard |
| Version Control | Git + GitHub | Code versioning |
| Reporting | Jupyter nbconvert / LaTeX | Final report generation |

---

## 9. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA SOURCES                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐  │
│  │   Crop   │ │   Soil   │ │ Weather  │ │ Irrigation│  │
│  │Production│ │  Health  │ │  / IMD   │ │   Census  │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └─────┬─────┘  │
│       │             │            │              │        │
└───────┼─────────────┼────────────┼──────────────┼────────┘
        │             │            │              │
        ▼             ▼            ▼              ▼
┌─────────────────────────────────────────────────────────┐
│              DATA PREPROCESSING MODULE                  │
│  • Missing value imputation                             │
│  • Outlier detection & treatment                        │
│  • Feature encoding & scaling                           │
│  • Feature engineering                                  │
│  • Dataset merging on (District, Year, Crop)            │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
┌──────────────────┐      ┌───────────────────────┐
│   EDA MODULE     │      │  K-MEANS CLUSTERING   │
│  • Distributions │      │  MODULE               │
│  • Correlations  │      │  • Elbow method       │
│  • Trend plots   │      │  • Silhouette score   │
│  • Box plots     │      │  • Soil profiling     │
└──────────────────┘      │  • PCA visualisation  │
                          └───────────┬───────────┘
                                      │
                                      │ Cluster labels
                                      ▼
                          ┌───────────────────────┐
                          │  RANDOM FOREST        │
                          │  REGRESSION MODULE    │
                          │  • Train/test split   │
                          │  • Hyperparameter     │
                          │    tuning             │
                          │  • Model training     │
                          │  • Feature importance │
                          └───────────┬───────────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │  EVALUATION MODULE    │
                          │  • R², RMSE, MAE      │
                          │  • Actual vs Predicted│
                          │  • Residual analysis  │
                          │  • Baseline comparison│
                          └───────────┬───────────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │  VISUALISATION &      │
                          │  REPORTING MODULE     │
                          │  • Charts & plots     │
                          │  • Choropleth maps    │
                          │  • Final report       │
                          │  • Dashboard (opt.)   │
                          └───────────────────────┘
```

---

## 10. Module Breakdown

### Module 1: Data Collection & Integration
- **Input**: Raw CSV files from multiple sources
- **Process**: Load, inspect, clean column names, standardise district/state names, merge on common keys
- **Output**: Unified master DataFrame

### Module 2: Data Preprocessing
- **Input**: Master DataFrame
- **Process**: Handle nulls, remove duplicates, treat outliers, encode categoricals, scale numericals, engineer features
- **Output**: Clean, analysis-ready DataFrame

### Module 3: Exploratory Data Analysis
- **Input**: Clean DataFrame
- **Process**: Statistical summaries, correlation analysis, distribution plots, trend analysis
- **Output**: EDA visualisations and insights report

### Module 4: K-Means Soil Profiling
- **Input**: Soil feature subset (pH, N, P, K, OC, moisture)
- **Process**: Feature scaling → Elbow/Silhouette analysis → K-Means fit → PCA visualisation
- **Output**: Cluster labels appended to DataFrame, cluster summary statistics

### Module 5: Random Forest Regression
- **Input**: Full feature set including cluster labels
- **Process**: Train/test split → Model training → Hyperparameter tuning → Feature importance extraction
- **Output**: Trained model, predictions, feature importance rankings

### Module 6: Model Evaluation
- **Input**: Actual and predicted yield values
- **Process**: Compute metrics, generate diagnostic plots, compare with baselines
- **Output**: Evaluation metrics table, diagnostic visualisations

### Module 7: Visualisation & Reporting
- **Input**: All intermediate and final results
- **Process**: Generate publication-quality charts, optional dashboard, compile report
- **Output**: Final project report, presentation materials

---

## 11. Use Cases

### UC-01: Data Scientist Trains Yield Prediction Model
| Aspect | Detail |
|--------|--------|
| **Actor** | Data Scientist / Student |
| **Precondition** | Raw datasets are available in CSV format |
| **Flow** | 1. Load datasets → 2. Preprocess → 3. Run EDA → 4. Perform clustering → 5. Train RF model → 6. Evaluate → 7. Generate report |
| **Postcondition** | Trained model with evaluation metrics and visualisations |

### UC-02: Agricultural Planner Queries Yield Forecast
| Aspect | Detail |
|--------|--------|
| **Actor** | Agricultural Planner |
| **Precondition** | Trained model is available |
| **Flow** | 1. Input district soil & weather parameters → 2. System assigns soil cluster → 3. RF model predicts yield → 4. Display prediction with confidence |
| **Postcondition** | District-level yield prediction displayed |

### UC-03: Policy Maker Analyses Regional Patterns
| Aspect | Detail |
|--------|--------|
| **Actor** | Policy Maker |
| **Precondition** | Model outputs and cluster analysis are complete |
| **Flow** | 1. View soil profile clusters on map → 2. Compare predicted yields across districts → 3. Identify low-yield, high-potential areas → 4. Export insights |
| **Postcondition** | Actionable insights for resource allocation |

---

## 12. Risk Analysis

| # | Risk | Impact | Probability | Mitigation |
|---|------|--------|-------------|------------|
| R1 | Inconsistent district names across datasets | High | High | Create district name mapping/normalisation dictionary |
| R2 | High missing data ratio in soil/weather datasets | High | Medium | Use multiple imputation; fallback to state-level averages |
| R3 | Insufficient data for rare crops or remote districts | Medium | Medium | Focus on major crops; aggregate rare categories |
| R4 | Overfitting of Random Forest model | High | Medium | Use cross-validation, limit tree depth, tune hyperparameters |
| R5 | Temporal data leakage | High | Low | Ensure chronological train/test split where applicable |
| R6 | Computational resource constraints | Low | Low | Use sampling for large datasets; leverage Google Colab GPUs |

---

## 13. Constraints & Assumptions

### Constraints
- Project is limited to **publicly available** datasets
- No real-time data pipeline — batch processing only
- Prediction granularity is **district-level** (not field/farm-level)
- Development environment limited to Python ecosystem

### Assumptions
- Historical yield data is reasonably accurate and representative
- Soil characteristics remain relatively stable over the analysis period (5–10 years)
- Weather data is available at district or nearest weather station level
- Irrigation data reflects actual ground conditions
- District boundaries and names remain consistent across datasets (or can be mapped)

---

## 14. Deliverables

| # | Deliverable | Format | Description |
|---|------------|--------|-------------|
| D1 | Requirement Analysis Document | MD/PDF | This document |
| D2 | Cleaned & Merged Dataset | CSV | Final analysis-ready dataset |
| D3 | Jupyter Notebook(s) | .ipynb | Complete code with EDA, clustering, modelling, evaluation |
| D4 | Trained Model | .pkl/.joblib | Serialised Random Forest model |
| D5 | Visualisation Outputs | PNG/HTML | All charts, heatmaps, and cluster plots |
| D6 | Project Report | PDF/DOCX | Comprehensive report with methodology, results, and conclusions |
| D7 | Presentation | PPTX | Summary slides for project demonstration |
| D8 | Source Code Repository | GitHub | Version-controlled codebase with README |

---

## 15. Timeline (Indicative)

| Phase | Duration | Activities |
|-------|----------|------------|
| Phase 1: Planning & Data Collection | Week 1–2 | Requirements finalisation, dataset sourcing, literature review |
| Phase 2: Data Preprocessing & EDA | Week 3–4 | Cleaning, merging, feature engineering, exploratory analysis |
| Phase 3: Clustering & Modelling | Week 5–6 | K-Means profiling, Random Forest training, hyperparameter tuning |
| Phase 4: Evaluation & Visualisation | Week 7 | Model evaluation, diagnostic plots, choropleth maps |
| Phase 5: Reporting & Presentation | Week 8 | Report writing, dashboard (optional), final presentation |

---

## 16. Glossary

| Term | Definition |
|------|-----------|
| **Crop Yield** | Production per unit area (tonnes/hectare) |
| **K-Means Clustering** | Unsupervised ML algorithm that partitions data into k groups based on feature similarity |
| **Random Forest Regression** | Ensemble supervised ML method using multiple decision trees for continuous value prediction |
| **EDA** | Exploratory Data Analysis — initial investigation of data to discover patterns |
| **PCA** | Principal Component Analysis — dimensionality reduction technique for visualisation |
| **R² Score** | Coefficient of determination; proportion of variance explained by the model (1.0 = perfect) |
| **RMSE** | Root Mean Squared Error — average magnitude of prediction errors |
| **MAE** | Mean Absolute Error — average absolute prediction error |
| **NPK** | Nitrogen, Phosphorus, Potassium — primary soil macronutrients |
| **Silhouette Score** | Metric measuring how similar a point is to its own cluster vs. neighbouring clusters |
| **Choropleth Map** | Thematic map where areas are shaded proportional to a variable's value |
| **IMD** | India Meteorological Department |
| **ICRISAT** | International Crops Research Institute for the Semi-Arid Tropics |

---

> **Document Version**: 1.0  
> **Prepared By**: HCL Project Group 9  
> **Date**: June 2026  
> **Status**: Draft — Awaiting Review
