# Physics-Informed Clustering & Class-Balanced ML for Industrial Pump Health Monitoring

A research-oriented machine learning workflow for industrial pump health monitoring that combines **exploratory data analysis, statistical characterization, per-pump clustering, physics-informed labeling, class-balanced supervised learning, Bayesian hyperparameter optimization, and explainable AI (LIME)**.

> **Research status:** The associated paper is a preprint and is explicitly marked as **not peer-reviewed**.

## Project Overview

Industrial pump predictive maintenance is difficult because real failure labels are usually scarce, operating conditions vary across assets, and the boundary between normal operation and early degradation can be gradual rather than discrete.

This project develops a multi-stage framework that first discovers operational regimes from sensor data and then maps those regimes to physically meaningful health states. The workflow is designed to reduce dependence on retrospective failure labels and to provide an interpretable path from raw pump telemetry to maintenance-oriented health classification.

### Core workflow

```text
Pump sensor / operational data
            |
            v
Exploratory Data Analysis
            |
            v
Statistical characterization
            |
            v
Feature preprocessing / Z-score scaling
            |
            v
Per-pump unsupervised clustering
            |
            v
Cluster statistical validation
(ANOVA + Tukey HSD)
            |
            v
Physics-Informed Labeling (PIL)
            |
            v
Health-state labels
Healthy / Normal / Impending Failure / End of Life
            |
            v
Class-balanced supervised ML
            |
            v
Bayesian hyperparameter optimization
            |
            v
LIME-based local explanations
```

## Repository Contents

```text
.
├── README.md
├── Large_Industrial_Pump_Predictive_Analysis.ipynb
└── Physics_Informed_Cluster.pdf
```

### `Large_Industrial_Pump_Predictive_Analysis.ipynb`

The notebook contains the analytical workflow for the large industrial pump dataset, including:

- Data loading and dataset inspection
- Missing-value and duplicate checks
- Descriptive statistics
- Skewness and kurtosis analysis
- Univariate distribution and box-plot analysis
- Q-Q plots and Shapiro-Wilk normality testing
- IQR-based outlier analysis
- Pairwise feature analysis
- Correlation analysis
- Hopkins clusterability analysis
- Maintenance-flag comparisons
- Per-pump data separation
- Standardization using `StandardScaler`
- K-Means clustering and cluster-profile analysis for individual pumps

The supplied notebook currently contains **per-pump K-Means experimentation using three clusters in its clustering cells**. The associated research paper extends this work into a four-state physics-informed framework.

### `Physics_Informed_Cluster.pdf`

The research paper associated with this repository:

**A Physics-Informed Clustering and Class-Balanced Machine Learning Framework for Industrial Pump Health Monitoring**

Authors: **Aravindan Natarajan and Aravind Kumar Arunagiri**

Posted: **29 July 2026**

DOI: **10.20944/preprints202607.2098.v1**

## Dataset

The study uses a multivariate industrial pump dataset containing:

- 20,000 observations
- 5 pump assets
- Temperature
- Vibration
- Pressure
- Flow Rate
- RPM
- Operational Hours
- Maintenance Flag

The research paper notes that the original dataset does not disclose complete pump configuration, sensor specifications, and measurement units. The study therefore treats the dataset primarily as a proof-of-concept environment for the proposed methodology.

### Data source

The original dataset is available from Kaggle:

https://www.kaggle.com/datasets/selonamaris/large-industrial-pump-maintenance-dataset

**Important:** Do not commit the raw dataset to this repository unless you have verified that its license and redistribution terms permit it.

## Methodology

### 1. Exploratory Data Analysis

The initial analysis establishes data integrity and statistical characteristics before clustering or predictive modeling.

The notebook examines:

- Dataset shape and schema
- Null values
- Duplicate records
- Descriptive statistics
- Feature distributions
- Box plots
- Q-Q plots
- Shapiro-Wilk tests
- IQR-based outlier detection
- Pearson correlation
- Pair plots
- Pump-wise maintenance distributions

### 2. Preprocessing and Feature Engineering

The research framework applies Z-score normalization to the continuous operational variables.

An additional interaction feature is used to represent coupled thermal and mechanical effects:

```text
Temperature × Vibration
```

The supplied notebook uses `StandardScaler` and separates `Pump_ID` and `Maintenance_Flag` from the variables being scaled.

### 3. Clusterability Assessment

The Hopkins statistic is used to assess whether the data exhibit a natural tendency toward clustering.

This is important because a clustering algorithm will produce partitions even when the underlying feature space has weak or no natural cluster structure.

### 4. Per-Pump Clustering

The framework emphasizes **localized clustering by pump asset** rather than assuming all pumps share an identical degradation trajectory.

K-Means / K-Means++ is used to identify latent operational regimes.

The research paper reports selecting the cluster count using the elbow method and applying localized clustering separately to each pump.

### 5. Statistical Cluster Validation

After clustering, the framework evaluates whether the clusters are statistically distinct using:

- One-way ANOVA
- Tukey HSD post-hoc testing

This step is intended to determine which sensor variables actually distinguish the identified operational regimes.

### 6. Physics-Informed Labeling (PIL)

The unsupervised clusters are mapped to domain-level health states using pump and reliability engineering knowledge.

The four states used in the research framework are:

| Health State | Interpretation |
|---|---|
| **Healthy** | Early-life / near-Best-Efficiency-Point operating condition |
| **Normal** | Stable operation within expected operating envelopes |
| **Impending Failure** | Early degradation indicated by localized thermal or vibration anomalies |
| **End of Life** | Wear-out / advanced lifecycle condition |

The paper relates the labeling logic to the **bathtub failure-rate curve** and **pump performance characteristics**.

### 7. Supervised Classification

The research framework evaluates multiple classifiers, including:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- Balanced Random Forest
- Easy Ensemble
- Balanced Bagging

Because diagnostic states may be imbalanced, evaluation emphasizes metrics that are less sensitive to class prevalence than raw accuracy.

### 8. Evaluation Metrics

The research framework reports:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Matthews Correlation Coefficient (MCC)

The paper uses MCC as the primary benchmark for class-imbalanced multi-class evaluation.

### 9. Bayesian Hyperparameter Optimization

Hyperparameter optimization is performed using **Hyperopt** with the **Tree-structured Parzen Estimator (TPE)**.

The optimization objective is tied to cross-validated MCC rather than accuracy alone.

### 10. Explainable AI

**LIME (Local Interpretable Model-agnostic Explanations)** is used to provide instance-level explanations of model predictions.

The intent is to translate a predicted health class into interpretable sensor-level evidence that maintenance and reliability engineers can investigate.

## Reported Research Results

For the optimized test-set evaluation reported in the paper, the following metrics are provided for the final models:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | MCC |
|---|---:|---:|---:|---:|---:|
| LightGBM | 0.5445 | 0.5352 | 0.5229 | 0.5271 | 0.3774 |
| Logistic Regression | 0.5378 | 0.5227 | 0.5164 | 0.5168 | 0.3685 |
| Balanced Random Forest | 0.5233 | 0.5156 | 0.5256 | 0.5168 | 0.3606 |
| Random Forest | 0.5315 | 0.5201 | 0.5072 | 0.5112 | 0.3589 |
| XGBoost | 0.5255 | 0.5115 | 0.5036 | 0.5065 | 0.3519 |
| Easy Ensemble | 0.5088 | 0.4972 | 0.5090 | 0.4967 | 0.3440 |
| Balanced Bagging | 0.5045 | 0.4922 | 0.4991 | 0.4939 | 0.3304 |
| Decision Tree | 0.4553 | 0.4379 | 0.4390 | 0.4383 | 0.2593 |

These values are **reported results from the associated preprint**, not a guarantee of performance on real industrial pump data.

## Important Research Limitations

The paper identifies several limitations:

- The current framework treats telemetry largely as static snapshots rather than fully modeling temporal degradation trajectories.
- The **Normal vs. Impending Failure** boundary can contain overlapping physical signatures.
- Discrete classes do not directly provide a continuous probabilistic health measure.
- The baseline framework assumes a relatively homogeneous fleet operating manifold.
- The dataset is synthetic / idealized enough that its distributions and inter-asset behavior should not be treated as a direct substitute for high-fidelity plant SCADA or condition-monitoring data.

## Future Extensions

The paper proposes extensions including:

- LSTM / GRU temporal health-state modeling
- Remaining Useful Life (RUL) estimation
- Pump-specific digital twins
- Bayesian neural networks or Gaussian-process classifiers
- Isolation Forest / deep autoencoder anomaly detection within Normal regions
- Longitudinal techno-economic validation and Total Cost of Ownership analysis

## Installation

Create a Python environment and install the main notebook dependencies:

```bash
pip install numpy pandas scipy matplotlib seaborn statsmodels scikit-learn
```

For the extended research framework, the paper additionally uses / discusses packages such as:

```bash
pip install lightgbm hyperopt imbalanced-learn lime yellowbrick
```

## Running the Notebook

1. Clone the repository.

```bash
git clone https://github.com/AAravind-sys/Physics_Informed_Clustering_Class_Balanced_ML.git
cd Physics_Informed_Clustering_Class_Balanced_ML
```

2. Open:

```text
Large_Industrial_Pump_Predictive_Analysis.ipynb
```

3. Download the dataset from Kaggle and place it in a local data directory.

4. Update the CSV path in the **Data Loading** cell. The original notebook contains a machine-specific Windows path, so this path must be changed on another system.

5. Run the notebook from top to bottom.

## Reproducibility Notes

The research paper states that experiments were developed in Python using a Google Colab environment and that exact package versions are documented in the accompanying project repository.

For a cleaner reproducible version of this repository, it is recommended to add:

```text
requirements.txt
environment.yml
```

and to replace machine-specific absolute paths with relative paths such as:

```python
from pathlib import Path

DATA_PATH = Path("data/Large_Industrial_Pump_Maintenance_Dataset.csv")
```

## Research Context

This repository is intended as a bridge between:

**Industrial condition monitoring → Statistical characterization → Unsupervised state discovery → Physics-informed health labeling → Class-balanced ML → Explainable predictive maintenance**

The central idea is to use domain knowledge to convert statistically discovered operating regimes into maintenance-oriented health states before building the final predictive classifier.

## Citation

Please cite the associated preprint when reusing the methodology, code, or analysis:

```text
Natarajan, A., & Arunagiri, A. K. (2026).
A Physics-Informed Clustering and Class-Balanced Machine Learning
Framework for Industrial Pump Health Monitoring.
Preprints.org.
https://doi.org/10.20944/preprints202607.2098.v1
```

## Authors

**Aravindan Natarajan**  
Independent Researcher, Bangalore, India

**Aravind Kumar Arunagiri**  
Bangalore, India

## License and Data

The associated preprint states that the research work is distributed under **CC BY 4.0** and that the original Kaggle dataset is available under its stated dataset license.

Before redistributing the dataset itself, verify the dataset's current license and terms.

---

### Project Status

**Research / Proof-of-Concept**

The repository is intended for methodological study, reproducibility, experimentation, and extension toward real industrial pump condition-monitoring applications.
