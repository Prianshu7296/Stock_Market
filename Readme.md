### Stock Market Movement Prediction Pipeline

Overview:
A modular, production-ready machine learning pipeline for predicting annual stock price movements and key financial metrics (such as Earnings Per Share, returns, or valuation ratios) for NSE 500 companies.
The system is fully configurable through a single YAML file and requires no code changes to switch between targets, enable/disable models, or adjust hyperparameters. Every stage—from data preprocessing to model evaluation—is reproducible, automated, and designed for scalability.

Data Source:
This project uses comprehensive financial data from Prowess, a database by CMIE (Centre for Monitoring Indian Economy) containing detailed financial statements, stock prices, and operational metrics for Indian companies including the NSE 500.

 “Dataset not included due to privacy/compliance. The pipeline, code, and config are fully documented. Provide your own data in the specified format to run end-to-end".
 
### Key Highlights:

 - Automatic Task Detection: Intelligently switches between classification (stock up/down) and regression (continuous value prediction) based on your target column
 - Zero-Code Configuration: Fully configurable through a single YAML file—switch targets, models, and hyperparameters without touching code
 - Modular Architecture: Clean separation of preprocessing, feature engineering, training, and evaluation##
 - Production-Ready: Docker containerization + CI/CD with GitHub Actions
 - Comprehensive Evaluation: ROC-AUC, F1/F2, Precision/Recall for classification; R², MAE, RMSE for regression
 - Reproducible: Every stage is logged, versioned, and designed for scalability

### Sample Results:
### Classification Performance: ROC Curve
<img width="2400" height="1600" alt="image" src="https://github.com/user-attachments/assets/648b0f7b-9d53-47af-98ef-b7b16c08d2bd" />



ROC Curve comparing model performance for stock movement classification.  
Gradient Boosting and Random Forest achieved the highest AUC (>0.85),  
outperforming linear and kernel-based classifiers.  
Dashed line shows random chance baseline (AUC=0.5).

### Quick Start
Prerequisites:
Python 3.11 or higher
pip package manager
Docker (optional, for containerized deployment)

Installation
Clone the repository
```bash
git clone https://github.com/...
```
cd stock-market-pipeline

Install dependencies
```bash
 pip install -r requirements.txt
```

- Prepare your data
- Place your financial data CSV in data/


Configure the pipeline
- bash# Edit config/config.yaml
- nano config/config.yaml

Usage:
Run the complete pipeline:
```bash
python main.py
```

View results:
- Metrics: results/metrics/

<pre>
Stock_Market/
|
|-- data/                           # Raw and processed datasets
|
|-- config/
|   |-- config.yaml                 # Pipeline settings, file paths, targets, splits, and model parameters
|
|-- src/
|   |-- data_preprocessing.py       # Cleans, imputes missing values, filters outliers
|   |-- feature_engineering.py      # Constructs domain-specific and statistical features
|   |-- models/
|   |   |-- random_forest.py
|   |   |-- logistic_regression.py
|   |   |-- svm.py
|   |   |-- decision_tree.py
|   |   |-- gradient_boosting.py
|   |
|   |-- evaluation.py               # Model scoring, visual reports, and metric aggregation
|   |-- utils.py                    # Config loader, logger, I/O helpers, and serialization routines
|
|-- results/                        # Auto-generated model metrics, plots, and evaluation summaries
|
|-- main.py                         # End-to-end orchestrator integrating all modules via config
|
|-- notebooks/
|   |-- main.ipynb                  # Interactive demo / visualization notebook
|
|-- requirements.txt                # Project dependencies
|-- README.md
|-- Dockerfile                      # For deployment purposes
|-- .gitignore
</pre>

### Pipeline Description:
1. Configuration (config.YAML)
All settings are centralized here, data paths, target column(s), model toggles, hyperparameters, and output options.

2. Data Preprocessing (src/data_preprocessing.py)
Cleans and merges multiple financial data sources.
Handles missing values via smart imputation (mean/median or interpolation).
Removes duplicate and high-correlation features using Variance Inflation Factor (VIF) thresholds.
Detects and filters statistical outliers via interquartile range or z-score techniques.

3. Feature Engineering (src/feature_engineering.py)
Generates domain-specific and temporal features, including:
Rolling averages and lag features.
Ratio-based metrics like P/E, Debt/Equity, and margin percentages.
Macro-economic indicators such as GDP growth, CPI/Inflation, and rupee-dollar exchange trends.
Correlates company-specific and market-wide data to produce a rich input set for model training.

4. Model Training (src/models/)
Modularized models for different families:
Tree-based: Random Forest, Decision Tree, Gradient Boosting
Linear: Logistic Regression, Linear Regression
Kernel-based: Support Vector Machines (SVM)
Automatically detects the problem type from configuration:
For classification: predicts movement (stock up/down, outperform/underperform)
For regression: predicts continuous values (EPS, returns, ratios)
Hyperparameters are dynamically loaded from config.yaml.
Trained model objects are saved as .pkl files under models

5. Evaluation & Visualization (src/evaluation.py)
For classification:
Accuracy, Precision, Recall, F1 Score, ROC-AUC, Confusion Matrix
For regression:
R², MAE, RMSE, and Residual plots
Generates visualizations such as ROC curves, correlation heatmaps, and prediction residuals.
Saves metrics and visual outputs into the /results/ directory for easy tracking and comparison.
Supports both batch (CLI) and interactive (Jupyter notebook) display modes.

6. Pipeline Orchestration (main.py)
main.py acts as the master controller. It:
Loads configuration from config/config.yaml
Invokes data preprocessing and feature engineering
Trains all enabled models according to the configuration
Evaluates performance and saves outputs
Logs every step for reproducibility and debugging

7. Interactive Testing (notebooks/main.ipynb)
Use main.ipynb for:
Step-by-step walkthrough of each stage.
Interactive feature importance visualization.
Comparative benchmarking of models.
Exploratory analysis of correlation matrices, data distributions, and prediction outcomes.

Deployment:
(Docker Support)
This project can be run in a fully reproducible Docker container. No Python installation required—just Docker
Quickstart:
docker build -t stock-pipeline .
docker run stock-pipeline
See the provided Dockerfile for full build steps and customization.

Continuous Integration (CI/CD)
This repository uses [GitHub Actions](https://docs.github.com/en/actions) for Continuous Integration.
Every push or pull request automatically runs the full pipeline with the latest code and dependencies. See the `.github/workflows/python-app.yml` workflow for details.
