import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    r2_score, mean_absolute_error, mean_squared_error
)
from tqdm import tqdm

from src.data_processing import get_processed_data
from src.feature_engineering import feature_engineering_pipeline
from src.models import random_forest, logistic_regression, svm, decision_tree, gradient_boosting
from src.utils import load_config, save_model

def get_task_type(y):
    values = np.unique(y.dropna())
    if y.dtype.kind in "ifc" and len(values) > 10:
        return "regression"
    elif len(values) == 2 and set(values) <= {0, 1}:
        return "binary"
    elif y.dtype.kind in "iu" and len(values) > 2:
        return "multiclass"
    else:
        return "unknown"

def main():
    config = load_config("configs/config.YAML")
    X, y = get_processed_data(config['data_path'], target_column=config['target_column'])
    X = feature_engineering_pipeline(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['test_size'], random_state=config['random_seed'])

    task_type = get_task_type(y_train)
    print(f"Auto-detected task type: {task_type}")

    models = {
        "Random Forest": (random_forest.train_rf, config['models']['random_forest']),
        "Logistic Regression": (logistic_regression.train_logreg, config['models']['logistic_regression']),
        "SVM": (svm.train_svm, config['models']['svm'][task_type]),
        "Decision Tree": (decision_tree.train_tree, config['models']['decision_tree']),
        "Gradient Boosting": (gradient_boosting.train_gb, config['models']['gradient_boosting']),
    }

    results = []

    for name, (train_fn, params) in tqdm(models.items(), desc="Training Models"):
        print(f"\nTraining {name}...")
        model = train_fn(X_train, y_train, params, task_type)
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test)

        if task_type == "binary":
            # Optional: Handle predict_proba absence
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision(Negative)": precision_score(y_test, y_pred, pos_label=0),
                "Precision(Positive)": precision_score(y_test, y_pred, pos_label=1),
                "Recall(Negative)": recall_score(y_test, y_pred, pos_label=0),
                "Recall(Positive)": recall_score(y_test, y_pred, pos_label=1),
                "F1(Negative)": f1_score(y_test, y_pred, pos_label=0),
                "F1(Positive)": f1_score(y_test, y_pred, pos_label=1),
                "ROC_AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else "NA"
            })
        elif task_type == "regression":
            results.append({
                "Model": name,
                "R2_Score": r2_score(y_test, y_pred),
                "MAE": mean_absolute_error(y_test, y_pred),
                "MSE": mean_squared_error(y_test, y_pred)
            })

        model_path = os.path.join(config['model_dir'], f"{name.lower().replace(' ', '_')}_model.pkl")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        save_model(model, model_path)
        print(f"Saved {name} model to {model_path}")

    os.makedirs(config['results_dir'], exist_ok=True)
    results_path = os.path.join(config['results_dir'], "model_metrics.xlsx")
    results_df = pd.DataFrame(results)
    results_df.to_excel(results_path, index=False)
    print(f"Saved evaluation results to {results_path}")

if __name__ == "__main__":
    main()


 
