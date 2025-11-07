from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
def save_metrics_to_excel(results, file_path):
    """
    Save dictionary of model evaluation results to an Excel file.
    'results' should be dict of {model_name: {metric_name: value}}.
    """
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_excel(file_path)
    print(f"Saved evaluation results to {file_path}")

def evaluate_classification(y_true, y_pred, y_proba=None):
    """
    Print and return main classification metrics.
    """
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    if y_proba is not None:
        print("ROC AUC:", roc_auc_score(y_true, y_proba))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba) if y_proba is not None else None
    }
