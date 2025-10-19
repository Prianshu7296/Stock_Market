from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,r2_score

def train_logreg(X_train, y_train, params, task_type):
    if task_type == "regression":
        from sklearn.linear_model import LinearRegression
        # Allowed params for LinearRegression
        allowed = {"fit_intercept", "copy_X", "n_jobs", "positive"}
        params = {k: v for k, v in params.items() if k in allowed}
        model = LinearRegression(**params)
    else:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return model



def evaluate_logreg(model, X_test, y_test, task_type):
    y_pred = model.predict(X_test)
    if task_type == "regression":
        score = r2_score(y_test, y_pred)
        print(f"Linear Regression R2 Score: {score:.3f}")
    else:
        score = accuracy_score(y_test, y_pred)
        print(f"Logistic Regression Accuracy: {score:.3f}")
    return score