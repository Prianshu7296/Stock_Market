from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,r2_score

def train_gb(X_train, y_train, params, task_type):
    if task_type == "regression":
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(**params)
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)
    return model



def evaluate_gb(model, X_test, y_test, task_type):
    y_pred = model.predict(X_test)
    if task_type == "regression":
        score = r2_score(y_test, y_pred)
        print(f"Gradient Boosting R2 Score: {score:.3f}")
    else:
        score = accuracy_score(y_test, y_pred)
        print(f"Gradient Boosting Accuracy: {score:.3f}")
    return score
