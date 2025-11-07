from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,r2_score

def train_rf(X_train, y_train, params, task_type):
    if task_type == "regression":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(**params)
    else:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model



def evaluate_rf(model, X_test, y_test, task_type):
    y_pred = model.predict(X_test)
    if task_type == "regression":
        score = r2_score(y_test, y_pred)
        print(f"Random Forest R2 Score: {score:.3f}")
    else:
        score = accuracy_score(y_test, y_pred)
        print(f"Random Forest Accuracy: {score:.3f}")
    return score
