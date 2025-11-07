from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, r2_score

def train_tree(X_train, y_train, params, task_type):
    if task_type == "regression":
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor(**params)
    else:
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)
    return model



def evaluate_tree(model, X_test, y_test, task_type):
    y_pred = model.predict(X_test)
    if task_type == "regression":
        score = r2_score(y_test, y_pred)
        print(f"Decision Tree R2 Score: {score:.3f}")
    else:
        score = accuracy_score(y_test, y_pred)
        print(f"Decision Tree Accuracy: {score:.3f}")
    return score
