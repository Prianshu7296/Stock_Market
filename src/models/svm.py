from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

class SVMPipeline:
    def __init__(self, params, task_type):
        self.task_type = task_type
        self.scaler = StandardScaler()
        self.is_regression = task_type == "regression"
        # Store the params for grid search
        self.params = params

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        if self.is_regression:
            grid = GridSearchCV(
                SVR(),
                self.params,
                cv=3,
                scoring="r2"
            )
            grid.fit(X_scaled, y)
            self.model = grid.best_estimator_
        else:
            grid = GridSearchCV(
                SVC(probability=True),
                self.params,
                cv=3,
                scoring="accuracy"
            )
            grid.fit(X_scaled, y)
            self.model = grid.best_estimator_

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_scaled)
        return None

def train_svm(X_train, y_train, params, task_type):
    model = SVMPipeline(params, task_type)
    model.fit(X_train, y_train)
    return model
