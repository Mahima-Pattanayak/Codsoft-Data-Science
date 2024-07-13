from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def tune_model(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_
