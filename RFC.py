import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('RFC.csv')


X = df.drop(columns=['fake'])
y = df['fake']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}


rf_classifier = RandomForestClassifier(random_state=42)


grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy')


grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")

best_rf_model = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                       max_depth=best_params['max_depth'],
                                       min_samples_split=best_params['min_samples_split'],
                                       random_state=42)
best_rf_model.fit(X_train, y_train)

test_accuracy_before = best_rf_model.score(X_test, y_test)
print(f"Test Accuracy Before Fine-Tuning: {test_accuracy_before}")

test_accuracy_after = grid_search.best_estimator_.score(X_test, y_test)
print(f"Test Accuracy After Fine-Tuning: {test_accuracy_after}")

feature_importance = best_rf_model.feature_importances_
print("Feature Importance:")
for feature, importance in zip(X.columns, feature_importance):
    print(f"{feature}: {importance}")

joblib.dump(best_rf_model, 'best_rf_model.joblib')
