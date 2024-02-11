import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

df = pd.read_csv('RFC.csv')
X = df.drop(columns=['fake'])
y = df['fake']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lgb_model = lgb.LGBMClassifier(objective='binary', metric='binary_logloss', random_state=42)
lgb_model.fit(X_train, y_train)

y_pred = lgb_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy with LightGBM: {test_accuracy}")

feature_importance = lgb_model.feature_importances_
print("Feature Importance:")
for feature, importance in zip(X.columns, feature_importance):
    print(f"{feature}: {importance}")
