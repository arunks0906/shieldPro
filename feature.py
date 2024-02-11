import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generating synthetic data for binary classification
X, y = make_classification(n_samples=100, n_features=5, n_informative=5, n_redundant=0, random_state=42)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Get feature names (assuming you have them)
feature_names = [f"Feature {i}" for i in range(X.shape[1])]

# Plotting the coefficients
plt.figure(figsize=(10, 6))
plt.barh(feature_names, model.coef_[0], color='lightblue')
plt.xlabel('Coefficient Value')
plt.title('Logistic Regression Coefficients')
plt.show()
