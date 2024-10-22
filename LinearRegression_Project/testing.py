import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from LinearRegression.LinearRegression import LinearRegression  # Import your custom LinearRegression class
import pickle

# Load the dataset
data = pd.read_csv("./insurance.csv")

# Convert categorical columns to numerical (like sex, smoker, and region)
data['sex'] = data['sex'].map({'female': 0, 'male': 1})
data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})
data = pd.get_dummies(data, columns=['region'], drop_first=True)

# Check for missing values
if data.isnull().sum().any():
    data = data.dropna()

# Prepare the features (X) and the target (y)
X = data[['age', 'sex', 'bmi', 'children', 'smoker']]  # Select relevant features
y = data['expenses']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for future use
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1234)

# Initialize and train the Linear Regression model
reg = LinearRegression(lr=0.001, n_iters=1000)
reg.fit(X_train, y_train)

# Predict on the test set
y_pred_test = reg.predict(X_test)

# Evaluate the model (Mean Squared Error)
mse_test = mean_squared_error(y_test, y_pred_test)
print(f"Mean Squared Error on Test Data: {mse_test:.2f}")

# Save the trained model
with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(reg, f)
