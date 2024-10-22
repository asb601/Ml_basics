#For predicting from the model stored in the pkl file 
import numpy as np
import pandas as pd
import pickle

# Load the trained model and scaler
with open('linear_regression_model.pkl', 'rb') as f:
    reg = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Function to predict the cost based on user input
def predict_expenses(age, sex, bmi, children, smoker, reg, scaler):
    # Create a DataFrame with the same feature names that the scaler was trained on
    feature_names = ['age', 'sex', 'bmi', 'children', 'smoker']
    user_data = pd.DataFrame([[age, sex, bmi, children, smoker]], columns=feature_names)

    # Standardize the data using the scaler
    user_data_scaled = scaler.transform(user_data)
    
    prediction = reg.predict(user_data_scaled)
    return prediction[0]

# Get user input
def get_user_input():
    try:
        age = float(input("Enter your age: "))
        sex = int(input("Enter your sex (0 for female, 1 for male): "))
        bmi = float(input("Enter your BMI (Body Mass Index): "))
        children = int(input("Enter the number of children you have: "))
        smoker = int(input("Are you a smoker? (0 for no, 1 for yes): "))
        return age, sex, bmi, children, smoker
    except ValueError:
        print("Invalid input. Please enter numerical values only.")
        return None

# Test the prediction function
if __name__ == "__main__":
    user_input = get_user_input()

    if user_input:
        age, sex, bmi, children, smoker = user_input
        predicted_cost = predict_expenses(age, sex, bmi, children, smoker, reg, scaler)
        print(f"\nPredicted expenses: ${predicted_cost:.2f}")
