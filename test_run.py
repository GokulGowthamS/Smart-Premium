import pickle
import pandas as pd
import numpy as np

# Load the preprocessor
with open("F:\\Guvi Projects\\Smart_Premium\\pickles\\preprocessor.pkl", "rb") as file:
    preprocessor = pickle.load(file)

# Example input
test_input = pd.DataFrame([{
    "Age": 30,
    "Gender": "Male",
    "Annual Income": 50000,
    "Marital Status": "Single",
    "Number of Dependents": 2,
    "Education Level": "Bachelor's",
    "Occupation": "Employed",
    "Health Score": 50,
    "Location": "Urban",
    "Policy Type": "Comprehensive",
    "Previous Claims": 1,
    "Vehicle Age": 5,
    "Credit Score": 600,
    "Insurance Duration": 5,
    "Customer Feedback": "Good",
    "Smoking Status": "Non-Smoker",
    "Exercise Frequency": "Weekly",
    "Property Type": "Apartment"
}])

# Apply preprocessing
processed_input = preprocessor.__class__(test_input, is_prediction=True).preprocess()

# Check the processed input
print(processed_input.head())


# Load the trained model
with open("F:\\Guvi Projects\\Smart_Premium\\best_model.pkl", "rb") as file:
    model = pickle.load(file)

# Get prediction
prediction = model.predict(processed_input)

# Convert back from log scale (if needed)
final_prediction = np.expm1(prediction[0])

print("Predicted Premium Amount:", final_prediction)