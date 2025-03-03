import streamlit as st
import pandas as pd
import numpy as np
import pickle
import traceback
import warnings
warnings.filterwarnings('ignore')
from Scripts.Preprocess import DataPreprocessor

@st.cache_data
def load_model():
    with open("pickles/best_model.pkl", "rb") as file:
        return pickle.load(file)

@st.cache_data
def load_preprocessor():
    with open("pickles/preprocessor.pkl", "rb") as file:
        return pickle.load(file)

def preprocess_user_input(user_input, preprocessor):
    user_input["Annual Income"] = np.log1p(user_input["Annual Income"])
    processed_data = preprocessor.preprocess()
    return processed_data

def main():
    st.title("Insurance Premium Prediction App")
    st.write("Enter the details below to get a premium prediction")
    
    age = st.slider("Age", 18, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    annual_income = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=50000)
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    dependents = st.slider("Number of Dependents", 0, 10, 2)
    education_level = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
    occupation = st.selectbox("Occupation", ["UnEmployed", "Self-Employed", "Employed"])
    health_score = st.slider("Health Score", 0, 100, 50)
    location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
    policy_type = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
    previous_claims = st.slider("Previous Claims", 0, 5, 1)
    vehicle_age = st.slider("Vehicle Age", 0, 20, 5)
    credit_score = st.slider("Credit Score", 300, 850, 600)
    insurance_duration = st.slider("Insurance Duration (Years)", 1, 10, 5)
    customer_feedback = st.selectbox("Customer Feedback", ["Poor", "Average", "Good"])
    smoking_status = st.selectbox("Smoking Status", ["Smoker", "Non-Smoker"])
    exercise_frequency = st.selectbox("Exercise Frequency", ["Rarely", "Weekly", "Monthly", "Daily"])
    property_type = st.selectbox("Property Type", ["Apartment", "Condo", "House"])

    user_input = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Annual Income": annual_income,
        "Marital Status": marital_status,
        "Number of Dependents": dependents,
        "Education Level": education_level,
        "Occupation": occupation,
        "Health Score": health_score,
        "Location": location,
        "Policy Type": policy_type,
        "Previous Claims": previous_claims,
        "Vehicle Age": vehicle_age,
        "Credit Score": credit_score,
        "Insurance Duration": insurance_duration,
        "Customer Feedback": customer_feedback,
        "Smoking Status": smoking_status,
        "Exercise Frequency": exercise_frequency,
        "Property Type": property_type
    }])

    if st.button("Predict Premium"):
        try:
            model = load_model()
            preprocessor = load_preprocessor()
            
            processed_data = preprocess_user_input(user_input, preprocessor)
            processed_data = processed_data[model.feature_names_in_]
            
            prediction = model.predict(processed_data)
            prediction = np.expm1(prediction)
            
            st.success(f"Predicted Premium Amount: ${prediction[0]:.2f}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.text(traceback.format_exc())

if __name__ == "__main__":
    main()