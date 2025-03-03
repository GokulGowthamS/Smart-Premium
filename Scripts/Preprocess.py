import pandas as pd
import numpy as np
import pickle
import os

class DataPreprocessor:

    def __init__(self, data):
        self.data = data

    def drop_columns(self, columns_to_drop):
        self.data.drop(columns_to_drop, axis=1, errors='ignore', inplace=True)

    def fill_missing_values(self):
        for col in self.data.select_dtypes(include=['int64', 'float64']).columns:
            self.data[col].fillna(self.data[col].mean(), inplace=True)
        for col in self.data.select_dtypes(include='object').columns:
            self.data[col].fillna(self.data[col].mode()[0], inplace=True)

    def create_feature_groups(self):
        def age_category(data):
            if 18 < data <= 30: return 0
            elif 30 < data <= 40: return 1
            elif 40 < data <= 50: return 2
            elif 50 < data <= 64: return 3
            else: return 4

        def dependent_category(data):
            if data == 0: return 0
            elif 0 < data <= 2: return 1
            elif 2 < data <= 3: return 2
            else: return 3

        def health_category(data):
            if 0 < data <= 15: return 0
            elif 15 < data <= 25: return 1
            elif 25 < data <= 35: return 2
            else: return 3

        def claims(data):
            if 0 < data <= 1: return 0
            elif 1 < data <= 2: return 1
            else: return 2

        def vehicle(data):
            if 0 < data <= 5: return 0
            elif 5 < data <= 10: return 1
            elif 10 < data <= 20: return 2
            else: return 3

        def credit(data):
            if 0 < data <= 300: return 0
            elif 300 < data <= 600: return 1
            elif 600 < data < 800: return 2
            else: return 3

        def insurance(data):
            if 0 < data <= 3: return 0
            elif 3 < data <= 6: return 1
            elif 6 < data < 9: return 2
            else: return 3

        self.data['Age'] = self.data['Age'].apply(age_category).astype(int)
        self.data['Number of Dependents'] = self.data['Number of Dependents'].apply(dependent_category).astype(int)
        self.data['Health Score'] = self.data['Health Score'].apply(health_category).astype(int)
        self.data['Previous Claims'] = self.data['Previous Claims'].apply(claims).astype(int)
        self.data['Vehicle Age'] = self.data['Vehicle Age'].apply(vehicle).astype(int)
        self.data['Credit Score'] = self.data['Credit Score'].apply(credit).astype(int)
        self.data['Insurance Duration'] = self.data['Insurance Duration'].apply(insurance).astype(int)

    def replace_mappings(self):
        mappings = {
            "Education Level": {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3},
            "Customer Feedback": {"Poor": 0, "Average": 1, "Good": 2},
            "Exercise Frequency": {"Rarely": 0, "Weekly": 1, "Monthly": 2, "Daily": 3},
            "Policy Type": {"Basic": 0, "Comprehensive": 1, "Premium": 2}
        }
        self.data.replace(mappings, inplace=True)

    def encode_categorical_columns(self):
        mappings = {
            "Gender": {"Male": 0, "Female": 1},
            "Marital Status": {"Single": 0, "Married": 1},
            "Occupation": {"UnEmployed": 0, "Self-Employed": 1, "Employed": 2},
            "Location": {"Urban": 0, "Suburban": 1, "Rural": 2},
            "Smoking Status": {"Non-Smoker": 0, "Smoker": 1},
            "Property Type": {"Apartment": 0, "Condo": 1, "House": 2}
        }
        for col, mapping in mappings.items():
            if col in self.data.columns:
                self.data[col] = self.data[col].map(mapping).fillna(-1).astype(int)
    
    def ensure_positive_values(self, columns):
        for col in columns:
            min_val = self.data[col].min()
            if min_val < 0:
                self.data[col] = self.data[col] - min_val + 1

    def log_transform(self, columns_to_transform):
        self.ensure_positive_values(columns_to_transform)
        for col in columns_to_transform:
            self.data[col] = np.log1p(self.data[col])

    def preprocess(self):
        self.drop_columns(['id', 'Policy Start Date'])
        self.fill_missing_values()
        self.create_feature_groups()
        self.replace_mappings()
        self.encode_categorical_columns()
        self.log_transform(['Annual Income'])
        return self.data

# Load data dynamically
data_path = os.path.join(os.getcwd(), 'playground-series-s4e12', 'train.csv')
data = pd.read_csv(data_path)
preprocessor = DataPreprocessor(data)
processed_data = preprocessor.preprocess()

# Save preprocessor
pickle_path = os.path.join(os.getcwd(), 'pickles', 'preprocessor.pkl')
with open(pickle_path, 'wb') as file:
    pickle.dump(preprocessor, file)

print("Preprocessor saved successfully.")
