# 🏆 Smart Insurance Premium Prediction

This project predicts **insurance premium amounts** using **Machine Learning models**.  
It processes raw insurance data, applies **feature engineering**, and trains multiple **regression models** to find the best-performing one.

---

## 🚀 *Key Features*
### __Data Preprocessing__  
- **Handles missing values** in both numerical and categorical columns.  
- **Removes outliers** using **IQR (Interquartile Range)**.  
- **Encodes categorical data** using Label Encoding.  
- **Feature Engineering**: Converts raw numerical data into meaningful groups (e.g., Age Groups, Credit Score Categories).  

### __Machine Learning Model Training__  
- Trains multiple models, including:
  - **Linear Regression**
  - **Decision Tree Regressor**
  - **Random Forest Regressor**
  - **XGBoost Regressor**
- **Bayesian Optimization** is used for **hyperparameter tuning**.  
- Evaluates models using:
  - **Root Mean Squared Log Error (RMSLE)**
  - **Root Mean Squared Error (RMSE)**
  - **Mean Absolute Error (MAE)**
  - **R² Score**  
- Saves the best model as **`best_model.pkl`**.  

### __Model Deployment__  
- Loads the trained model and **predicts insurance premiums on new data**.  
- Uses **MLflow** to track predictions and store model artifacts.  
- Saves final predictions to **`Test_Predictions.csv`**.  

### __Interactive Streamlit Web App__  
- A user-friendly **web interface** for entering customer details and predicting insurance premiums.  
- Displays raw input and preprocessed data for debugging.  
- Shows **real-time predictions** based on trained models.  

---

## 💻 *Technology Stack*
### __Programming Language__
- Python  

### __Machine Learning Libraries__
- **scikit-learn**  
- **XGBoost**  
- **Bayesian Optimization**  

### __Data Processing & Storage__
- **Pandas, NumPy** (for data handling and preprocessing)  
- **Pickle** (for model storage)  

### __Deployment & Tracking__
- **MLflow** (for model logging and tracking)  
- **Streamlit** (for the web application)  

---

## 📂 *Project Structure*
📜 train.csv -> Raw training dataset  
📜 test.csv -> Raw test dataset  
📜 Cleaned_data.csv -> Data after cleaning & transformation  
📜 Encoded_data.csv -> Fully processed dataset for training  
📜 best_model.pkl -> Saved best model after training  
📜 preprocessor.pkl -> Data preprocessor for prediction  
📜 Test_Predictions.csv -> Predictions on new test data  
📜 ML_Pipeline.py -> Model inference pipeline  
📜 Model_Building.py -> Script for model training  
📜 Preprocess.py -> Script for data preprocessing  
📜 Streamlit_App.py -> Streamlit UI for predictions  
📜 README.md -> Project documentation
