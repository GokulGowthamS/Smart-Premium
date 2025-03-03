import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, root_mean_squared_log_error
from bayes_opt import BayesianOptimization
import pickle
import warnings 
warnings.filterwarnings('ignore')

train_file = 'F:\\Guvi Projects\\Smart_Premium\\playground-series-s4e12\\train.csv'

data = pd.read_csv(train_file)

data.shape

data.head()

data.drop(['id','Policy Start Date'], axis = 1, inplace = True)

data.head()

data.isnull().sum()

numerical_features = data.select_dtypes(include = ['int64', 'float64']).columns
categorical_features = data.select_dtypes(include = 'object').columns

for col in numerical_features:
    data[col].fillna(data[col].mean(), inplace=True)
        
for col in categorical_features:
    data[col].fillna(data[col].mode()[0], inplace=True)

data.isnull().sum()

num_col = data[numerical_features].columns
num_col

def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    return df

cleaned_data = remove_outliers_iqr(data.copy(), num_col)
cleaned_data.head()

cleaned_data.to_csv('F:\\Guvi Projects\\Smart_Premium\\research_data\\Cleaned_data.csv', index = False)

def age_category(data):
    if 18 < data <= 30:
        return '18-30'
    elif 30 < data <= 40:
        return '31-40'
    elif 40 < data <= 50:
        return '41-50'
    elif 50 < data <= 64:
        return '51-64'
    else:
        return '<64'

def dependent_category(data):
    if data == 0:
        return '0'
    elif 0 < data <= 2:
        return '0-2'
    elif 2 < data <= 3:
        return '2-3'
    else:
        return '<3'

def health_category(data):
    if 0 < data <= 15:
        return '0-15'
    elif 15 < data <= 25:
        return '15-25'
    elif 25 < data <= 35:
        return '15-35'
    else:
        return '<35'

def claims(data):
    if 0 < data <= 1:
        return '0-1'
    elif 1 < data <= 2:
        return '1-2'
    else:
        return '<2'

def vehicle(data):
    if 0 < data <= 5:
        return '0-5'
    elif 5 < data <= 10:
        return '5-10'
    elif 10 < data <= 20:
        return '10-20'
    else:
        return '<20'

def credit(data):
    if 0 < data <= 300:
        return '0-300'
    elif 300 < data <= 600:
        return '300-600'
    elif 600 < data < 800:
        return '600-800'
    else:
        return '<800'

def insurance(data):
    if 0 < data <= 3:
        return '0-3'
    elif 3 < data <= 6:
        return '3-6'
    elif 6 < data < 9:
        return '6-9'
    else:
        return '<9'

cleaned_data['Age_Group'] = cleaned_data['Age'].apply(age_category)

cleaned_data['Dependent_Group'] = cleaned_data['Number of Dependents'].apply(dependent_category)

cleaned_data['Health_Group'] = cleaned_data['Health Score'].apply(health_category)

cleaned_data['Prev_Claims_Group'] = cleaned_data['Previous Claims'].apply(claims)

cleaned_data['Vehicle_Group'] = cleaned_data['Vehicle Age'].apply(vehicle)

cleaned_data['Credit_Group'] = cleaned_data['Credit Score'].apply(credit)

cleaned_data['Insurance_Group'] = cleaned_data['Insurance Duration'].apply(insurance)

mappings = {
    "Education Level":{"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3},
    "Customer Feedback":{"Poor": 0, "Average": 1, "Good": 2},
    "Exercise Frequency":{"Rarely": 0, "Weekly": 1, "Monthly": 2, "Daily": 3 },
    "Policy Type":{"Basic": 0, "Comprehensive": 1, "Premium": 2}
}

cleaned_data.replace(mappings, inplace = True)

columns_to_encode = cleaned_data[['Age_Group', 'Dependent_Group', 'Health_Group', 'Prev_Claims_Group', 'Vehicle_Group', 'Credit_Group', 'Insurance_Group', 'Gender', 'Marital Status', 'Occupation', 'Location', 'Smoking Status', 'Property Type']]

le = LabelEncoder()

for i in columns_to_encode.columns:
    cleaned_data[i] = le.fit_transform(cleaned_data[i])

cleaned_data.head()

encoded_data = pd.DataFrame({
    'Age': cleaned_data['Age_Group'],
    'Gender': cleaned_data['Gender'],
    'Annual Income': cleaned_data['Annual Income'],
    'Marital Status': cleaned_data['Marital Status'],
    'Number of Dependents': cleaned_data['Dependent_Group'],
    'Education Level': cleaned_data['Education Level'],
    'Occupation': cleaned_data['Occupation'],
    'Health Score': cleaned_data['Health_Group'],
    'Location': cleaned_data['Location'],
    'Policy Type': cleaned_data['Policy Type'],
    'Previous Claims': cleaned_data['Prev_Claims_Group'],
    'Vehicle Age': cleaned_data['Vehicle_Group'],
    'Credit Score': cleaned_data['Credit_Group'],
    'Insurance Duration': cleaned_data['Insurance_Group'],
    'Customer Feedback': cleaned_data['Customer Feedback'],
    'Smoking Status': cleaned_data['Smoking Status'],
    'Exercise Frequency': cleaned_data['Exercise Frequency'],
    'Property Type': cleaned_data['Property Type'],
    'Premium Amount': cleaned_data['Premium Amount']
})

encoded_data.head()

encoded_data.to_csv('F:\\Guvi Projects\\Smart_Premium\\research_data\\Encoded_data.csv', index = False)

def log_transform(data, columns_to_transform):
    for col in columns_to_transform:
        data[f'{col}_log'] = np.log1p(data[col])
        data.drop(columns=[col], inplace=True)  
        data.rename(columns={f'{col}_log': col}, inplace=True)  
    
    return data

transformed_data = log_transform(encoded_data, ['Annual Income'])

transformed_data.head()

transformed_data.to_csv('F:\\Guvi Projects\\Smart_Premium\\research_data\\Transformed_Data.csv')

X = transformed_data.drop('Premium Amount', axis = 1)
Y = transformed_data['Premium Amount']

scaler = StandardScaler()
scaler.fit_transform(transformed_data[['Annual Income']])

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=45)

X_train = X_train.reshape(-1, 1) if len(X_train.shape) == 1 else X_train
X_val = X_val.reshape(-1, 1) if len(X_val.shape) == 1 else X_val

Y_train = np.log1p(Y_train)
Y_val = np.log1p(Y_val)

model_params = {
    "LinearRegression": {
        "model": LinearRegression(),
        "params": {}
    },
    "DecisionTreeRegressor": {
        "model": DecisionTreeRegressor(),
        "params": {
            "max_depth": (2, 20),
            "min_samples_split": (2, 20)
        }
    },
    "RandomForestRegressor": {
        "model": RandomForestRegressor(),
        "params": {
            "n_estimators": (10, 100),
            "max_depth": (2, 20),
            "min_samples_split": (2, 20)
        }
    },
    "XGBRegressor": {
        "model": XGBRegressor(),
        "params": {
            "n_estimators": (10, 100),
            "max_depth": (2, 20),
            "learning_rate": (0.01, 0.3)
        }
    }
}

def evaluate_model(model, params):
    model.set_params(**params)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_val)
    
    Y_val_exp = np.expm1(Y_val)
    Y_pred_exp = np.expm1(Y_pred)

    return root_mean_squared_log_error(Y_val_exp, Y_pred_exp)

results = {}

for model_name, mp in model_params.items():
    print(f"Optimizing {model_name}...")

    def objective(**params):
        if "max_depth" in params:
            params["max_depth"] = int(params["max_depth"])
        if "n_estimators" in params:
            params["n_estimators"] = int(params["n_estimators"])
        if "min_samples_split" in params:
            params["min_samples_split"] = int(params["min_samples_split"])

        valid_params = {k: v for k, v in params.items() if k in mp["params"]}

        return evaluate_model(mp["model"], valid_params)

    if mp["params"]:
        optimizer = BayesianOptimization(
            f=objective,
            pbounds=mp["params"],
            random_state=42
        )
        optimizer.maximize(init_points=5, n_iter=10)

        best_params = {k: int(v) if "depth" in k or "n_estimators" in k else v for k, v in optimizer.max["params"].items()}
        best_rmsle = optimizer.max["target"]

        best_model = mp["model"].set_params(**best_params)
    else:
        best_model = mp["model"]
        best_model.fit(X_train, Y_train)
        best_rmsle = evaluate_model(best_model, {})

    Y_pred = best_model.predict(X_val)
    Y_val_exp = np.expm1(Y_val)
    Y_pred_exp = np.expm1(Y_pred)

    results[model_name] = {
        "Best Model": best_model,
        "Best Params": best_params if mp["params"] else "Default",
        "RMSLE": best_rmsle,
        "RMSE": root_mean_squared_error(Y_val_exp, Y_pred_exp),
        "MAE": mean_absolute_error(Y_val_exp, Y_pred_exp),
        "R2 SCORE": r2_score(Y_val_exp, Y_pred_exp)
    }

best_model_name = min(results, key=lambda x: results[x]["RMSLE"])
best_model_object = results[best_model_name]["Best Model"]

for model, metrics in results.items():
    print(f"Model: {model}")
    print(f"RMSLE: {metrics['RMSLE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"R2 SCORE: {metrics['R2 SCORE']:.4f}")
    print(f"Best Parameters: {metrics['Best Params']}")
    print("--" * 10)

print(f"The Best Model: {best_model_name} with RMSLE = {results[best_model_name]['RMSLE']:.4f}")
print(f"Best model '{best_model_name}' saved to 'best_model.pkl'")

pickle_path = 'F:\\Guvi Projects\\Smart_Premium\\pickles\\best_model.pkl'

with open(pickle_path, "wb") as file:
    pickle.dump(best_model_object, file)

print('best_model.pkl saved successfully...')




