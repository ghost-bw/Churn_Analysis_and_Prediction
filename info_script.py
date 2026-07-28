import os
import joblib
import pandas as pd

output_lines = []

def log(msg):
    print(msg)
    output_lines.append(str(msg))

# 1. Inspect model
model_path = 'models/logistic_regression_best_model.joblib'
if not os.path.exists(model_path):
    model_path = 'models/logistic_regression_best_model.pkl'

log(f"Model path found: {model_path}")

try:
    if model_path.endswith('.joblib'):
        model = joblib.load(model_path)
    else:
        import pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    log(f"Successfully loaded model. Model class: {type(model)}")
    if hasattr(model, 'steps'):
        log("Model is a Pipeline with steps:")
        for step in model.steps:
            log(f"  {step[0]}: {type(step[1])}")
except Exception as e:
    log(f"Error loading model: {e}")

# 2. Inspect dataset
data_paths = ['data/exported_churn_data.csv', 'data/cleaned_churn_data.csv']
data_path = None
for dp in data_paths:
    if os.path.exists(dp):
        data_path = dp
        break

log(f"Dataset path found: {data_path}")

if data_path:
    try:
        df = pd.read_csv(data_path)
        log(f"Dataset shape: {df.shape}")
        log("Columns:")
        log(list(df.columns))
        
        # Categorical columns unique values
        categorical_features = ['subscription_type', 'plan_type', 'contract_type', 'country', 'state', 'gender', 'escalations']
        log("\nUnique values for categorical features:")
        for col in categorical_features:
            if col in df.columns:
                log(f"  {col}: {df[col].dropna().unique().tolist()}")
            else:
                log(f"  {col}: NOT FOUND IN DATASET")
                
        # Numeric columns summary
        numeric_features = ['monthly_charges', 'cltv', 'churn_score', 'csat_score', 'complaint_count']
        log("\nSummary of numeric features:")
        for col in numeric_features:
            if col in df.columns:
                log(f"  {col}: min={df[col].min()}, max={df[col].max()}, median={df[col].median()}")
            else:
                log(f"  {col}: NOT FOUND IN DATASET")
    except Exception as e:
        log(f"Error reading dataset: {e}")

with open('info.txt', 'w') as f:
    f.write('\n'.join(output_lines))
log("info.txt written successfully")
