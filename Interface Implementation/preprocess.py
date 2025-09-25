import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# -------------------------------
# Binary columns
# -------------------------------
BINARY_COLS = [
    'Gender', 'Senior Citizen', 'Partner', 'Dependents',
    'Phone Service', 'Multiple Lines', 'Online Security', 'Online Backup',
    'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies',
    'Paperless Billing', 'Secure_Contract'
]

BINARY_MAP = {
    'Yes': 1,
    'No': 0,
    'Male': 1,
    'Female': 0,
    'No phone service': 0,
    'No internet service': 0
}

# -------------------------------
# Multi-class categorical columns
# -------------------------------
MULTI_COLS = ['Internet Service', 'Contract', 'Payment Method', 'Tenure Category']

# -------------------------------
# Numeric columns
# -------------------------------
NUMERIC_COLS = ['Tenure Months', 'Monthly Charges', 'Total Charges', 'CLTV', 'Cost_per_Month']

# -------------------------------
# Final selected features (from training)
# -------------------------------
FINAL_FEATURES = [
    'CLTV', 'Paperless Billing', 'Tenure Months', 'Secure_Contract',
    'Monthly Charges', 'Total Charges', 'Cost_per_Month', 'Multiple Lines',
    'Partner', 'Streaming Movies', 'Online Security', 'Gender', 'Device Protection',
    'Senior Citizen', 'Online Backup', 'Streaming TV', 'Dependents', 'Tech Support',
    'Internet Service_DSL', 'Internet Service_Fiber optic', 'Internet Service_No',
    'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
    'Payment Method_Bank transfer (automatic)', 'Payment Method_Credit card (automatic)',
    'Payment Method_Electronic check', 'Payment Method_Mailed check',
    'Tenure Category_Long', 'Tenure Category_Mid', 'Tenure Category_Short'
]

# -------------------------------
# Preprocess a dataframe
# -------------------------------
def preprocess_input(df):
    df_processed = df.copy()

    # Strip whitespace from object columns
    for col in df_processed.select_dtypes(include='object').columns:
        df_processed[col] = df_processed[col].str.strip()

    # Map binary columns
    for col in BINARY_COLS:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].map(BINARY_MAP).fillna(0)

    # One-hot encode multi-class categorical columns
    for col in MULTI_COLS:
        if col in df_processed.columns:
            df_processed = pd.get_dummies(df_processed, columns=[col], drop_first=False)

    # Handle missing numeric values
    for col in NUMERIC_COLS:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
        else:
            df_processed[col] = 0

    # Numeric scaling
    scaler = StandardScaler()
    df_processed[NUMERIC_COLS] = scaler.fit_transform(df_processed[NUMERIC_COLS])

    # Align columns with final selected features
    for col in FINAL_FEATURES:
        if col not in df_processed.columns:
            df_processed[col] = 0  # missing column → fill with 0

    # Drop extra columns not in FINAL_FEATURES
    extra_cols = [col for col in df_processed.columns if col not in FINAL_FEATURES]
    if extra_cols:
        df_processed.drop(columns=extra_cols, inplace=True)

    # Reorder columns
    df_processed = df_processed[FINAL_FEATURES]

    return df_processed

# -------------------------------
# Preprocess single row input
# -------------------------------
def preprocess_single_input(input_dict):
    df = pd.DataFrame([input_dict])
    return preprocess_input(df)
