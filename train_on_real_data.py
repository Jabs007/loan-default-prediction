"""
Standalone script to train models on the real LendingClub dataset.
This script maps raw LendingClub columns to the format expected by the src modules.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys

from src.data_utils import validate_loan_data
from src.feature_engineering import engineer_all_features
from src.preprocessing import create_preprocessing_pipeline
from src.train import train_multiple_models, save_model

def map_lending_club_data(df):
    """Map raw LendingClub columns to project format."""
    print("Mapping columns...")
    
    # 1. Map target variable (default)
    # Mapping: Fully Paid = 0, Charged Off = 1
    # We ignore currently active loans for training
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
    df['default'] = (df['loan_status'] == 'Charged Off').astype(int)
    
    # 2. Map basic features
    mapping = {
        'loan_amnt': 'loan_amount',
        'annual_inc': 'income',
        'int_rate': 'interest_rate',
        'dti': 'debt_to_income',
        'purpose': 'purpose',
        'home_ownership': 'home_ownership'
    }
    df = df.rename(columns=mapping)
    
    # 3. Process credit score (average of low and high)
    df['credit_score'] = (df['fico_range_low'] + df['fico_range_high']) / 2
    
    # 4. Process loan term (e.g., " 36 months" -> 36)
    df['loan_term'] = df['term'].str.extract(r'(\d+)').astype(float)
    
    # 5. Process employment length
    def parse_emp_length(x):
        if pd.isna(x) or x == 'n/a': return 0
        if '< 1' in x: return 0
        if '10+' in x: return 10
        return int(x.split()[0])
    
    df['employment_length'] = df['emp_length'].apply(parse_emp_length)
    
    # 6. Process age (LendingClub doesn't provide age, we'll estimate based on earliest_cr_line)
    # This is a proxy: current year (2018) - year of earliest credit line + 18
    try:
        df['earliest_cr_year'] = pd.to_datetime(df['earliest_cr_line']).dt.year
        df['age'] = 2018 - df['earliest_cr_year'] + 18
        df['age'] = df['age'].fillna(35) # Default age
    except:
        df['age'] = 35 # Fallback
    
    # Keep only necessary columns
    cols_to_keep = [
        'age', 'income', 'employment_length', 'loan_amount', 
        'interest_rate', 'loan_term', 'credit_score', 
        'debt_to_income', 'home_ownership', 'purpose', 'default'
    ]
    
    return df[cols_to_keep]

def main():
    raw_data_path = 'data/raw/accepted_2007_to_2018Q4.csv.gz'
    
    if not os.path.exists(raw_data_path):
        print(f"❌ Error: Raw data not found at {raw_data_path}")
        return

    print(f"📖 Loading raw data from {raw_data_path}...")
    # Loading a sample for performance
    df_raw = pd.read_csv(raw_data_path, compression='gzip', nrows=20000)
    print(f"✅ Loaded {len(df_raw)} rows")
    
    # Map columns
    df_mapped = map_lending_club_data(df_raw)
    print(f"✅ Mapped to {len(df_mapped)} valid training samples")
    
    # Feature engineering
    print("🛠️ Performing feature engineering...")
    # Mocking streamlit info/success for the script
    import streamlit as st
    st.info = lambda x: print(f"  INFO: {x}")
    st.success = lambda x: print(f"  SUCCESS: {x}")
    
    df_engineered = engineer_all_features(df_mapped)
    
    # Prepare data for training
    X = df_engineered.drop('default', axis=1)
    y = df_engineered['default']
    
    # Preprocessing
    print("⚙️ Creating preprocessing pipeline...")
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    preprocessor = create_preprocessing_pipeline(numerical_features, categorical_features)
    
    # Train models
    print("🤖 Training models (this may take a while)...")
    models_to_train = ["Logistic Regression", "Random Forest", "XGBoost"]
    
    # Mocking more streamlit functions
    st.header = lambda x: print(f"\n--- {x} ---")
    st.spinner = lambda x: st # Just return something with a __enter__/__exit__
    class DummySpinner:
        def __enter__(self): pass
        def __exit__(self, *args): pass
    st.spinner = lambda x: DummySpinner()
    
    models, X_train, X_test, y_train, y_test = train_multiple_models(
        X, y, preprocessor, models_to_train
    )
    
    # Save models
    print("\n💾 Saving models...")
    for model_name, model in models.items():
        filename = model_name.lower().replace(" ", "_") + ".pkl"
        save_model(model, f"models/{filename}", model_name)
    
    print("\n✨ Training complete! Models saved in models/ directory.")

if __name__ == "__main__":
    main()
