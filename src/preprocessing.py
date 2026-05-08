"""
Data preprocessing utilities for loan default prediction.

This module handles data cleaning, scaling, encoding, and feature selection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Optional
import joblib
from pathlib import Path
import streamlit as st

def create_preprocessing_pipeline(
    numerical_features: List[str], 
    categorical_features: List[str],
    target_column: str = 'default'
) -> ColumnTransformer:
    """
    Create a preprocessing pipeline for loan data.
    
    This pipeline handles:
    - Missing value imputation
    - Categorical encoding
    - Numerical scaling
    - Feature selection
    
    Args:
        numerical_features: List of numerical column names
        categorical_features: List of categorical column names  
        target_column: Name of target variable (not processed)
        
    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    
    # Numerical pipeline
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ], remainder='drop')
    
    return preprocessor

def preprocess_data(
    df: pd.DataFrame, 
    target_column: str = 'default',
    fit_pipeline: bool = True,
    pipeline_path: Optional[str] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[ColumnTransformer]]:
    """
    Preprocess loan data for machine learning.
    
    Args:
        df: Raw loan data DataFrame
        target_column: Name of target variable
        fit_pipeline: Whether to fit the pipeline (True for training, False for inference)
        pipeline_path: Path to save/load preprocessing pipeline
        
    Returns:
        Tuple of (features, target, pipeline)
        - features: Preprocessed feature matrix
        - target: Target vector (None if target_column not in df)
        - pipeline: Fitted pipeline (None if fit_pipeline=False)
    """
    
    # Separate features and target
    if target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column].values
    else:
        X = df
        y = None
    
    # Identify feature types
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target from feature lists if present
    numerical_features = [f for f in numerical_features if f != target_column]
    categorical_features = [f for f in categorical_features if f != target_column]
    
    # Load existing pipeline or create new one
    if not fit_pipeline and pipeline_path and Path(pipeline_path).exists():
        preprocessor = joblib.load(pipeline_path)
        st.success("Loaded existing preprocessing pipeline")
    else:
        preprocessor = create_preprocessing_pipeline(
            numerical_features, categorical_features, target_column
        )
    
    # Fit and transform data
    if fit_pipeline:
        X_processed = preprocessor.fit_transform(X)
        
        # Save pipeline if path provided
        if pipeline_path:
            Path(pipeline_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(preprocessor, pipeline_path)
            st.success(f"Saved preprocessing pipeline to {pipeline_path}")
    else:
        X_processed = preprocessor.transform(X)
    
    return X_processed, y, preprocessor if fit_pipeline else None

def handle_missing_values(df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
    """
    Handle missing values in loan data.
    
    Args:
        df: DataFrame with missing values
        strategy: Strategy for handling missing values ('auto', 'drop', 'impute')
        
    Returns:
        pd.DataFrame: Data with missing values handled
    """
    
    df_clean = df.copy()
    
    if strategy == 'auto':
        # Automatic strategy selection based on missing percentage
        missing_pct = df_clean.isnull().sum() / len(df_clean)
        
        for col in df_clean.columns:
            if missing_pct[col] > 0:
                if missing_pct[col] > 0.5:  # More than 50% missing
                    # Drop column
                    df_clean = df_clean.drop(columns=[col])
                    st.warning(f"Dropped column '{col}' due to high missing rate ({missing_pct[col]:.1%})")
                elif missing_pct[col] > 0.1:  # More than 10% missing
                    # Impute based on data type
                    if df_clean[col].dtype in ['object']:
                        mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'unknown'
                        df_clean[col] = df_clean[col].fillna(mode_val)
                    else:
                        median_val = df_clean[col].median()
                        df_clean[col] = df_clean[col].fillna(median_val)
                else:  # Less than 10% missing
                    # Forward fill for small amounts
                    df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
    
    elif strategy == 'drop':
        # Drop rows with any missing values
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna()
        st.info(f"Dropped {initial_rows - len(df_clean)} rows with missing values")
    
    elif strategy == 'impute':
        # Impute all missing values
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                if df_clean[col].dtype in ['object']:
                    mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'unknown'
                    df_clean[col] = df_clean[col].fillna(mode_val)
                else:
                    median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val)
    
    return df_clean

def detect_outliers(df: pd.DataFrame, columns: List[str], method: str = 'iqr') -> pd.DataFrame:
    """
    Detect outliers in numerical columns.
    
    Args:
        df: DataFrame to check for outliers
        columns: List of columns to check
        method: Method for outlier detection ('iqr', 'zscore')
        
    Returns:
        pd.DataFrame: Boolean mask indicating outliers
    """
    
    outlier_mask = pd.DataFrame(False, index=df.index, columns=columns)
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outlier_mask[col] = z_scores > 3
    
    return outlier_mask

def create_feature_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create meaningful feature interactions for loan data.
    
    Args:
        df: DataFrame with base features
        
    Returns:
        pd.DataFrame: DataFrame with interaction features added
    """
    
    df_features = df.copy()
    
    # Loan-to-income ratio
    if 'loan_amount' in df_features.columns and 'income' in df_features.columns:
        df_features['loan_to_income_ratio'] = df_features['loan_amount'] / df_features['income']
    
    # Interest burden (monthly interest payment as % of income)
    if all(col in df_features.columns for col in ['loan_amount', 'interest_rate', 'loan_term', 'income']):
        monthly_rate = df_features['interest_rate'] / 100 / 12
        n_payments = df_features['loan_term']
        monthly_payment = df_features['loan_amount'] * (monthly_rate * (1 + monthly_rate)**n_payments) / ((1 + monthly_rate)**n_payments - 1)
        monthly_income = df_features['income'] / 12
        df_features['interest_burden'] = (monthly_payment / monthly_income) * 100
    
    # Credit utilization proxy
    if 'credit_score' in df_features.columns and 'debt_to_income' in df_features.columns:
        df_features['credit_risk_score'] = (850 - df_features['credit_score']) / 550 * df_features['debt_to_income']
    
    # Age-income interaction (proxy for career stage)
    if 'age' in df_features.columns and 'income' in df_features.columns:
        df_features['age_income_interaction'] = df_features['age'] * np.log(df_features['income'])
    
    return df_features