"""
Data utilities for loan default prediction system.

This module handles data loading, validation, and sample data generation.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union
from pathlib import Path

def load_data(file_path: Union[str, 'streamlit.uploaded_file_manager.UploadedFile']) -> pd.DataFrame:
    """
    Load loan data from CSV file.
    
    Args:
        file_path: Path to CSV file or uploaded file object
        
    Returns:
        pd.DataFrame: Loaded loan data
        
    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file doesn't exist
    """
    try:
        if hasattr(file_path, 'read'):
            # It's an uploaded file from Streamlit
            df = pd.read_csv(file_path)
        else:
            # It's a file path
            df = pd.read_csv(file_path)
        
        # Basic validation
        if df.empty:
            raise ValueError("The uploaded file is empty")
        
        if len(df) < 10:
            st.warning("Dataset is very small. Consider using more data for better predictions.")
        
        return df
        
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty or corrupted")
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")

def generate_sample_data(n_samples: int = 1000, random_state: Optional[int] = 42) -> pd.DataFrame:
    """
    Generate realistic sample loan data for demonstration purposes.
    
    This function creates synthetic data that mimics real-world loan applications
    with realistic correlations between features and default rates.
    
    Args:
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Generated sample data
    """
    np.random.seed(random_state)
    
    # Generate base features
    age = np.random.normal(35, 10, n_samples)
    age = np.clip(age, 18, 75).astype(int)
    
    # Income with realistic distribution (right-skewed)
    income = np.random.lognormal(10.5, 0.8, n_samples)
    income = np.clip(income, 15000, 250000).astype(int)
    
    # Employment length
    employment_length = np.random.choice(
        range(0, 21), 
        size=n_samples, 
        p=[0.05, 0.08, 0.10, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 
           0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01, 0.02]
    )
    
    # Loan amount (correlated with income)
    loan_amount = income * np.random.uniform(0.5, 3.0, n_samples)
    loan_amount = np.clip(loan_amount, 1000, 50000).astype(int)
    
    # Interest rate (based on credit risk factors)
    base_rate = 5.0
    risk_premium = np.random.uniform(0, 15, n_samples)
    interest_rate = base_rate + risk_premium
    
    # Loan term
    loan_term = np.random.choice([12, 24, 36, 48, 60, 72, 84], size=n_samples)
    
    # Credit score (inversely related to some risk factors)
    credit_score = 850 - (risk_premium * 5) - np.random.uniform(0, 100, n_samples)
    credit_score = np.clip(credit_score, 300, 850).astype(int)
    
    # Debt-to-income ratio
    debt_to_income = np.random.beta(2, 5, n_samples) * 60
    debt_to_income = np.clip(debt_to_income, 0, 60)
    
    # Home ownership
    home_ownership = np.random.choice(
        ['OWN', 'RENT', 'MORTGAGE', 'OTHER'], 
        size=n_samples,
        p=[0.25, 0.45, 0.25, 0.05]
    )
    
    # Purpose
    purpose = np.random.choice(
        ['debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase', 
         'small_business', 'medical', 'vacation', 'wedding', 'moving', 'other'],
        size=n_samples,
        p=[0.35, 0.20, 0.15, 0.08, 0.07, 0.05, 0.03, 0.03, 0.02, 0.02]
    )
    
    # Create default target variable with realistic relationships
    # Higher risk factors increase default probability
    risk_score = (
        (850 - credit_score) / 550 * 0.3 +  # Credit score impact
        (debt_to_income / 60) * 0.2 +        # DTI impact
        (interest_rate / 20) * 0.2 +         # Interest rate impact
        (employment_length == 0) * 0.1 +     # Unemployment impact
        (loan_amount / income > 2) * 0.1 +   # High loan-to-income ratio
        np.random.uniform(0, 0.1)             # Random noise
    )
    
    default = (risk_score > 0.5).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'employment_length': employment_length,
        'loan_amount': loan_amount,
        'interest_rate': interest_rate,
        'loan_term': loan_term,
        'credit_score': credit_score,
        'debt_to_income': debt_to_income,
        'home_ownership': home_ownership,
        'purpose': purpose,
        'default': default
    })
    
    return df

def validate_loan_data(df: pd.DataFrame) -> dict:
    """
    Validate loan data for required columns and data quality.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        dict: Validation results with status and messages
    """
    validation_results = {
        'is_valid': True,
        'warnings': [],
        'errors': []
    }
    
    # Check for required columns
    required_columns = ['age', 'income', 'loan_amount', 'credit_score', 'default']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        validation_results['errors'].append(f"Missing required columns: {missing_columns}")
        validation_results['is_valid'] = False
    
    # Check for missing values
    if df.isnull().any().any():
        missing_cols = df.columns[df.isnull().any()].tolist()
        validation_results['warnings'].append(f"Columns with missing values: {missing_cols}")
    
    # Check data types
    numeric_columns = ['age', 'income', 'loan_amount', 'credit_score']
    for col in numeric_columns:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            validation_results['errors'].append(f"Column '{col}' must be numeric")
            validation_results['is_valid'] = False
    
    # Check for reasonable ranges
    if 'age' in df.columns:
        if (df['age'] < 18).any() or (df['age'] > 100).any():
            validation_results['warnings'].append("Age values outside reasonable range (18-100)")
    
    if 'credit_score' in df.columns:
        if (df['credit_score'] < 300).any() or (df['credit_score'] > 850).any():
            validation_results['warnings'].append("Credit scores outside FICO range (300-850)")
    
    if 'income' in df.columns:
        if (df['income'] < 0).any():
            validation_results['errors'].append("Income cannot be negative")
            validation_results['is_valid'] = False
    
    # Check class balance
    if 'default' in df.columns:
        default_rate = df['default'].mean()
        if default_rate < 0.05 or default_rate > 0.95:
            validation_results['warnings'].append(f"Extreme class imbalance: {default_rate:.1%} default rate")
    
    return validation_results

def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for the dataset.
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        dict: Summary statistics
    """
    summary = {
        'n_samples': len(df),
        'n_features': len(df.columns),
        'n_numerical': len(df.select_dtypes(include=[np.number]).columns),
        'n_categorical': len(df.select_dtypes(include=['object']).columns),
        'missing_values': df.isnull().sum().sum(),
        'default_rate': df['default'].mean() if 'default' in df.columns else None,
        'avg_loan_amount': df['loan_amount'].mean() if 'loan_amount' in df.columns else None,
        'avg_income': df['income'].mean() if 'income' in df.columns else None,
        'avg_credit_score': df['credit_score'].mean() if 'credit_score' in df.columns else None
    }
    
    return summary