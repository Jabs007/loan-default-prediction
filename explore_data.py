"""
Data exploration script for Stage 2 - Data Understanding
This script demonstrates how to examine loan data for ML modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_utils import generate_sample_data, validate_loan_data, get_data_summary

def explore_loan_data():
    """Comprehensive data exploration for loan default prediction."""
    
    print("🏦 LOAN DEFAULT PREDICTION - DATA UNDERSTANDING")
    print("=" * 60)
    
    # Step 1: Generate sample data (or load real data)
    print("\n📊 STEP 1: Loading Data")
    print("-" * 30)
    
    # Generate realistic sample data
    df = generate_sample_data(n_samples=1000, random_state=42)
    print(f"✅ Generated {len(df)} loan applications")
    
    # Step 2: Basic Data Validation
    print("\n🔍 STEP 2: Data Validation")
    print("-" * 30)
    
    validation = validate_loan_data(df)
    
    if validation['is_valid']:
        print("✅ Data validation PASSED")
    else:
        print("❌ Data validation FAILED")
        for error in validation['errors']:
            print(f"  • {error}")
    
    if validation['warnings']:
        print("⚠️  Warnings:")
        for warning in validation['warnings']:
            print(f"  • {warning}")
    
    # Step 3: Data Summary
    print("\n📈 STEP 3: Data Summary")
    print("-" * 30)
    
    summary = get_data_summary(df)
    
    print(f"📋 Dataset Overview:")
    print(f"  • Total samples: {summary['n_samples']:,}")
    print(f"  • Total features: {summary['n_features']}")
    print(f"  • Numerical features: {summary['n_numerical']}")
    print(f"  • Categorical features: {summary['n_categorical']}")
    print(f"  • Missing values: {summary['missing_values']}")
    
    print(f"\n💰 Financial Overview:")
    print(f"  • Default rate: {summary['default_rate']:.1%}")
    print(f"  • Average loan amount: ${summary['avg_loan_amount']:,.0f}")
    print(f"  • Average income: ${summary['avg_income']:,.0f}")
    print(f"  • Average credit score: {summary['avg_credit_score']:.0f}")
    
    # Step 4: Detailed Feature Analysis
    print("\n🔎 STEP 4: Feature Analysis")
    print("-" * 30)
    
    # Identify feature types
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    target_variable = 'default'
    
    # Remove target from feature lists
    if target_variable in numerical_features:
        numerical_features.remove(target_variable)
    
    print(f"🎯 Target Variable: '{target_variable}' (Binary: 0=Good, 1=Default)")
    print(f"🔢 Numerical Features ({len(numerical_features)}):")
    for feature in numerical_features:
        print(f"  • {feature}")
    
    print(f"🏷️  Categorical Features ({len(categorical_features)}):")
    for feature in categorical_features:
        print(f"  • {feature}")
    
    # Step 5: Missing Values Analysis
    print("\n🔍 STEP 5: Missing Values Analysis")
    print("-" * 30)
    
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        print("Missing values found:")
        for col, missing in missing_data[missing_data > 0].items():
            print(f"  • {col}: {missing} ({missing/len(df)*100:.1f}%)")
    else:
        print("✅ No missing values detected")
    
    # Step 6: Class Imbalance Analysis
    print("\n⚖️  STEP 6: Class Imbalance Analysis")
    print("-" * 30)
    
    default_counts = df['default'].value_counts()
    print(f"Class distribution:")
    print(f"  • Non-default (0): {default_counts[0]} ({default_counts[0]/len(df)*100:.1f}%)")
    print(f"  • Default (1): {default_counts[1]} ({default_counts[1]/len(df)*100:.1f}%)")
    
    # Imbalance ratio
    imbalance_ratio = default_counts[0] / default_counts[1]
    print(f"  • Imbalance ratio: {imbalance_ratio:.1f}:1")
    
    if imbalance_ratio > 10:
        print("⚠️  SEVERE class imbalance detected - may need special handling")
    elif imbalance_ratio > 5:
        print("⚠️  Moderate class imbalance - consider resampling techniques")
    else:
        print("✅ Reasonable class balance")
    
    # Step 7: Statistical Summary
    print("\n📊 STEP 7: Statistical Summary")
    print("-" * 30)
    
    print("Numerical Features Summary:")
    numerical_summary = df[numerical_features].describe()
    print(numerical_summary.round(2))
    
    print("\nCategorical Features Summary:")
    for feature in categorical_features:
        print(f"\n{feature}:")
        print(df[feature].value_counts().head())
    
    # Step 8: Data Quality Flags
    print("\n🚩 STEP 8: Data Quality Flags")
    print("-" * 30)
    
    flags = []
    
    # Check for outliers in key financial variables
    for col in ['income', 'loan_amount', 'credit_score']:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                flags.append(f"{col}: {outliers} potential outliers detected")
    
    # Check for impossible values
    if (df['age'] < 18).any():
        flags.append("Age: Applicants under 18 detected")
    
    if (df['credit_score'] < 300).any() or (df['credit_score'] > 850).any():
        flags.append("Credit Score: Values outside FICO range (300-850)")
    
    if (df['debt_to_income'] > 100).any():
        flags.append("Debt-to-Income: Values > 100% detected")
    
    if flags:
        print("Data quality issues identified:")
        for flag in flags:
            print(f"  ⚠️  {flag}")
    else:
        print("✅ No major data quality issues detected")
    
    print("\n" + "=" * 60)
    print("🏁 Data Understanding Complete!")
    print("Next: Exploratory Data Analysis (EDA)")
    
    return df

if __name__ == "__main__":
    # Run the exploration
    df = explore_loan_data()
    
    # Save sample data for later use
    df.to_csv('data/sample_loan_data.csv', index=False)
    print(f"\n💾 Sample data saved to: data/sample_loan_data.csv")