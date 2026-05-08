"""
Feature engineering utilities for loan default prediction.

This module creates meaningful financial features that improve model performance
by capturing domain-specific knowledge about credit risk.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import math

def create_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create financial ratio features that indicate creditworthiness.
    
    These ratios are commonly used in credit risk assessment and provide
    meaningful insights into an applicant's financial health.
    
    Args:
        df: DataFrame with base financial features
        
    Returns:
        pd.DataFrame: DataFrame with financial ratio features added
    """
    
    df_ratios = df.copy()
    
    # 1. Loan-to-Income Ratio (LTI)
    # Measures loan size relative to annual income
    # Higher ratio = higher risk
    if 'loan_amount' in df_ratios.columns and 'income' in df_ratios.columns:
        df_ratios['loan_to_income_ratio'] = df_ratios['loan_amount'] / df_ratios['income']
        
        # Create risk categories
        df_ratios['lti_risk_category'] = pd.cut(
            df_ratios['loan_to_income_ratio'],
            bins=[0, 0.5, 1.0, 2.0, float('inf')],
            labels=['low', 'medium', 'high', 'very_high']
        )
    
    # 2. Monthly Payment Burden
    # Estimates monthly loan payment as percentage of monthly income
    if all(col in df_ratios.columns for col in ['loan_amount', 'interest_rate', 'loan_term', 'income']):
        # Calculate monthly payment using loan amortization formula
        monthly_rate = df_ratios['interest_rate'] / 100 / 12
        n_payments = df_ratios['loan_term']
        
        # Avoid division by zero
        monthly_rate = np.maximum(monthly_rate, 0.001)
        
        # Monthly payment calculation: M = P[r(1+r)^n]/[(1+r)^n-1]
        numerator = monthly_rate * np.power(1 + monthly_rate, n_payments)
        denominator = np.power(1 + monthly_rate, n_payments) - 1
        monthly_payment = df_ratios['loan_amount'] * (numerator / denominator)
        
        monthly_income = df_ratios['income'] / 12
        df_ratios['monthly_payment_burden'] = (monthly_payment / monthly_income) * 100
        
        # Risk categories for payment burden
        df_ratios['payment_burden_category'] = pd.cut(
            df_ratios['monthly_payment_burden'],
            bins=[0, 10, 20, 30, float('inf')],
            labels=['low', 'medium', 'high', 'very_high']
        )
    
    # 3. Debt Service Coverage Ratio (DSCR)
    # Measures ability to service debt from income
    # Higher ratio = better ability to pay
    if 'debt_to_income' in df_ratios.columns:
        df_ratios['dscr'] = 1 / (df_ratios['debt_to_income'] / 100 + 0.01)  # Add small constant to avoid division by zero
        
        # Create categories
        df_ratios['dscr_category'] = pd.cut(
            df_ratios['dscr'],
            bins=[0, 1, 2, 5, float('inf')],
            labels=['poor', 'fair', 'good', 'excellent']
        )
    
    # 4. Credit Utilization Proxy
    # Combines credit score and debt burden
    if 'credit_score' in df_ratios.columns and 'debt_to_income' in df_ratios.columns:
        # Normalize credit score to 0-1 scale (higher is better)
        credit_score_norm = (df_ratios['credit_score'] - 300) / (850 - 300)
        
        # Invert debt-to-income (lower is better)
        dti_inverse = 1 - (df_ratios['debt_to_income'] / 100)
        
        # Combined credit health score
        df_ratios['credit_health_score'] = (credit_score_norm * 0.7 + dti_inverse * 0.3) * 100
        
        # Risk categories
        df_ratios['credit_health_category'] = pd.cut(
            df_ratios['credit_health_score'],
            bins=[0, 40, 60, 80, 100],
            labels=['poor', 'fair', 'good', 'excellent']
        )
    
    return df_ratios

def create_employment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create employment-related features that indicate income stability.
    
    Employment stability is a key factor in credit risk assessment.
    
    Args:
        df: DataFrame with employment features
        
    Returns:
        pd.DataFrame: DataFrame with employment features added
    """
    
    df_emp = df.copy()
    
    # 1. Employment Stability Score
    if 'employment_length' in df_emp.columns:
        # Create employment stability categories
        df_emp['employment_stability'] = pd.cut(
            df_emp['employment_length'],
            bins=[-1, 0, 1, 3, 5, float('inf')],
            labels=['unemployed', 'new', 'junior', 'experienced', 'senior']
        )
        
        # Binary stability indicator
        df_emp['is_employed'] = (df_emp['employment_length'] > 0).astype(int)
        df_emp['is_stable_employed'] = (df_emp['employment_length'] >= 2).astype(int)
        
        # Income stability proxy (employment length * income)
        if 'income' in df_emp.columns:
            df_emp['income_stability_proxy'] = df_emp['employment_length'] * np.log(df_emp['income'])
    
    # 2. Career Stage Indicator
    if 'age' in df_emp.columns and 'employment_length' in df_emp.columns:
        # Calculate career stage based on age and employment
        df_emp['career_stage'] = np.where(
            df_emp['age'] < 25, 'early',
            np.where(df_emp['age'] < 35, 'establishing',
                np.where(df_emp['age'] < 50, 'peak', 'senior'))
        )
        
        # Employment consistency (should increase with age)
        expected_emp_length = np.maximum(0, df_emp['age'] - 22)  # Assume work starts at 22
        df_emp['employment_consistency'] = np.minimum(
            1.0, df_emp['employment_length'] / (expected_emp_length + 1)
        )
    
    return df_emp

def create_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create credit-specific features that indicate creditworthiness.
    
    These features capture nuanced aspects of credit risk beyond basic scores.
    
    Args:
        df: DataFrame with credit features
        
    Returns:
        pd.DataFrame: DataFrame with credit features added
    """
    
    df_credit = df.copy()
    
    # 1. Credit Score Tiers
    if 'credit_score' in df_credit.columns:
        df_credit['credit_tier'] = pd.cut(
            df_credit['credit_score'],
            bins=[299, 579, 669, 739, 799, 850],
            labels=['poor', 'fair', 'good', 'very_good', 'exceptional']
        )
        
        # Credit score relative to age group (expected credit maturity)
        if 'age' in df_credit.columns:
            # Expected credit score based on age (simplified model)
            expected_score = np.minimum(850, 300 + df_credit['age'] * 8)
            df_credit['credit_score_vs_expected'] = df_credit['credit_score'] - expected_score
            
            # Credit improvement potential
            df_credit['credit_improvement_potential'] = np.maximum(0, 850 - df_credit['credit_score'])
    
    # 2. Risk Score Combinations
    if 'credit_score' in df_credit.columns and 'debt_to_income' in df_credit.columns:
        # Multiplicative risk score (lower is better)
        credit_factor = (850 - df_credit['credit_score']) / 550  # 0 to 1
        dti_factor = df_credit['debt_to_income'] / 100  # 0 to 1
        df_credit['combined_risk_score'] = credit_factor * dti_factor
        
        # Weighted risk score
        df_credit['weighted_risk_score'] = credit_factor * 0.7 + dti_factor * 0.3
    
    # 3. Credit Maturity Indicators
    if 'age' in df_credit.columns:
        # Credit history length proxy (assumes credit starts at 18)
        df_credit['credit_history_length'] = np.maximum(0, df_credit['age'] - 18)
        
        # Credit maturity score
        df_credit['credit_maturity_score'] = np.minimum(1.0, df_credit['credit_history_length'] / 15)
    
    return df_credit

def create_loan_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create loan-specific features that capture loan risk characteristics.
    
    Args:
        df: DataFrame with loan features
        
    Returns:
        pd.DataFrame: DataFrame with loan features added
    """
    
    df_loan = df.copy()
    
    # 1. Loan Risk Indicators
    if 'loan_amount' in df_loan.columns and 'income' in df_loan.columns:
        # Large loan indicator (relative to income)
        df_loan['is_large_loan'] = (df_loan['loan_amount'] > df_loan['income']).astype(int)
        
        # Loan size category
        df_loan['loan_size_category'] = pd.cut(
            df_loan['loan_amount'],
            bins=[0, 5000, 15000, 30000, float('inf')],
            labels=['small', 'medium', 'large', 'jumbo']
        )
    
    # 2. Interest Rate Risk
    if 'interest_rate' in df_loan.columns:
        # High interest rate indicator
        df_loan['is_high_interest'] = (df_loan['interest_rate'] > 15).astype(int)
        
        # Interest rate category
        df_loan['interest_rate_category'] = pd.cut(
            df_loan['interest_rate'],
            bins=[0, 8, 12, 16, float('inf')],
            labels=['low', 'medium', 'high', 'very_high']
        )
        
        # Rate vs market rate (simplified)
        market_rate = 8.0  # Assume 8% is market rate
        df_loan['rate_vs_market'] = df_loan['interest_rate'] - market_rate
    
    # 3. Loan Term Features
    if 'loan_term' in df_loan.columns:
        # Long term loan indicator
        df_loan['is_long_term'] = (df_loan['loan_term'] > 48).astype(int)
        
        # Loan term category
        df_loan['loan_term_category'] = pd.cut(
            df_loan['loan_term'],
            bins=[0, 24, 48, 72, float('inf')],
            labels=['short', 'medium', 'long', 'very_long']
        )
        
        # Monthly payment amount (approximate)
        if 'loan_amount' in df_loan.columns and 'interest_rate' in df_loan.columns:
            monthly_rate = df_loan['interest_rate'] / 100 / 12
            n_payments = df_loan['loan_term']
            numerator = monthly_rate * np.power(1 + monthly_rate, n_payments)
            denominator = np.power(1 + monthly_rate, n_payments) - 1
            df_loan['approx_monthly_payment'] = df_loan['loan_amount'] * (numerator / denominator)
    
    # 4. Purpose-based Risk
    if 'purpose' in df_loan.columns:
        # Map loan purposes to risk levels (simplified industry knowledge)
        purpose_risk_map = {
            'debt_consolidation': 'high',
            'credit_card': 'high', 
            'small_business': 'high',
            'medical': 'medium',
            'major_purchase': 'medium',
            'home_improvement': 'medium',
            'moving': 'low',
            'vacation': 'low',
            'wedding': 'low',
            'other': 'medium'
        }
        
        df_loan['purpose_risk_level'] = df_loan['purpose'].map(purpose_risk_map)
        
        # Binary high-risk indicator
        df_loan['is_high_risk_purpose'] = (df_loan['purpose_risk_level'] == 'high').astype(int)
    
    return df_loan

def create_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregate features that combine multiple risk factors.
    
    Args:
        df: DataFrame with base features
        
    Returns:
        pd.DataFrame: DataFrame with aggregate features added
    """
    
    df_agg = df.copy()
    
    # 1. Overall Risk Score
    risk_components = []
    
    if 'loan_to_income_ratio' in df_agg.columns:
        # Normalize LTI to 0-1 risk scale
        lti_risk = np.minimum(1.0, df_agg['loan_to_income_ratio'] / 3.0)
        risk_components.append(lti_risk)
    
    if 'debt_to_income' in df_agg.columns:
        # DTI risk (normalize to 0-1)
        dti_risk = df_agg['debt_to_income'] / 60.0  # Assume 60% is max reasonable DTI
        risk_components.append(dti_risk)
    
    if 'credit_score' in df_agg.columns:
        # Credit score risk (invert and normalize)
        credit_risk = (850 - df_agg['credit_score']) / 550
        risk_components.append(credit_risk)
    
    if 'interest_rate' in df_agg.columns:
        # Interest rate risk (normalize to 0-1)
        rate_risk = np.minimum(1.0, (df_agg['interest_rate'] - 5) / 20.0)
        risk_components.append(rate_risk)
    
    if risk_components:
        # Average risk score
        df_agg['overall_risk_score'] = np.mean(risk_components, axis=0)
        
        # Risk category
        df_agg['overall_risk_category'] = pd.cut(
            df_agg['overall_risk_score'],
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=['low', 'medium', 'high', 'very_high']
        )
    
    # 2. Financial Health Score
    health_components = []
    
    if 'income' in df_agg.columns:
        # Income level (normalize to 0-1)
        income_health = np.minimum(1.0, (np.log(df_agg['income']) - 9) / 3.0)
        health_components.append(income_health)
    
    if 'employment_length' in df_agg.columns:
        # Employment stability (normalize to 0-1)
        emp_health = np.minimum(1.0, df_agg['employment_length'] / 10.0)
        health_components.append(emp_health)
    
    if 'credit_score' in df_agg.columns:
        # Credit health (normalize to 0-1)
        credit_health = (df_agg['credit_score'] - 300) / 550
        health_components.append(credit_health)
    
    if health_components:
        df_agg['financial_health_score'] = np.mean(health_components, axis=0)
        
        # Health category
        df_agg['financial_health_category'] = pd.cut(
            df_agg['financial_health_score'],
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=['poor', 'fair', 'good', 'excellent']
        )
    
    # 3. Final Recommendation Score
    if 'overall_risk_score' in df_agg.columns and 'financial_health_score' in df_agg.columns:
        # Combine risk and health scores for final recommendation
        df_agg['recommendation_score'] = (
            df_agg['financial_health_score'] * 0.6 - 
            df_agg['overall_risk_score'] * 0.4
        )
        
        # Recommendation categories
        df_agg['recommendation'] = pd.cut(
            df_agg['recommendation_score'],
            bins=[-1, -0.1, 0.1, 0.3, 1.0],
            labels=['reject', 'review', 'approve_caution', 'approve']
        )
    
    return df_agg

def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps to create a comprehensive feature set.
    
    This is the main function that orchestrates all feature engineering steps.
    
    Args:
        df: Raw DataFrame with base features
        
    Returns:
        pd.DataFrame: DataFrame with all engineered features
    """
    
    # Start with base dataframe
    df_engineered = df.copy()
    
    # Apply all feature engineering steps
    st.info("Creating financial ratios...")
    df_engineered = create_financial_ratios(df_engineered)
    
    st.info("Creating employment features...")
    df_engineered = create_employment_features(df_engineered)
    
    st.info("Creating credit features...")
    df_engineered = create_credit_features(df_engineered)
    
    st.info("Creating loan-specific features...")
    df_engineered = create_loan_specific_features(df_engineered)
    
    st.info("Creating aggregate features...")
    df_engineered = create_aggregate_features(df_engineered)
    
    # Log feature engineering summary
    original_features = len(df.columns)
    final_features = len(df_engineered.columns)
    new_features = final_features - original_features
    
    st.success(f"Feature engineering complete! Added {new_features} new features ({final_features} total)")
    
    return df_engineered