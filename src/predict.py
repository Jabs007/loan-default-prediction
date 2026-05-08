"""
Prediction utilities for loan default prediction.

This module handles making predictions with trained models and provides
risk assessment functionality.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List
import streamlit as st
import joblib
from pathlib import Path

def make_prediction(model: Any, X: np.ndarray, 
                   return_probability: bool = True) -> Dict[str, Any]:
    """
    Make predictions using a trained model.
    
    Args:
        model: Trained model object
        X: Input features
        return_probability: Whether to return prediction probabilities
        
    Returns:
        Dict: Prediction results including class and probability
    """
    
    try:
        # Get prediction probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
            
            # Handle different probability formats
            if probabilities.ndim == 2 and probabilities.shape[1] == 2:
                # Binary classification with two columns
                prob_default = probabilities[:, 1]
                prob_no_default = probabilities[:, 0]
            else:
                # Single column or multi-class
                prob_default = probabilities.flatten()
                prob_no_default = 1 - prob_default
        else:
            # Models without predict_proba
            predictions = model.predict(X)
            prob_default = np.ones_like(predictions) * 0.5  # Placeholder
            prob_no_default = 1 - prob_default
        
        # Get class predictions
        if hasattr(model, 'predict'):
            predictions = model.predict(X)
        else:
            # Use probability threshold if no predict method
            predictions = (prob_default >= 0.5).astype(int)
        
        # Create results dictionary
        results = {
            'predictions': predictions,
            'probabilities': {
                'default': prob_default,
                'no_default': prob_no_default
            },
            'model_type': type(model).__name__
        }
        
        return results
        
    except Exception as e:
        st.error(f":material/cancel: Prediction failed: {str(e)}")
        return None

def assess_risk(probability: float, thresholds: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Assess risk level based on default probability.
    
    Args:
        probability: Probability of default (0-1)
        thresholds: Custom risk thresholds
        
    Returns:
        Dict: Risk assessment including level and recommendation
    """
    
    # Default thresholds if not provided
    if thresholds is None:
        thresholds = {
            'low': 0.2,
            'medium': 0.4,
            'high': 0.7
        }
    
    # Determine risk level
    if probability < thresholds['low']:
        risk_level = 'low'
        recommendation = 'APPROVE'
        color = 'green'
        description = 'Low risk applicant - recommended for approval'
    elif probability < thresholds['medium']:
        risk_level = 'medium'
        recommendation = 'APPROVE_WITH_CONDITIONS'
        color = 'yellow'
        description = 'Medium risk - consider additional conditions'
    elif probability < thresholds['high']:
        risk_level = 'high'
        recommendation = 'REVIEW_MANUALLY'
        color = 'orange'
        description = 'High risk - requires manual review'
    else:
        risk_level = 'very_high'
        recommendation = 'REJECT'
        color = 'red'
        description = 'Very high risk - recommended for rejection'
    
    # Calculate confidence (distance from nearest threshold)
    if risk_level == 'low':
        confidence = 1 - (probability / thresholds['low'])
    elif risk_level == 'very_high':
        confidence = (probability - thresholds['high']) / (1 - thresholds['high'])
    else:
        # For medium and high, use distance from boundaries
        if probability < thresholds['medium']:
            confidence = 1 - abs(probability - thresholds['low']) / (thresholds['medium'] - thresholds['low'])
        else:
            confidence = 1 - abs(probability - thresholds['high']) / (thresholds['high'] - thresholds['medium'])
    
    assessment = {
        'probability': probability,
        'risk_level': risk_level,
        'recommendation': recommendation,
        'color': color,
        'description': description,
        'confidence': confidence,
        'thresholds_used': thresholds
    }
    
    return assessment

def create_prediction_summary(prediction_results: Dict[str, Any],
                           feature_data: Optional[pd.DataFrame] = None,
                           sample_idx: int = 0) -> Dict[str, Any]:
    """
    Create a comprehensive prediction summary.
    
    Args:
        prediction_results: Results from make_prediction
        feature_data: Original feature data
        sample_idx: Index of sample to summarize
        
    Returns:
        Dict: Comprehensive prediction summary
    """
    
    # Get prediction and probability for the sample
    prediction = prediction_results['predictions'][sample_idx]
    prob_default = prediction_results['probabilities']['default'][sample_idx]
    prob_no_default = prediction_results['probabilities']['no_default'][sample_idx]
    
    # Assess risk
    risk_assessment = assess_risk(prob_default)
    
    # Create summary
    summary = {
        'prediction': prediction,
        'probabilities': {
            'default': prob_default,
            'no_default': prob_no_default
        },
        'risk_assessment': risk_assessment,
        'model_used': prediction_results['model_type']
    }
    
    # Add feature information if available
    if feature_data is not None:
        sample_features = feature_data.iloc[sample_idx].to_dict()
        summary['features'] = sample_features
        
        # Add key risk factors
        risk_factors = {}
        
        # Credit score risk
        if 'credit_score' in sample_features:
            if sample_features['credit_score'] < 600:
                risk_factors['credit_score'] = 'poor'
            elif sample_features['credit_score'] < 700:
                risk_factors['credit_score'] = 'fair'
            else:
                risk_factors['credit_score'] = 'good'
        
        # Income risk
        if 'income' in sample_features:
            if sample_features['income'] < 30000:
                risk_factors['income'] = 'low'
            elif sample_features['income'] < 75000:
                risk_factors['income'] = 'medium'
            else:
                risk_factors['income'] = 'high'
        
        # Debt-to-income risk
        if 'debt_to_income' in sample_features:
            if sample_features['debt_to_income'] > 40:
                risk_factors['debt_to_income'] = 'high'
            elif sample_features['debt_to_income'] > 30:
                risk_factors['debt_to_income'] = 'medium'
            else:
                risk_factors['debt_to_income'] = 'low'

        # Interest rate risk
        if 'interest_rate' in sample_features:
            if sample_features['interest_rate'] > 15:
                risk_factors['interest_rate'] = 'high'
            elif sample_features['interest_rate'] > 10:
                risk_factors['interest_rate'] = 'medium'
            else:
                risk_factors['interest_rate'] = 'low'
        
        summary['risk_factors'] = risk_factors
    
    return summary

def display_prediction_results(summary: Dict[str, Any], 
                             show_features: bool = True) -> None:
    """
    Display prediction results in Streamlit.
    
    Args:
        summary: Prediction summary from create_prediction_summary
        show_features: Whether to show feature details
    """
    
    # Header with prediction result
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if summary['prediction'] == 1:
            st.error(":material/error: **DEFAULT PREDICTED**")
        else:
            st.success(":material/check_circle: **NO DEFAULT PREDICTED**")
    
    with col2:
        risk_level = summary['risk_assessment']['risk_level']
        color = summary['risk_assessment']['color']
        
        if color == 'red':
            st.error(f"**Risk Level: {risk_level.upper()}**")
        elif color == 'orange':
            st.warning(f"**Risk Level: {risk_level.upper()}**")
        elif color == 'yellow':
            st.info(f"**Risk Level: {risk_level.upper()}**")
        else:
            st.success(f"**Risk Level: {risk_level.upper()}**")
    
    with col3:
        prob_default = summary['probabilities']['default']
        st.metric("Default Probability", f"{prob_default:.1%}")
    
    # Detailed information
    st.subheader(":material/analytics: Prediction Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Probabilities:**")
        st.write(f"- Default: {summary['probabilities']['default']:.1%}")
        st.write(f"- No Default: {summary['probabilities']['no_default']:.1%}")
        
        st.write("**Recommendation:**")
        recommendation = summary['risk_assessment']['recommendation']
        if recommendation == 'APPROVE':
            st.success(f":material/check_circle: {recommendation.replace('_', ' ')}")
        elif recommendation == 'REJECT':
            st.error(f":material/cancel: {recommendation.replace('_', ' ')}")
        else:
            st.warning(f":material/warning: {recommendation.replace('_', ' ')}")
    
    with col2:
        st.write("**Confidence:**")
        confidence = summary['risk_assessment']['confidence']
        st.write(f"{confidence:.1%}")
        
        st.write("**Description:**")
        st.write(summary['risk_assessment']['description'])
    
    # Feature details
    if show_features and 'features' in summary:
        st.subheader(":material/search: Feature Details")
        
        features_df = pd.DataFrame([
            {'Feature': k, 'Value': v} for k, v in summary['features'].items()
        ])
        st.dataframe(features_df)
        
        # Risk factors
        if 'risk_factors' in summary:
            st.write("**Key Risk Factors:**")
            for factor, level in summary['risk_factors'].items():
                if level == 'poor' or level == 'high':
                    st.write(f":material/error: {factor}: {level}")
                elif level == 'fair' or level == 'medium':
                    st.write(f":material/warning: {factor}: {level}")
                else:
                    st.write(f":material/check_circle: {factor}: {level}")

def batch_predict(models: Dict[str, Any], X: np.ndarray,
                feature_names: List[str]) -> pd.DataFrame:
    """
    Make predictions with multiple models for comparison.
    
    Args:
        models: Dictionary of trained models
        X: Input features
        feature_names: Names of features
        
    Returns:
        pd.DataFrame: Predictions from all models
    """
    
    results = []
    
    for model_name, model in models.items():
        # Make prediction
        prediction_result = make_prediction(model, X)
        
        if prediction_result:
            # Create summary for each sample
            for i in range(len(X)):
                summary = create_prediction_summary(prediction_result, sample_idx=i)
                
                results.append({
                    'model': model_name,
                    'sample_idx': i,
                    'prediction': summary['prediction'],
                    'prob_default': summary['probabilities']['default'],
                    'risk_level': summary['risk_assessment']['risk_level'],
                    'recommendation': summary['risk_assessment']['recommendation']
                })
    
    return pd.DataFrame(results)

def save_prediction_results(results: Dict[str, Any], path: str) -> None:
    """
    Save prediction results to file.
    
    Args:
        results: Prediction results dictionary
        path: Path to save results
    """
    
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(results, path)
        st.success(f":material/save: Saved prediction results to {path}")
    except Exception as e:
        st.error(f":material/cancel: Failed to save prediction results: {str(e)}")

def load_prediction_results(path: str) -> Optional[Dict[str, Any]]:
    """
    Load prediction results from file.
    
    Args:
        path: Path to load results from
        
    Returns:
        Dict: Loaded prediction results or None if failed
    """
    
    try:
        results = joblib.load(path)
        st.success(f":material/folder_open: Loaded prediction results from {path}")
        return results
    except Exception as e:
        st.error(f":material/cancel: Failed to load prediction results: {str(e)}")
        return None

def create_prediction_api(model: Any, preprocessor: Any = None) -> callable:
    """
    Create a prediction API function for deployment.
    
    Args:
        model: Trained model
        preprocessor: Optional preprocessing pipeline
        
    Returns:
        callable: API function that takes raw data and returns predictions
    """
    
    def predict_api(raw_data: pd.DataFrame) -> Dict[str, Any]:
        """
        API function for making predictions.
        
        Args:
            raw_data: Raw input data as DataFrame
            
        Returns:
            Dict: Prediction results
        """
        
        try:
            # Preprocess data if preprocessor provided
            if preprocessor is not None:
                processed_data = preprocessor.transform(raw_data)
            else:
                processed_data = raw_data.values
            
            # Make prediction
            results = make_prediction(model, processed_data)
            
            # Add preprocessing info
            if preprocessor is not None:
                results['preprocessing_applied'] = True
            else:
                results['preprocessing_applied'] = False
            
            return results
            
        except Exception as e:
            return {
                'error': str(e),
                'predictions': None,
                'probabilities': None
            }
    
    return predict_api