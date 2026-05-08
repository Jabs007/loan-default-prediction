"""
Explainability utilities for loan default prediction using SHAP.

This module provides model interpretability through SHAP values, 
helping explain why models make specific predictions - crucial for 
financial applications where transparency is required.
"""

import pandas as pd
import numpy as np
import shap
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from typing import Dict, Any, List, Optional
import joblib
from pathlib import Path

def create_shap_explainer(model: Any, X_train: np.ndarray, 
                         feature_names: Optional[List[str]] = None) -> shap.Explainer:
    """
    Create a SHAP explainer for the trained model.
    
    Args:
        model: Trained model object
        X_train: Training data used to create background distribution
        feature_names: Names of features for interpretability
        
    Returns:
        shap.Explainer: SHAP explainer object
    """
    
    st.header(":material/psychology: Creating SHAP Explainer")
    
    try:
        # Determine explainer type based on model
        if hasattr(model, 'predict_proba'):
            # Tree-based models
            if hasattr(model, 'estimators_'):
                explainer = shap.TreeExplainer(model)
                st.success(":material/check_circle: Created TreeExplainer for tree-based model")
            else:
                # General models with predict_proba
                explainer = shap.KernelExplainer(model.predict_proba, X_train[:100])  # Use subset for speed
                st.success(":material/check_circle: Created KernelExplainer for general model")
        else:
            # Models without predict_proba
            explainer = shap.KernelExplainer(model.predict, X_train[:100])
            st.success(":material/check_circle: Created KernelExplainer")
        
        return explainer
        
    except Exception as e:
        st.error(f":material/cancel: Failed to create SHAP explainer: {str(e)}")
        return None

def calculate_shap_values(explainer: shap.Explainer, X_test: np.ndarray,
                         sample_size: Optional[int] = None) -> np.ndarray:
    """
    Calculate SHAP values for test data.
    
    Args:
        explainer: SHAP explainer object
        X_test: Test data to explain
        sample_size: Number of samples to explain (for performance)
        
    Returns:
        np.ndarray: SHAP values
    """
    
    if sample_size and len(X_test) > sample_size:
        # Sample data for performance
        indices = np.random.choice(len(X_test), sample_size, replace=False)
        X_sample = X_test[indices]
        st.info(f"Calculating SHAP values for {sample_size} samples")
    else:
        X_sample = X_test
        st.info(f"Calculating SHAP values for all {len(X_test)} samples")
    
    try:
        # Calculate SHAP values
        if hasattr(explainer, 'shap_values'):
            shap_values = explainer.shap_values(X_sample)
            # Handle different output formats
            if isinstance(shap_values, list):
                # For binary classification, take the positive class
                shap_values = shap_values[1]
        else:
            # For KernelExplainer
            shap_values = explainer.shap_values(X_sample)
        
        st.success(f":material/check_circle: Calculated SHAP values for {len(X_sample)} samples")
        return shap_values, X_sample
        
    except Exception as e:
        st.error(f":material/cancel: Failed to calculate SHAP values: {str(e)}")
        return None, X_sample

def plot_global_feature_importance(shap_values: np.ndarray, 
                                 feature_names: List[str], 
                                 model_name: str = "Model") -> go.Figure:
    """
    Create global feature importance plot using SHAP values.
    
    Args:
        shap_values: SHAP values array
        feature_names: Names of features
        model_name: Name of the model
        
    Returns:
        go.Figure: Plotly figure object
    """
    
    # Calculate mean absolute SHAP values for global importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=importance_df['importance'],
        y=importance_df['feature'],
        orientation='h',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title=f'Global Feature Importance - {model_name}',
        xaxis_title='Mean Absolute SHAP Value',
        yaxis_title='Feature',
        width=700,
        height=400,
        showlegend=False
    )
    
    return fig

def plot_local_prediction_explanation(shap_values: np.ndarray, 
                                    X_sample: np.ndarray,
                                    feature_names: List[str],
                                    sample_idx: int = 0,
                                    model_name: str = "Model") -> go.Figure:
    """
    Create local explanation plot for a single prediction.
    
    Args:
        shap_values: SHAP values array
        X_sample: Sample data
        feature_names: Names of features
        sample_idx: Index of sample to explain
        model_name: Name of the model
        
    Returns:
        go.Figure: Plotly figure object
    """
    
    # Get SHAP values for the selected sample
    sample_shap = shap_values[sample_idx]
    sample_features = X_sample[sample_idx]
    
    # Create DataFrame for plotting
    explanation_df = pd.DataFrame({
        'feature': feature_names,
        'value': sample_features,
        'shap_value': sample_shap
    }).sort_values('shap_value')
    
    # Separate positive and negative contributions
    positive_mask = explanation_df['shap_value'] > 0
    
    fig = go.Figure()
    
    # Add negative contributions (towards no default)
    fig.add_trace(go.Bar(
        x=explanation_df[~positive_mask]['shap_value'],
        y=explanation_df[~positive_mask]['feature'],
        orientation='h',
        name='Towards No Default',
        marker_color='lightgreen',
        text=[f"{row['feature']}: {row['value']:.2f}" 
              for _, row in explanation_df[~positive_mask].iterrows()],
        textposition='auto'
    ))
    
    # Add positive contributions (towards default)
    fig.add_trace(go.Bar(
        x=explanation_df[positive_mask]['shap_value'],
        y=explanation_df[positive_mask]['feature'],
        orientation='h',
        name='Towards Default',
        marker_color='lightcoral',
        text=[f"{row['feature']}: {row['value']:.2f}" 
              for _, row in explanation_df[positive_mask].iterrows()],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f'Local Explanation - Sample {sample_idx} - {model_name}',
        xaxis_title='SHAP Value (Impact on Prediction)',
        yaxis_title='Feature',
        width=700,
        height=400,
        showlegend=True,
        barmode='relative'
    )
    
    return fig

def create_interactive_explanation_dashboard(shap_values: np.ndarray,
                                          X_sample: np.ndarray,
                                          feature_names: List[str],
                                          model_name: str = "Model") -> None:
    """
    Create an interactive Streamlit dashboard for exploring SHAP explanations.
    
    Args:
        shap_values: SHAP values array
        X_sample: Sample data
        feature_names: Names of features
        model_name: Name of the model
    """
    
    st.header(f":material/search: SHAP Explainability Dashboard - {model_name}")
    
    # Global Feature Importance
    st.subheader("🌍 Global Feature Importance")
    st.markdown("This shows which features are most important for the model overall.")
    
    global_fig = plot_global_feature_importance(shap_values, feature_names, model_name)
    st.plotly_chart(global_fig, use_container_width=True)
    
    # Summary plot
    st.subheader(":material/analytics: SHAP Summary Plot")
    st.markdown("This shows the distribution of SHAP values for each feature.")
    
    try:
        # Create summary plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.warning(f"Could not create summary plot: {str(e)}")
    
    # Local explanations
    st.subheader("🔬 Individual Prediction Explanations")
    st.markdown("Explore how the model makes predictions for individual samples.")
    
    # Sample selector
    sample_idx = st.selectbox(
        "Select sample to explain:",
        range(len(X_sample)),
        format_func=lambda x: f"Sample {x}"
    )
    
    # Show prediction details
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Feature Values:**")
        feature_df = pd.DataFrame({
            'Feature': feature_names,
            'Value': X_sample[sample_idx]
        })
        st.dataframe(feature_df)
    
    with col2:
        st.write("**SHAP Values:**")
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': shap_values[sample_idx]
        }).sort_values('SHAP Value', ascending=False)
        st.dataframe(shap_df)
    
    # Local explanation plot
    local_fig = plot_local_prediction_explanation(
        shap_values, X_sample, feature_names, sample_idx, model_name
    )
    st.plotly_chart(local_fig, use_container_width=True)
    
    # Feature interaction analysis
    st.subheader(":material/link: Feature Interactions")
    st.markdown("Explore how features interact with each other.")
    
    # Select two features for interaction plot
    feature_1 = st.selectbox("Select first feature:", feature_names)
    feature_2 = st.selectbox("Select second feature:", 
                           [f for f in feature_names if f != feature_1])
    
    try:
        # Create interaction plot
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.dependence_plot(
            feature_1, 
            shap_values, 
            X_sample,
            feature_names=feature_names,
            interaction_index=feature_2,
            show=False
        )
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.warning(f"Could not create interaction plot: {str(e)}")

def explain_prediction(model: Any, X_instance: np.ndarray, feature_names: List[str],
                      explainer: Optional[shap.Explainer] = None) -> Dict[str, Any]:
    """
    Explain a single prediction using SHAP.
    
    Args:
        model: Trained model
        X_instance: Single instance to explain
        feature_names: Names of features
        explainer: Pre-computed SHAP explainer (optional)
        
    Returns:
        Dict: Explanation including SHAP values and feature contributions
    """
    
    # Create explainer if not provided
    if explainer is None:
        # Use a simple background dataset
        background_data = np.random.randn(100, X_instance.shape[0])
        explainer = shap.KernelExplainer(model.predict_proba, background_data)
    
    # Calculate SHAP values for this instance
    shap_values = explainer.shap_values(X_instance.reshape(1, -1))
    
    # Handle different output formats
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class for binary classification
    
    # Create explanation dictionary
    explanation = {
        'shap_values': shap_values[0],
        'feature_names': feature_names,
        'feature_values': X_instance,
        'base_value': explainer.expected_value if hasattr(explainer, 'expected_value') else None
    }
    
    # Calculate feature contributions
    contributions = []
    for i, (name, value, shap_val) in enumerate(zip(feature_names, X_instance, shap_values[0])):
        contributions.append({
            'feature': name,
            'value': value,
            'shap_value': shap_val,
            'contribution': 'positive' if shap_val > 0 else 'negative'
        })
    
    # Sort by absolute SHAP value
    contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
    explanation['contributions'] = contributions
    
    return explanation

def save_explainer(explainer: shap.Explainer, path: str) -> None:
    """
    Save SHAP explainer to disk.
    
    Args:
        explainer: SHAP explainer to save
        path: Path to save the explainer
    """
    
    try:
        joblib.dump(explainer, path)
        st.success(f":material/save: Saved SHAP explainer to {path}")
    except Exception as e:
        st.error(f":material/cancel: Failed to save explainer: {str(e)}")

def load_explainer(path: str) -> Optional[shap.Explainer]:
    """
    Load SHAP explainer from disk.
    
    Args:
        path: Path to the saved explainer
        
    Returns:
        shap.Explainer: Loaded explainer or None if failed
    """
    
    try:
        explainer = joblib.load(path)
        st.success(f":material/folder_open: Loaded SHAP explainer from {path}")
        return explainer
    except Exception as e:
        st.error(f":material/cancel: Failed to load explainer: {str(e)}")
        return None