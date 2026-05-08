"""
Model evaluation utilities for loan default prediction.

This module provides comprehensive evaluation metrics and visualizations
for assessing model performance in the context of credit risk.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, average_precision_score
)
import streamlit as st
from typing import Dict, Any, Tuple, List
import joblib

def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_proba: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    In credit risk, different metrics have different importance:
    - Recall (Sensitivity): Critical - we want to catch defaulters
    - Precision: Important - avoid rejecting good customers
    - F1-Score: Balanced view of precision and recall
    - AUC-ROC: Overall discriminative ability
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        
    Returns:
        Dict: Dictionary of metrics
    """
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'average_precision': average_precision_score(y_true, y_proba)
    }
    
    return metrics

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         model_name: str = "Model") -> go.Figure:
    """
    Create an interactive confusion matrix plot.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        model_name: Name of the model for the title
        
    Returns:
        go.Figure: Plotly figure object
    """
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Create annotations
    annotations = []
    for i in range(2):
        for j in range(2):
            annotations.append(
                dict(
                    x=j, y=i,
                    text=str(cm[i, j]),
                    showarrow=False,
                    font=dict(size=20, color="white" if cm[i, j] > cm.max()/2 else "black")
                )
            )
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted: No Default', 'Predicted: Default'],
        y=['Actual: No Default', 'Actual: Default'],
        colorscale='Blues',
        showscale=True,
        colorbar=dict(title="Count")
    ))
    
    fig.update_layout(
        title=f'Confusion Matrix - {model_name}',
        xaxis_title='Predicted Label',
        yaxis_title='Actual Label',
        width=500,
        height=400,
        annotations=annotations
    )
    
    return fig

def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, 
                  model_name: str = "Model") -> go.Figure:
    """
    Create an interactive ROC curve plot.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        model_name: Name of the model for the title
        
    Returns:
        go.Figure: Plotly figure object
    """
    
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    
    fig = go.Figure()
    
    # Add ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'{model_name} (AUC = {auc_score:.3f})',
        line=dict(color='blue', width=2)
    ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=1, dash='dash')
    ))
    
    fig.update_layout(
        title=f'ROC Curve - {model_name}',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate (Recall)',
        width=600,
        height=500,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05])
    )
    
    return fig

def plot_precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray,
                               model_name: str = "Model") -> go.Figure:
    """
    Create an interactive Precision-Recall curve plot.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        model_name: Name of the model for the title
        
    Returns:
        go.Figure: Plotly figure object
    """
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    
    fig = go.Figure()
    
    # Add PR curve
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name=f'{model_name} (AP = {avg_precision:.3f})',
        line=dict(color='red', width=2)
    ))
    
    # Add baseline (random classifier performance)
    baseline = np.sum(y_true) / len(y_true)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[baseline, baseline],
        mode='lines',
        name=f'Baseline (AP = {baseline:.3f})',
        line=dict(color='gray', width=1, dash='dash')
    ))
    
    fig.update_layout(
        title=f'Precision-Recall Curve - {model_name}',
        xaxis_title='Recall (True Positive Rate)',
        yaxis_title='Precision',
        width=600,
        height=500,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05])
    )
    
    return fig

def plot_probability_distribution(y_true: np.ndarray, y_proba: np.ndarray,
                               model_name: str = "Model") -> go.Figure:
    """
    Create probability distribution plots for both classes.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        model_name: Name of the model for the title
        
    Returns:
        go.Figure: Plotly figure object
    """
    
    # Separate probabilities by class
    prob_default = y_proba[y_true == 1]
    prob_no_default = y_proba[y_true == 0]
    
    fig = go.Figure()
    
    # Add histogram for defaults
    fig.add_trace(go.Histogram(
        x=prob_default,
        name='Actual Defaults',
        opacity=0.7,
        marker_color='red',
        nbinsx=30
    ))
    
    # Add histogram for non-defaults
    fig.add_trace(go.Histogram(
        x=prob_no_default,
        name='Actual Non-Defaults',
        opacity=0.7,
        marker_color='blue',
        nbinsx=30
    ))
    
    fig.update_layout(
        title=f'Probability Distribution - {model_name}',
        xaxis_title='Predicted Probability of Default',
        yaxis_title='Count',
        width=600,
        height=400,
        barmode='overlay'
    )
    
    return fig

def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray,
                  model_name: str = "Model", threshold: float = 0.5) -> Dict[str, Any]:
    """
    Comprehensive model evaluation with metrics and visualizations.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        model_name: Name of the model
        threshold: Classification threshold
        
    Returns:
        Dict: Evaluation results including metrics and plots
    """
    
    # Get predictions and probabilities
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate metrics
    metrics = calculate_classification_metrics(y_test, y_pred, y_proba)
    
    # Create visualizations
    results = {
        'metrics': metrics,
        'confusion_matrix': plot_confusion_matrix(y_test, y_pred, model_name),
        'roc_curve': plot_roc_curve(y_test, y_proba, model_name),
        'precision_recall_curve': plot_precision_recall_curve(y_test, y_proba, model_name),
        'probability_distribution': plot_probability_distribution(y_test, y_proba, model_name)
    }
    
    return results

def display_metrics_summary(metrics: Dict[str, float], model_name: str = "Model"):
    """
    Display metrics in a formatted Streamlit layout.
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
    """
    
    st.subheader(f":material/analytics: {model_name} Performance Metrics")
    
    # Create columns for metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    
    with col2:
        st.metric("Precision", f"{metrics['precision']:.3f}")
    
    with col3:
        st.metric("Recall", f"{metrics['recall']:.3f}")
    
    with col4:
        st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
    
    with col5:
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    
    # Add explanations for credit risk context
    with st.expander("ℹ️ Understanding These Metrics for Credit Risk"):
        st.markdown("""
        **Accuracy**: Overall correctness of predictions. However, can be misleading if classes are imbalanced.
        
        **Precision**: Of all loans predicted as defaults, how many actually defaulted? 
        High precision means fewer false alarms (good customers incorrectly flagged).
        
        **Recall**: Of all actual defaults, how many did we correctly identify?
        **Critical for credit risk** - missing a defaulter is very costly!
        
        **F1-Score**: Balanced measure of precision and recall.
        
        **ROC-AUC**: Overall ability to distinguish between defaulters and non-defaulters.
        Higher values indicate better discriminative power.
        """)

def compare_models(models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray,
                  thresholds: Dict[str, float] = None) -> pd.DataFrame:
    """
    Compare multiple models across various metrics.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test targets
        thresholds: Dictionary of classification thresholds per model
        
    Returns:
        pd.DataFrame: Comparison table
    """
    
    if thresholds is None:
        thresholds = {name: 0.5 for name in models.keys()}
    
    comparison_data = []
    
    for model_name, model in models.items():
        # Get predictions
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= thresholds[model_name]).astype(int)
        
        # Calculate metrics
        metrics = calculate_classification_metrics(y_test, y_pred, y_proba)
        
        # Add model name and threshold
        metrics['model'] = model_name
        metrics['threshold'] = thresholds[model_name]
        
        comparison_data.append(metrics)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.set_index('model')
    
    return comparison_df

def plot_model_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """
    Create a radar chart comparing models across multiple metrics.
    
    Args:
        comparison_df: DataFrame with model comparison metrics
        
    Returns:
        go.Figure: Plotly figure object
    """
    
    # Select metrics for radar chart
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    fig = go.Figure()
    
    for model in comparison_df.index:
        values = comparison_df.loc[model, metrics].values.tolist()
        values += values[:1]  # Complete the circle
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            fill='toself',
            name=model
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Model Performance Comparison",
        width=600,
        height=500
    )
    
    return fig

def generate_evaluation_report(model: Any, X_test: np.ndarray, y_test: np.ndarray,
                             model_name: str = "Model", save_path: str = None) -> str:
    """
    Generate a comprehensive text report of model evaluation.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        model_name: Name of the model
        save_path: Path to save the report
        
    Returns:
        str: Evaluation report text
    """
    
    # Get predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    
    # Calculate metrics
    metrics = calculate_classification_metrics(y_test, y_pred, y_proba)
    
    # Generate classification report
    class_report = classification_report(y_test, y_pred, target_names=['No Default', 'Default'])
    
    # Create comprehensive report
    report = f"""
# Model Evaluation Report - {model_name}

## Summary Statistics
- **Test Set Size**: {len(y_test)} samples
- **Default Rate**: {y_test.mean():.1%}
- **Predicted Default Rate**: {y_pred.mean():.1%}

## Performance Metrics
- **Accuracy**: {metrics['accuracy']:.3f}
- **Precision**: {metrics['precision']:.3f}
- **Recall**: {metrics['recall']:.3f} 
- **F1-Score**: {metrics['f1_score']:.3f}
- **ROC-AUC**: {metrics['roc_auc']:.3f}
- **Average Precision**: {metrics['average_precision']:.3f}

## Detailed Classification Report
{class_report}

## Business Impact Analysis
- **True Positives**: {(y_test & y_pred).sum()} - Correctly identified defaulters
- **False Positives**: {((~y_test.astype(bool)) & y_pred).sum()} - Good customers rejected
- **True Negatives**: {((~y_test.astype(bool)) & (~y_pred.astype(bool))).sum()} - Correctly approved
- **False Negatives**: {(y_test & (~y_pred)).sum()} - Defaulters missed (COSTLY!)

## Recommendations
Based on the model performance:
1. The model shows {'good' if metrics['roc_auc'] > 0.8 else 'moderate' if metrics['roc_auc'] > 0.7 else 'poor'} discriminative ability (AUC = {metrics['roc_auc']:.3f})
2. {'Good' if metrics['recall'] > 0.8 else 'Moderate' if metrics['recall'] > 0.6 else 'Poor'} recall rate for identifying defaulters ({metrics['recall']:.1%})
3. {'Low' if metrics['precision'] > 0.8 else 'Moderate' if metrics['precision'] > 0.6 else 'High'} false positive rate ({(1-metrics['precision']):.1%})

## Risk Assessment
- The model {'effectively' if metrics['f1_score'] > 0.75 else 'moderately' if metrics['f1_score'] > 0.6 else 'poorly'} balances precision and recall
- {'Consider' if metrics['recall'] < 0.8 else 'Good'} recall performance for credit risk applications
- {'Monitor' if metrics['precision'] < 0.7 else 'Acceptable'} precision to minimize false alarms
"""
    
    # Save report if path provided
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        st.success(f"📄 Evaluation report saved to {save_path}")
    
    return report