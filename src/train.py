"""
Model training utilities for loan default prediction.

This module handles training multiple machine learning models with 
hyperparameter tuning and cross-validation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import joblib
from typing import Dict, Any, Tuple, Optional
import streamlit as st
from pathlib import Path

def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
               random_state: int = 42, stratify: bool = True) -> Tuple:
    """
    Split data into training and testing sets.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        stratify: Whether to maintain class distribution
        
    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    st.success(f"Data split complete: {len(X_train)} training, {len(X_test)} testing samples")
    st.info(f"Training set default rate: {y_train.mean():.1%}")
    st.info(f"Test set default rate: {y_test.mean():.1%}")
    
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray,
                            hyperparameter_tuning: bool = True) -> LogisticRegression:
    """
    Train a Logistic Regression model.
    
    Logistic Regression is a good baseline model for binary classification
    and provides interpretable coefficients.
    
    Args:
        X_train: Training features
        y_train: Training targets
        hyperparameter_tuning: Whether to perform hyperparameter tuning
        
    Returns:
        LogisticRegression: Trained model
    """
    
    st.header(":material/my_location: Training Logistic Regression")
    
    if hyperparameter_tuning:
        # Define parameter grid
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],
            'class_weight': [None, 'balanced']
        }
        
        # Perform grid search with cross-validation
        st.info("Performing hyperparameter tuning...")
        grid_search = GridSearchCV(
            LogisticRegression(random_state=42, max_iter=1000),
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        with st.spinner('Tuning hyperparameters...'):
            grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        st.success(f"Best parameters: {grid_search.best_params_}")
        st.info(f"Best cross-validation ROC-AUC: {grid_search.best_score_:.3f}")
        
    else:
        # Use default parameters with balanced class weights
        best_model = LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        )
        
        with st.spinner('Training model...'):
            best_model.fit(X_train, y_train)
    
    return best_model

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                       hyperparameter_tuning: bool = True) -> RandomForestClassifier:
    """
    Train a Random Forest model.
    
    Random Forest is an ensemble method that handles non-linear relationships
    and provides feature importance rankings.
    
    Args:
        X_train: Training features
        y_train: Training targets
        hyperparameter_tuning: Whether to perform hyperparameter tuning
        
    Returns:
        RandomForestClassifier: Trained model
    """
    
    st.header("🌲 Training Random Forest")
    
    if hyperparameter_tuning:
        # Define parameter grid (reduced for faster training)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': [None, 'balanced']
        }
        
        # Perform grid search with cross-validation
        st.info("Performing hyperparameter tuning...")
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            param_grid,
            cv=3,  # Reduced CV for faster training
            scoring='roc_auc',
            n_jobs=-1
        )
        
        with st.spinner('Tuning hyperparameters...'):
            grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        st.success(f"Best parameters: {grid_search.best_params_}")
        st.info(f"Best cross-validation ROC-AUC: {grid_search.best_score_:.3f}")
        
    else:
        # Use good default parameters
        best_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        with st.spinner('Training model...'):
            best_model.fit(X_train, y_train)
    
    return best_model

def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                 hyperparameter_tuning: bool = True) -> xgb.XGBClassifier:
    """
    Train an XGBoost model.
    
    XGBoost is a state-of-the-art gradient boosting method that often
    provides the best performance for tabular data.
    
    Args:
        X_train: Training features
        y_train: Training targets
        hyperparameter_tuning: Whether to perform hyperparameter tuning
        
    Returns:
        xgb.XGBClassifier: Trained model
    """
    
    st.header(":material/rocket_launch: Training XGBoost")
    
    # Calculate scale_pos_weight for imbalanced data
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    st.info(f"Using scale_pos_weight: {scale_pos_weight:.2f}")
    
    if hyperparameter_tuning:
        # Define parameter grid (reduced for faster training)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        }
        
        # Perform grid search with cross-validation
        st.info("Performing hyperparameter tuning...")
        grid_search = GridSearchCV(
            xgb.XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                scale_pos_weight=scale_pos_weight
            ),
            param_grid,
            cv=3,  # Reduced CV for faster training
            scoring='roc_auc',
            n_jobs=-1
        )
        
        with st.spinner('Tuning hyperparameters...'):
            grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        st.success(f"Best parameters: {grid_search.best_params_}")
        st.info(f"Best cross-validation ROC-AUC: {grid_search.best_score_:.3f}")
        
    else:
        # Use good default parameters
        best_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight
        )
        
        with st.spinner('Training model...'):
            best_model.fit(X_train, y_train)
    
    return best_model

def train_all_models(X_train: np.ndarray, y_train: np.ndarray,
                    models_to_train: list = ['logistic_regression', 'random_forest', 'xgboost'],
                    hyperparameter_tuning: bool = True) -> Dict[str, Any]:
    """
    Train multiple models and return them in a dictionary.
    
    Args:
        X_train: Training features
        y_train: Training targets
        models_to_train: List of model names to train
        hyperparameter_tuning: Whether to perform hyperparameter tuning
        
    Returns:
        Dict: Dictionary of trained models
    """
    
    trained_models = {}
    
    model_training_functions = {
        'logistic_regression': train_logistic_regression,
        'random_forest': train_random_forest,
        'xgboost': train_xgboost
    }
    
    for model_name in models_to_train:
        if model_name in model_training_functions:
            try:
                model = model_training_functions[model_name](X_train, y_train, hyperparameter_tuning)
                trained_models[model_name] = model
                st.success(f":material/check_circle: Successfully trained {model_name}")
            except Exception as e:
                st.error(f":material/cancel: Failed to train {model_name}: {str(e)}")
        else:
            st.warning(f":material/warning: Unknown model: {model_name}")
    
    return trained_models

def train_multiple_models(X: pd.DataFrame, y: pd.Series, preprocessor: Any, 
                         models_to_train: list = ["Logistic Regression", "Random Forest", "XGBoost"],
                         hyperparameter_tuning: bool = False) -> Tuple:
    """
    Train multiple models with preprocessing.
    
    Args:
        X: Input features
        y: Target variable
        preprocessor: Scikit-learn preprocessing pipeline or ColumnTransformer
        models_to_train: List of model names to train
        
    Returns:
        Tuple: (models_dict, X_train_processed, X_test_processed, y_train, y_test)
    """
    st.info("Applying preprocessing and splitting data...")
    
    # Map model names to internal names
    model_name_mapping = {
        "Logistic Regression": "logistic_regression",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost"
    }
    
    mapped_models = [model_name_mapping.get(m, m.lower().replace(" ", "_")) for m in models_to_train]
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Apply preprocessing
    with st.spinner("Preprocessing features..."):
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Try to maintain DataFrame format if possible (for SHAP)
        try:
            if hasattr(preprocessor, 'get_feature_names_out'):
                feature_names = preprocessor.get_feature_names_out()
                X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names)
                X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names)
        except Exception as e:
            st.warning(f"Could not extract feature names: {e}")
            X_train_processed = pd.DataFrame(X_train_processed)
            X_test_processed = pd.DataFrame(X_test_processed)
            
    # Train models
    models_dict = train_all_models(X_train_processed, y_train, mapped_models, hyperparameter_tuning=hyperparameter_tuning)
    
    # Map keys back to original names
    reverse_mapping = {v: k for k, v in model_name_mapping.items()}
    final_models = {reverse_mapping.get(k, k): v for k, v in models_dict.items()}
    
    # Reset index for y to match X's indices
    if isinstance(y_train, pd.Series):
        y_train = y_train.reset_index(drop=True)
    if isinstance(y_test, pd.Series):
        y_test = y_test.reset_index(drop=True)
        
    return final_models, X_train_processed, X_test_processed, y_train, y_test

def save_model(model: Any, model_path: str, model_name: str = "model") -> None:
    """
    Save trained model to disk.
    
    Args:
        model: Trained model object
        model_path: Path to save the model
        model_name: Name of the model for logging
    """
    
    try:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        st.success(f":material/save: Saved {model_name} model to {model_path}")
    except Exception as e:
        st.error(f":material/cancel: Failed to save {model_name}: {str(e)}")

def load_model(model_path: str) -> Any:
    """
    Load trained model from disk.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded model object
    """
    
    try:
        model = joblib.load(model_path)
        st.success(f":material/folder_open: Loaded model from {model_path}")
        return model
    except Exception as e:
        st.error(f":material/cancel: Failed to load model from {model_path}: {str(e)}")
        return None

def get_model_info(model: Any) -> Dict[str, Any]:
    """
    Get basic information about a trained model.
    
    Args:
        model: Trained model object
        
    Returns:
        Dict: Model information
    """
    
    model_info = {
        'type': type(model).__name__,
        'parameters': model.get_params() if hasattr(model, 'get_params') else {},
        'n_features': model.n_features_in_ if hasattr(model, 'n_features_in_') else 'unknown'
    }
    
    # Add model-specific information
    if hasattr(model, 'feature_importances_'):
        model_info['feature_importances'] = model.feature_importances_
    
    if hasattr(model, 'coef_'):
        model_info['coefficients'] = model.coef_
    
    return model_info