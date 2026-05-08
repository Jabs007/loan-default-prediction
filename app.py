"""
Loan Default Prediction System - Main Streamlit Application

A comprehensive machine learning application for predicting loan defaults with:
- Interactive data exploration
- Model performance comparison
- Real-time predictions
- SHAP explainability
- Professional dashboard interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import our modules
from data_utils import generate_sample_data, validate_loan_data, get_data_summary
from preprocessing import create_preprocessing_pipeline, handle_missing_values, detect_outliers
from feature_engineering import engineer_all_features
from train import train_multiple_models, save_model, load_model
from evaluate import calculate_classification_metrics, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from explainability import create_shap_explainer, plot_global_feature_importance, plot_local_prediction_explanation
from predict import make_prediction, assess_risk, create_prediction_summary, display_prediction_results

# Page configuration
st.set_page_config(
    page_title="🏦 Loan Default Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-low {
        color: #2ca02c;
        font-weight: bold;
    }
    .risk-medium {
        color: #ff7f0e;
        font-weight: bold;
    }
    .risk-high {
        color: #d62728;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=80)
    st.title("🏦 Loan Default Prediction")
    
    st.markdown("---")
    
    # Data source selection
    st.subheader("📊 Data Source")
    data_source = st.radio(
        "Choose data source:",
        ["Sample Data", "Upload CSV", "Use Saved Data"]
    )
    
    if data_source == "Sample Data":
        n_samples = st.slider("Number of samples:", 1000, 10000, 5000, step=500)
        if st.button("Generate Data"):
            with st.spinner("Generating sample data..."):
                df = generate_sample_data(n_samples=n_samples, random_state=42)
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.success(f"✅ Generated {len(df)} samples")
    
    elif data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                validation_results = validate_loan_data(df)
                if validation_results['is_valid']:
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    st.success("✅ Data uploaded successfully")
                else:
                    st.error("❌ Data validation failed")
                    if validation_results['errors']:
                        for error in validation_results['errors']:
                            st.write(f"- {error}")
            except Exception as e:
                st.error(f"❌ Failed to load file: {str(e)}")
    
    elif data_source == "Use Saved Data":
        data_path = st.text_input("Data file path:", "data/processed/loan_data_engineered.csv")
        if st.button("Load Data"):
            try:
                df = pd.read_csv(data_path)
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.success("✅ Data loaded successfully")
            except Exception as e:
                st.error(f"❌ Failed to load data: {str(e)}")
    
    st.markdown("---")
    
    # Model training section
    if st.session_state.data_loaded:
        st.subheader("🤖 Model Training")
        
        models_to_train = st.multiselect(
            "Select models to train:",
            ["Logistic Regression", "Random Forest", "XGBoost"],
            default=["Logistic Regression", "Random Forest", "XGBoost"]
        )
        
        if st.button("Train Models"):
            with st.spinner("Training models... This may take a few minutes."):
                try:
                    # Prepare data
                    df = st.session_state.df
                    
                    # Feature engineering
                    df_engineered = engineer_all_features(df)
                    
                    # Split features and target
                    X = df_engineered.drop('default', axis=1)
                    y = df_engineered['default']
                    
                    # Create preprocessing pipeline
                    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
                    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
                    
                    preprocessor = create_preprocessing_pipeline(
                        numerical_features, categorical_features
                    )
                    
                    # Train models
                    models, X_train, X_test, y_train, y_test = train_multiple_models(
                        X, y, preprocessor, models_to_train
                    )
                    
                    # Store in session state
                    st.session_state.models = models
                    st.session_state.preprocessor = preprocessor
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.models_trained = True
                    
                    st.success("✅ Models trained successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
        
        # Model loading section
        st.markdown("---")
        st.subheader("📁 Load Saved Models")
        
        if st.button("Load Pre-trained Models"):
            try:
                # Try to load saved models
                models = {}
                model_files = {
                    "Logistic Regression": "models/logistic_regression.pkl",
                    "Random Forest": "models/random_forest.pkl",
                    "XGBoost": "models/xgboost.pkl"
                }
                
                for model_name, file_path in model_files.items():
                    if os.path.exists(file_path):
                        models[model_name] = load_model(file_path)
                
                if models:
                    st.session_state.models = models
                    st.session_state.models_trained = True
                    st.success("✅ Models loaded successfully!")
                else:
                    st.warning("⚠️ No saved models found")
                    
            except Exception as e:
                st.error(f"❌ Failed to load models: {str(e)}")

# Main content area
def main():
    st.markdown('<div class="main-header">🏦 Loan Default Prediction System</div>', unsafe_allow_html=True)
    
    # Navigation tabs
    tabs = st.tabs(["📊 EDA & Data Overview", "🤖 Model Performance", "🔮 Predictions", "📚 Documentation"])
    
    # Tab 1: EDA & Data Overview
    with tabs[0]:
        st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.info("📋 Please load data first using the sidebar")
        else:
            df = st.session_state.df
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Samples", f"{len(df):,}")
            
            with col2:
                st.metric("Features", len(df.columns))
            
            with col3:
                default_rate = df['default'].mean()
                st.metric("Default Rate", f"{default_rate:.1%}")
            
            with col4:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            # Data preview
            st.subheader("📝 Data Preview")
            st.dataframe(df.head(10))
            
            # Statistical summary
            st.subheader("📈 Statistical Summary")
            st.dataframe(df.describe())
            
            # Default distribution
            st.subheader("🎯 Target Variable Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    values=df['default'].value_counts(),
                    names=['No Default', 'Default'],
                    title=f'Default Distribution (Rate: {default_rate:.1%})',
                    color_discrete_map={'No Default': 'lightgreen', 'Default': 'lightcoral'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Feature distributions
                numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                numerical_cols.remove('default')
                
                if numerical_cols:
                    selected_feature = st.selectbox("Select feature to visualize:", numerical_cols)
                    fig = px.histogram(df, x=selected_feature, color='default', 
                                     title=f'{selected_feature} Distribution by Default Status')
                    st.plotly_chart(fig, use_container_width=True)
            
            # Correlation analysis
            st.subheader("🔗 Correlation Analysis")
            if len(numerical_cols) > 1:
                correlation_matrix = df[numerical_cols + ['default']].corr()
                fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto",
                              title='Feature Correlation Matrix',
                              color_continuous_scale='RdBu_r', zmid=0)
                st.plotly_chart(fig, use_container_width=True)
            
            # Default rate by features
            st.subheader("📊 Default Rate Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Credit score analysis
                if 'credit_score' in df.columns:
                    df['credit_score_bin'] = pd.cut(df['credit_score'], bins=5, 
                                                  labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
                    credit_analysis = df.groupby('credit_score_bin')['default'].agg(['count', 'mean']).reset_index()
                    
                    fig = px.bar(credit_analysis, x='credit_score_bin', y='mean',
                               text='mean', title='Default Rate by Credit Score Range',
                               labels={'mean': 'Default Rate', 'credit_score_bin': 'Credit Score Range'},
                               hover_data=['count'])
                    fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Income analysis
                if 'income' in df.columns:
                    df['income_bin'] = pd.cut(df['income'], bins=5,
                                              labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
                    income_analysis = df.groupby('income_bin')['default'].agg(['count', 'mean']).reset_index()
                    
                    fig = px.bar(income_analysis, x='income_bin', y='mean',
                               text='mean', title='Default Rate by Income Range',
                               labels={'mean': 'Default Rate', 'income_bin': 'Income Range'},
                               hover_data=['count'])
                    fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Model Performance
    with tabs[1]:
        st.markdown('<div class="section-header">🤖 Model Performance</div>', unsafe_allow_html=True)
        
        if not st.session_state.models_trained:
            st.info("🤖 Please train models first using the sidebar")
        else:
            models = st.session_state.models
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            
            # Model selection
            selected_models = st.multiselect(
                "Select models to evaluate:",
                list(models.keys()),
                default=list(models.keys())
            )
            
            if selected_models:
                # Performance metrics
                st.subheader("📊 Performance Metrics")
                
                metrics_df = []
                for model_name in selected_models:
                    model = models[model_name]
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    metrics = calculate_classification_metrics(y_test, y_pred, y_proba)
                    metrics['Model'] = model_name
                    metrics_df.append(metrics)
                
                metrics_df = pd.DataFrame(metrics_df)
                st.dataframe(metrics_df)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # ROC Curves
                    fig = go.Figure()
                    for model_name in selected_models:
                        model = models[model_name]
                        if hasattr(model, 'predict_proba'):
                            y_proba = model.predict_proba(X_test)[:, 1]
                            from sklearn.metrics import roc_curve, roc_auc_score
                            fpr, tpr, _ = roc_curve(y_test, y_proba)
                            auc = roc_auc_score(y_test, y_proba)
                            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{model_name} (AUC={auc:.3f})'))
                    
                    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random', line=dict(dash='dash')))
                    fig.update_layout(title='ROC Curves', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Confusion Matrix for selected model
                    selected_model_cm = st.selectbox("Select model for confusion matrix:", selected_models)
                    model = models[selected_model_cm]
                    y_pred = model.predict(X_test)
                    
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(y_test, y_pred)
                    
                    fig = px.imshow(cm, text_auto=True, aspect="auto",
                                  labels=dict(x="Predicted", y="Actual", color="Count"),
                                  title=f'Confusion Matrix - {selected_model_cm}')
                    fig.update_xaxes(ticktext=['No Default', 'Default'], tickvals=[0, 1])
                    fig.update_yaxes(ticktext=['No Default', 'Default'], tickvals=[0, 1])
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance
                st.subheader("🔍 Feature Importance")
                
                selected_model_fi = st.selectbox("Select model for feature importance:", 
                                               [m for m in selected_models if hasattr(models[m], 'feature_importances_')])
                
                if selected_model_fi:
                    model = models[selected_model_fi]
                    if hasattr(model, 'feature_importances_'):
                        feature_names = X_test.columns if hasattr(X_test, 'columns') else [f'Feature_{i}' for i in range(X_test.shape[1])]
                        importance_df = pd.DataFrame({
                            'feature': feature_names,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=True).tail(15)
                        
                        fig = px.bar(importance_df, x='importance', y='feature',
                                   title=f'Top 15 Feature Importances - {selected_model_fi}',
                                   orientation='h')
                        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Predictions
    with tabs[2]:
        st.markdown('<div class="section-header">🔮 Real-time Predictions</div>', unsafe_allow_html=True)
        
        if not st.session_state.models_trained:
            st.info("🤖 Please train models first using the sidebar")
        else:
            # Model selection
            selected_model = st.selectbox("Select model for prediction:", list(models.keys()))
            
            # Input method selection
            input_method = st.radio("Choose input method:", ["Manual Input", "Upload CSV", "Use Test Sample"])
            
            if input_method == "Manual Input":
                st.subheader("📝 Enter Applicant Information")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    credit_score = st.slider("Credit Score", 300, 850, 650)
                    income = st.number_input("Annual Income ($)", 10000, 500000, 50000, step=1000)
                    loan_amount = st.number_input("Loan Amount ($)", 1000, 100000, 20000, step=1000)
                    interest_rate = st.slider("Interest Rate (%)", 1.0, 30.0, 8.0, step=0.1)
                
                with col2:
                    employment_length = st.slider("Employment Length (years)", 0, 50, 5)
                    debt_to_income = st.slider("Debt-to-Income Ratio (%)", 0, 100, 30)
                    loan_purpose = st.selectbox("Loan Purpose", 
                                              ["debt_consolidation", "home_improvement", "credit_card", "major_purchase"])
                    home_ownership = st.selectbox("Home Ownership", ["rent", "own", "mortgage"])
                
                if st.button("🔮 Make Prediction"):
                    # Create input dataframe
                    input_data = pd.DataFrame({
                        'credit_score': [credit_score],
                        'income': [income],
                        'loan_amount': [loan_amount],
                        'interest_rate': [interest_rate],
                        'employment_length': [employment_length],
                        'debt_to_income': [debt_to_income],
                        'loan_purpose': [loan_purpose],
                        'home_ownership': [home_ownership]
                    })
                    
                    # Apply feature engineering
                    input_engineered = engineer_all_features(input_data)
                    
                    # Make prediction
                    model = models[selected_model]
                    X_input = input_engineered.drop('default', axis=1) if 'default' in input_engineered.columns else input_engineered
                    
                    prediction_results = make_prediction(model, X_input.values)
                    
                    if prediction_results:
                        summary = create_prediction_summary(prediction_results, X_input)
                        display_prediction_results(summary)
                        
                        # SHAP explanation
                        if st.checkbox("Show SHAP Explanation"):
                            st.subheader("🧠 SHAP Explanation")
                            
                            # Create SHAP explainer
                            explainer = create_shap_explainer(model, st.session_state.X_train)
                            if explainer:
                                shap_values = explainer.shap_values(X_input.values)
                                
                                # Local explanation
                                fig = plot_local_prediction_explanation(
                                    shap_values, X_input.values, 
                                    X_input.columns.tolist(), 
                                    sample_idx=0, 
                                    model_name=selected_model
                                )
                                st.plotly_chart(fig, use_container_width=True)
            
            elif input_method == "Upload CSV":
                st.subheader("📁 Upload Batch Data")
                uploaded_file = st.file_uploader("Choose a CSV file for batch predictions", type="csv")
                
                if uploaded_file is not None:
                    try:
                        batch_data = pd.read_csv(uploaded_file)
                        st.write("Uploaded data preview:")
                        st.dataframe(batch_data.head())
                        
                        if st.button("🔮 Batch Predict"):
                            # Apply feature engineering
                            batch_engineered = engineer_all_features(batch_data)
                            X_batch = batch_engineered.drop('default', axis=1) if 'default' in batch_engineered.columns else batch_engineered
                            
                            # Make predictions
                            model = models[selected_model]
                            prediction_results = make_prediction(model, X_batch.values)
                            
                            if prediction_results:
                                # Create summary for each sample
                                results = []
                                for i in range(len(X_batch)):
                                    summary = create_prediction_summary(prediction_results, X_batch, sample_idx=i)
                                    results.append({
                                        'Sample': i + 1,
                                        'Prediction': 'Default' if summary['prediction'] == 1 else 'No Default',
                                        'Probability': f"{summary['probabilities']['default']:.1%}",
                                        'Risk Level': summary['risk_assessment']['risk_level'].title(),
                                        'Recommendation': summary['risk_assessment']['recommendation'].replace('_', ' ')
                                    })
                                
                                results_df = pd.DataFrame(results)
                                st.write("Batch prediction results:")
                                st.dataframe(results_df)
                                
                                # Download results
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results",
                                    data=csv,
                                    file_name="loan_predictions.csv",
                                    mime="text/csv"
                                )
                    
                    except Exception as e:
                        st.error(f"❌ Failed to process file: {str(e)}")
            
            elif input_method == "Use Test Sample":
                st.subheader("🎲 Use Test Sample")
                
                if st.session_state.X_test is not None:
                    sample_idx = st.slider("Select test sample:", 0, len(st.session_state.X_test) - 1, 0)
                    
                    if st.button("🔮 Predict Selected Sample"):
                        # Get test sample
                        X_sample = st.session_state.X_test.iloc[sample_idx:sample_idx + 1]
                        y_true = st.session_state.y_test.iloc[sample_idx]
                        
                        # Make prediction
                        model = models[selected_model]
                        prediction_results = make_prediction(model, X_sample.values)
                        
                        if prediction_results:
                            summary = create_prediction_summary(prediction_results, X_sample)
                            display_prediction_results(summary)
                            
                            # Show actual result
                            st.write(f"**Actual Result:** {'Default' if y_true == 1 else 'No Default'}")
                            
                            # SHAP explanation
                            if st.checkbox("Show SHAP Explanation"):
                                st.subheader("🧠 SHAP Explanation")
                                
                                # Create SHAP explainer
                                explainer = create_shap_explainer(model, st.session_state.X_train)
                                if explainer:
                                    shap_values = explainer.shap_values(X_sample.values)
                                    
                                    # Local explanation
                                    fig = plot_local_prediction_explanation(
                                        shap_values, X_sample.values, 
                                        X_sample.columns.tolist(), 
                                        sample_idx=0, 
                                        model_name=selected_model
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Documentation
    with tabs[3]:
        st.markdown('<div class="section-header">📚 Documentation & Help</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ## 🏦 Loan Default Prediction System
        
        This is a comprehensive machine learning application for predicting loan defaults, designed for production use in financial institutions.
        
        ### 🎯 Business Problem
        
        **Problem**: Financial institutions need to assess the risk of loan applicants defaulting on their loans to make informed lending decisions.
        
        **Solution**: A machine learning system that predicts the probability of loan default based on applicant characteristics and loan terms.
        
        **Business Impact**:
        - Reduce financial losses from bad loans
        - Improve loan approval efficiency
        - Ensure fair and consistent lending decisions
        - Meet regulatory requirements for explainable AI
        
        ### 🛠️ Technical Architecture
        
        **Data Pipeline**:
        1. Data ingestion (CSV upload or sample data generation)
        2. Data validation and quality checks
        3. Feature engineering (financial ratios, risk scores)
        4. Preprocessing (scaling, encoding, missing value handling)
        
        **Machine Learning Models**:
        - Logistic Regression (baseline model)
        - Random Forest (ensemble method)
        - XGBoost (gradient boosting)
        
        **Evaluation Metrics**:
        - Accuracy, Precision, Recall, F1-Score
        - ROC-AUC, Average Precision
        - Confusion Matrix analysis
        
        **Explainability**:
        - SHAP (SHapley Additive exPlanations) values
        - Global feature importance
        - Local prediction explanations
        
        ### 📊 How to Use This Application
        
        **1. Data Loading** (Sidebar):
        - Generate sample data for testing
        - Upload your own CSV file
        - Use previously saved data
        
        **2. Model Training** (Sidebar):
        - Select models to train
        - Click "Train Models" button
        - Monitor training progress
        
        **3. EDA Tab**:
        - Explore data distributions
        - Analyze default rates by features
        - View correlation matrices
        
        **4. Model Performance Tab**:
        - Compare model metrics
        - View ROC curves and confusion matrices
        - Analyze feature importance
        
        **5. Predictions Tab**:
        - Make single predictions with manual input
        - Process batch predictions with CSV upload
        - Use test samples for validation
        - View SHAP explanations for transparency
        
        ### 🔧 Technical Requirements
        
        **Python Libraries**:
        - pandas, numpy (data manipulation)
        - scikit-learn (machine learning)
        - xgboost (gradient boosting)
        - shap (explainability)
        - streamlit (web interface)
        - plotly (visualizations)
        
        **System Requirements**:
        - Python 3.7+
        - 4GB+ RAM recommended
        - Modern web browser
        
        ### 🚀 Deployment Options
        
        **Streamlit Cloud** (Recommended):
        - Free hosting for public repositories
        - Automatic deployment from GitHub
        - Built-in HTTPS and scalability
        
        **Local Deployment**:
        - Run with `streamlit run app.py`
        - Accessible at http://localhost:8501
        
        **Docker Deployment**:
        - Containerized application
        - Consistent environment
        - Easy scaling and management
        
        ### 📈 Model Performance Expectations
        
        **Typical Results** (with sample data):
        - ROC-AUC: 0.85-0.95
        - Precision: 0.70-0.85
        - Recall: 0.65-0.80
        - F1-Score: 0.70-0.80
        
        **Performance Factors**:
        - Data quality and quantity
        - Feature engineering effectiveness
        - Model hyperparameter tuning
        - Class imbalance handling
        
        ### 🔍 Explainability Features
        
        **SHAP Integration**:
        - Global feature importance across all predictions
        - Local explanations for individual predictions
        - Force plots showing feature contributions
        - Summary plots for model interpretability
        
        **Business Benefits**:
        - Regulatory compliance (GDPR, Fair Credit Reporting Act)
        - Customer communication and transparency
        - Model debugging and improvement
        - Risk factor identification
        
        ### 🛡️ Risk Management
        
        **Risk Categories**:
        - **Low Risk** (0-20%): Recommend approval
        - **Medium Risk** (20-40%): Approve with conditions
        - **High Risk** (40-70%): Manual review required
        - **Very High Risk** (70%+): Recommend rejection
        
        **Recommendation Engine**:
        - Automated risk assessment
        - Confidence scoring
        - Threshold-based decision making
        - Customizable risk parameters
        
        ### 🔮 Future Enhancements
        
        **Planned Features**:
        - Real-time model monitoring
        - Automated retraining pipelines
        - Advanced ensemble methods
        - Time-series analysis
        - Portfolio risk management
        - Mobile application interface
        
        **Performance Improvements**:
        - Hyperparameter optimization (Optuna)
        - Advanced feature selection
        - Deep learning models
        - Cross-validation enhancements
        
        ### 📞 Support & Contact
        
        **For Questions**:
        - Check the GitHub repository for issues
        - Review the documentation in the `/docs` folder
        - Contact the development team
        
        **Contributing**:
        - Fork the repository
        - Create a feature branch
        - Submit a pull request with detailed description
        
        ---
        
        **Version**: 1.0.0  
        **Last Updated**: January 2024  
        **License**: MIT License
        """)

if __name__ == "__main__":
    main()