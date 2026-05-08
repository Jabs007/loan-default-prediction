import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import shap

st.set_page_config(page_title="Loan Risk Analyzer", page_icon=":material/account_balance:", layout="wide")

st.title(":material/account_balance: Loan Risk Analyzer")

st.markdown("""
Welcome to the Loan Risk Analyzer. This application predicts the risk of loan default based on applicant data.
""")

@st.cache_data
def generate_data():
    np.random.seed(42)
    n_samples = 2000
    df = pd.DataFrame({
        'credit_score': np.random.randint(300, 850, n_samples),
        'income': np.random.randint(20000, 150000, n_samples),
        'loan_amount': np.random.randint(1000, 50000, n_samples),
        'interest_rate': np.random.uniform(3.0, 25.0, n_samples),
        'employment_length': np.random.randint(0, 40, n_samples),
        'debt_to_income': np.random.uniform(0, 60, n_samples)
    })
    
    # Calculate synthetic risk
    risk = (-df['credit_score']/850 * 2) + (df['debt_to_income']/60 * 1.5) + (df['interest_rate']/25 * 1.5)
    df['default'] = (risk > np.median(risk)).astype(int)
    return df

df = generate_data()

tabs = st.tabs([":material/analytics: Data Overview", ":material/smart_toy: Model Performance", ":material/online_prediction: Predictions"])

with tabs[0]:
    st.subheader(":material/analytics: Dataset Preview")
    st.dataframe(df.head())
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", len(df))
    col2.metric("Features", len(df.columns)-1)
    col3.metric("Default Rate", f"{df['default'].mean():.1%}")

# Train model
X = df.drop('default', axis=1)
y = df['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

with tabs[1]:
    st.subheader(":material/smart_toy: Model Evaluation")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    st.success(f"Model trained successfully! Accuracy: {acc:.2%}")
    
    st.text("Classification Report:")
    st.code(classification_report(y_test, preds))

with tabs[2]:
    st.subheader(":material/edit_document: Enter Applicant Details")
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            credit_score = st.number_input("Credit Score", 300, 850, 700)
            income = st.number_input("Annual Income ($)", 10000, 500000, 60000)
            loan_amount = st.number_input("Loan Amount ($)", 1000, 100000, 15000)
        with col2:
            interest_rate = st.number_input("Interest Rate (%)", 1.0, 30.0, 7.5)
            employment_length = st.number_input("Employment Length (years)", 0, 50, 5)
            debt_to_income = st.number_input("Debt-to-Income Ratio (%)", 0.0, 100.0, 20.0)
            
        submitted = st.form_submit_button("Predict Default Risk")
        
        if submitted:
            input_df = pd.DataFrame([[credit_score, income, loan_amount, interest_rate, employment_length, debt_to_income]],
                                    columns=X.columns)
            prob = model.predict_proba(input_df)[0][1]
            pred = model.predict(input_df)[0]
            
            if pred == 1:
                st.error(f":material/warning: High Risk! Probability of Default: {prob:.1%}")
            else:
                st.success(f":material/check_circle: Low Risk! Probability of Default: {prob:.1%}")