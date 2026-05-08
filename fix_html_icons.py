import os

filepath = r"c:\Users\HP\Pictures\LOAN DEFAULT PREDICTION\loan-default-prediction\app.py"

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace icons inside HTML div tags with empty string or outside the tag
# Actually, the user just wants the literal string ":material/account_balance:" gone from the rendered page, and preferably replaced with a working icon.
# Let's replace the st.markdown HTML with st.markdown text that natively supports Streamlit icons

replacements = {
    '<div class="main-header">:material/account_balance: Loan Default Prediction System</div>': '<div class="main-header"><span style="vertical-align: middle;" class="material-symbols-rounded">account_balance</span> Loan Default Prediction System</div>',
    '<div class="section-header">:material/analytics: Exploratory Data Analysis</div>': '<div class="section-header"><span style="vertical-align: middle;" class="material-symbols-rounded">analytics</span> Exploratory Data Analysis</div>',
    '<div class="section-header">:material/smart_toy: Model Performance</div>': '<div class="section-header"><span style="vertical-align: middle;" class="material-symbols-rounded">smart_toy</span> Model Performance</div>',
    '<div class="section-header">:material/online_prediction: Real-time Predictions</div>': '<div class="section-header"><span style="vertical-align: middle;" class="material-symbols-rounded">online_prediction</span> Real-time Predictions</div>',
    '<div class="section-header">:material/menu_book: Documentation & Help</div>': '<div class="section-header"><span style="vertical-align: middle;" class="material-symbols-rounded">menu_book</span> Documentation & Help</div>',
}

for old, new in replacements.items():
    content = content.replace(old, new)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed HTML material icons in app.py")
