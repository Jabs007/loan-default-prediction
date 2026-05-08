import os
import re

filepath = r"c:\Users\HP\Pictures\LOAN DEFAULT PREDICTION\loan-default-prediction\app.py"

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Strip out the span tags we added
content = re.sub(r'<span[^>]*class="material-symbols-rounded"[^>]*>[^<]*</span>\s*', '', content)

# Give it a simpler name
content = content.replace("Loan Default Prediction System", "Loan Risk Analyzer")

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Removed broken HTML icons and simplified title.")
