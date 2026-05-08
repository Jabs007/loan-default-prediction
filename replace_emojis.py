import os
import re

emoji_map = {
    '🏦': ':material/account_balance:',
    '📊': ':material/analytics:',
    '🤖': ':material/smart_toy:',
    '🔮': ':material/online_prediction:',
    '📚': ':material/menu_book:',
    '📋': ':material/list_alt:',
    '📝': ':material/edit_document:',
    '📈': ':material/trending_up:',
    '🎯': ':material/my_location:',
    '🔗': ':material/link:',
    '🔍': ':material/search:',
    '✅': ':material/check_circle:',
    '❌': ':material/cancel:',
    '⚠️': ':material/warning:',
    '🚨': ':material/error:',
    '🧠': ':material/psychology:',
    '📁': ':material/folder:',
    '📥': ':material/download:',
    '🎲': ':material/casino:',
    '🛠️': ':material/build:',
    '🔧': ':material/handyman:',
    '🚀': ':material/rocket_launch:',
    '🛡️': ':material/shield:',
    '📞': ':material/phone:',
    '💾': ':material/save:',
    '📂': ':material/folder_open:',
    '🔴': ':material/error:',
    '🟡': ':material/warning:',
    '🟢': ':material/check_circle:'
}

def replace_emojis_in_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace page_icon="🏦" -> page_icon=":material/account_balance:"
        content = content.replace('page_icon="🏦"', 'page_icon=":material/account_balance:"')
        
        for emoji, icon in emoji_map.items():
            content = content.replace(emoji, icon)
            
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

# Process app.py and src/*.py
app_py = r"c:\Users\HP\Pictures\LOAN DEFAULT PREDICTION\loan-default-prediction\app.py"
src_dir = r"c:\Users\HP\Pictures\LOAN DEFAULT PREDICTION\loan-default-prediction\src"

replace_emojis_in_file(app_py)

for filename in os.listdir(src_dir):
    if filename.endswith(".py"):
        filepath = os.path.join(src_dir, filename)
        replace_emojis_in_file(filepath)
