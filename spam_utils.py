import re

def clean_text(s):
    s = s.lower()
    s = re.sub(r'(https?://\S+|www\.\S+)', ' urltoken ', s)
    s = re.sub(r'[£$€]\s?\d[\d,.]*|\d[\d,.]*\s?(?:p|pounds|usd)\b', ' moneytoken ', s)
    s = re.sub(r'\d[\d\- ]{4,}\d|\d+', ' numtoken ', s)
    s = re.sub(r'[^a-z\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()
