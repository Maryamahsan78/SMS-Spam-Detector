"""Streamlit demo for the SMS spam detector.
Run locally:  streamlit run streamlit_app.py
"""
import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from spam_utils import clean_text

DATA_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Load the saved pipeline. If it cannot be loaded (for example a different
    scikit-learn version), retrain the same tuned pipeline from the public dataset."""
    try:
        return joblib.load("spam_model.joblib")
    except Exception:
        df = pd.read_csv(DATA_URL, sep="\t", header=None, names=["label", "message"]).drop_duplicates("message")
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(preprocessor=clean_text, stop_words="english", ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("model", MultinomialNB(alpha=0.1)),
        ])
        return pipe.fit(df["message"], (df["label"] == "spam").astype(int))


model = load_model()

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩")
st.title("📩 SMS Spam Detector")
st.write("TF-IDF + Multinomial Naive Bayes trained on the SMS Spam Collection (English). "
         "Paste a message to see the probability that it is spam.")

EXAMPLES = {
    "Spam example": "WINNER!! You have won a 1000 pound prize. Call 09061701461 now to claim.",
    "Normal example": "Hey, are we still meeting for lunch tomorrow?",
}
choice = st.radio("Try an example or write your own:", ["Write my own", *EXAMPLES], horizontal=True)
message = st.text_area("SMS message", value=EXAMPLES.get(choice, ""), height=120)

if st.button("Check message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        p_spam = float(model.predict_proba([message])[0, 1])
        if p_spam >= 0.5:
            st.error(f"SPAM ({p_spam:.1%} probability)")
        else:
            st.success(f"Not spam ({1 - p_spam:.1%} probability)")
        st.progress(p_spam, text=f"Spam probability: {p_spam:.1%}")

st.caption("Limitations: trained on an old English-only SMS dataset, so it may miss modern spam.")
