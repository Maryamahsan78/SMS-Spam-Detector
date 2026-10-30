"""Gradio demo for the SMS spam detector.
Run:  python app.py   (needs spam_model.joblib and spam_utils.py in the same folder)
"""
import joblib
import gradio as gr
from spam_utils import clean_text  # noqa: F401  (needed so joblib can load the saved pipeline)

model = joblib.load("spam_model.joblib")

EXAMPLES = [
    ["WINNER!! You have won a 1000 pound prize. Call 09061701461 now to claim."],
    ["Free entry in a weekly competition! Text WIN to 80086 to receive your prize."],
    ["Hey, are we still meeting for lunch tomorrow?"],
    ["Can you send me the notes from today's lecture?"],
]


def predict(message: str):
    if not message or not message.strip():
        return {"spam": 0.0, "ham": 0.0}
    p_spam = float(model.predict_proba([message])[0, 1])
    return {"spam": p_spam, "ham": 1.0 - p_spam}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=4, label="SMS message", placeholder="Paste a message here..."),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    examples=EXAMPLES,
    title="SMS Spam Detector",
    description="TF-IDF + Multinomial Naive Bayes trained on the SMS Spam Collection (English). "
                "Returns the probability that the message is spam.",
)

if __name__ == "__main__":
    demo.launch()
