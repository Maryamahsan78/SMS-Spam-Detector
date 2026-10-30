# SMS Spam Detection (TF-IDF + Naive Bayes)

Capstone project for my ML internship (Week 4). The model reads a text message and predicts whether it is **spam** or **ham** (a normal message). A small Gradio app lets you try it.

**Author:** Maryam ([GitHub: Maryamahsan78](https://github.com/Maryamahsan78))

## Problem
Binary text classification on the SMS Spam Collection. Marking a real message as spam (false positive) is the costlier mistake, so I judge the model by precision, recall and F1 for the spam class, not by accuracy. The data is imbalanced (about 87% ham), so a model that always says "ham" already gets about 0.87 accuracy.

## Data
- SMS Spam Collection (UCI Machine Learning Repository): 5,572 labelled English messages.
- 403 duplicate messages were removed, leaving 5,169 (12.6% spam).
- Stratified 80/20 split: 4,135 training and 1,034 test messages. The test set was used only once, at the end.

## Approach
1. **EDA:** spam messages are longer (about 139 vs 72 characters) and contain far more digits (about 16 vs 0.3 per message) and capital letters. Typical spam words are `free`, `txt`, `mobile`, `claim`, `reply`, `prize`.
2. **Cleaning:** lowercase, replace URLs, phone numbers, other numbers and money amounts with tokens, remove punctuation.
3. **Features:** TF-IDF (English stop words removed), fitted inside a scikit-learn `Pipeline` so there is no leakage during cross-validation.
4. **Models compared (5-fold CV, spam F1):** Linear SVM 0.935, Random Forest 0.918, Logistic Regression 0.859, Naive Bayes 0.850, Complement NB 0.838.
5. **Tuning:** `GridSearchCV` (scored by F1) over TF-IDF settings and the model's hyperparameter. Best Naive Bayes: `alpha=0.1`, word n-grams (1, 2), `min_df=1`, `sublinear_tf=True`, which raised CV F1 from 0.850 to 0.950. The tuned Linear SVM reached 0.952, a difference within the noise (fold-to-fold std about 0.008), so I kept Naive Bayes as the main model.

## Results (held-out test set, 1,034 messages)
| Metric | Score |
|---|---|
| Accuracy | 0.988 |
| Precision (spam) | 0.976 |
| Recall (spam) | 0.931 |
| F1 (spam) | 0.953 (95% bootstrap interval 0.925 to 0.976) |

The test F1 (0.953) matches the cross-validated F1 on the training set (0.950). Of 131 spam messages, 9 slipped through; only 3 of 903 normal messages were wrongly flagged (all very short messages such as "Then why you not responding"). The Linear SVM scored 0.941 on the test set and the always-ham baseline scored 0.

**Strongest spam signals learned by the model:** `claim`, `prize`, `guaranteed`, `won`, `cash`, `tone`, `ppm` and the url/number/money tokens.

## Limitations
- The data is small, old and English-only (mostly UK and Singapore messages), so the model may not work on modern spam or other languages.
- The test set has only 131 spam messages, so the scores have a wide margin of error.
- Spammers can avoid typical spam words, and the model sees only the words (not the sender or links).

## Demo
```bash
pip install -r requirements.txt
python app.py
```
Then open the local link that Gradio prints. Paste a message and see the spam probability.

## Files
| File | Purpose |
|---|---|
| `capstone_spam_detection.ipynb` | Full notebook: problem, EDA, preprocessing, modelling, evaluation, explanation |
| `spam_utils.py` | Text cleaning function (needed by the saved model and the app) |
| `spam_model.joblib` | Saved trained pipeline (cleaning + TF-IDF + Naive Bayes) |
| `app.py` | Gradio demo |
| `requirements.txt` | Python packages |

The notebook downloads the dataset automatically. If `spam_model.joblib` fails to load because of a different scikit-learn version, run the notebook once to create a new one.

## Tools
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Gradio, Google Colab

## Next steps
More data, character n-grams, a threshold tuned for higher precision, and a comparison with a small transformer model.
