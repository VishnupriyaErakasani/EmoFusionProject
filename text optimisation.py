# @title TextFeature optimisation(Preprocessing step)
print(type(df))
print(df.columns)
print("Total samples:", len(df))
print("Missing utterances:", df['Utterance'].isna().sum())
print(type(df))
print(df.columns)
print("Total samples:", len(df))
print("Missing utterances:", df['Utterance'].isna().sum())

import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)      # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)     # keep letters only
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['clean_text'] = df['Utterance'].astype(str).apply(clean_text)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),     # unigrams + bigrams
    max_features=15000,    # richer vocabulary
    min_df=3,              # remove rare noise
    max_df=0.9,            # remove very common words
    sublinear_tf=True      # log scaling
)

X_tfidf = vectorizer.fit_transform(df['clean_text'])

print("TF-IDF shape:", X_tfidf.shape)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Emotion'])

print("Classes:", label_encoder.classes_)

from sklearn.preprocessing import Normalizer

normalizer = Normalizer()
X_tfidf = normalizer.fit_transform(X_tfidf)

import pickle

pickle.dump(X_tfidf, open("X_text_tfidf.pkl", "wb"))
pickle.dump(y, open("y_text.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))
pickle.dump(label_encoder, open("label_encoder.pkl", "wb"))

print("Text features saved successfully")

# @title TextFeature optimisation(model comparision step)
import pickle

X_tfidf = pickle.load(open("X_text_tfidf.pkl", "rb"))
y = pickle.load(open("y_text.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

print(X_tfidf.shape, y.shape)

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(X_train.shape, X_val.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

lr_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_val)

print("Logistic Regression Accuracy:", accuracy_score(y_val, y_pred_lr))
print(classification_report(y_val, y_pred_lr, target_names=label_encoder.classes_))

from sklearn.svm import LinearSVC

svm_model = LinearSVC(
    class_weight="balanced"
)

svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_val)

print("Linear SVM Accuracy:", accuracy_score(y_val, y_pred_svm))
print(classification_report(y_val, y_pred_svm, target_names=label_encoder.classes_))

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_val)

print("Random Forest Accuracy:", accuracy_score(y_val, y_pred_rf))
print(classification_report(y_val, y_pred_rf, target_names=label_encoder.classes_))

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

models_data = {
    "Logistic Regression": y_pred_lr,
    "Linear SVM": y_pred_svm,
    "Random Forest": y_pred_rf
}

rows = []

for model_name, y_pred in models_data.items():
    report = classification_report(y_val, y_pred, output_dict=True)
    rows.append([
        model_name,
        accuracy_score(y_val, y_pred),
        report["macro avg"]["precision"],
        report["macro avg"]["recall"],
        report["macro avg"]["f1-score"],
        report["weighted avg"]["f1-score"]
    ])

comparison_df = pd.DataFrame(
    rows,
    columns=[
        "Model",
        "Accuracy",
        "Precision (Macro)",
        "Recall (Macro)",
        "F1-score (Macro)",
        "F1-score (Weighted)"
    ]
)

# Round for neat display
comparison_df.iloc[:, 1:] = comparison_df.iloc[:, 1:].round(4)

# Sort by best model
comparison_df = comparison_df.sort_values(
    by="F1-score (Weighted)",
    ascending=False
).reset_index(drop=True)

print(comparison_df)

