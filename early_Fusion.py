# @title Early Fusion

print(type(y))
print(y.shape)

import numpy as np

# Text
X_text = X_tfidf.toarray()   # convert sparse → dense
#y = y.values                # ensure numpy array

# Audio
X_audio = np.load("audio_features.npy")
y_audio = np.load("audio_labels.npy")

# Video
X_video = np.load("video_features.npy")
y_video = np.load("video_labels.npy")

print(X_text.shape, X_audio.shape, X_video.shape)

min_len = min(
    X_text.shape[0],
    X_audio.shape[0],
    X_video.shape[0]
)

X_text  = X_text[:min_len]
X_audio = X_audio[:min_len]
X_video = X_video[:min_len]

y = y[:min_len]

from sklearn.preprocessing import StandardScaler

scaler_audio = StandardScaler()
scaler_video = StandardScaler()

X_audio = scaler_audio.fit_transform(X_audio)
X_video = scaler_video.fit_transform(X_video)

X_fused = np.hstack([X_text, X_audio, X_video])

print("Early fusion feature shape:", X_fused.shape)

from sklearn.model_selection import train_test_split

X_train_f, X_val_f, y_train_f, y_val_f = train_test_split(
    X_fused,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

from sklearn.linear_model import LogisticRegression

fusion_model = LogisticRegression(
    max_iter=4000,
    class_weight="balanced",
    n_jobs=-1
)

fusion_model.fit(X_train_f, y_train_f)

from sklearn.metrics import accuracy_score, classification_report

y_pred_f = fusion_model.predict(X_val_f)

print("Early Fusion Accuracy:", accuracy_score(y_val_f, y_pred_f))
print(classification_report(y_val_f, y_pred_f))

import joblib

joblib.dump(fusion_model, "early_fusion_model.pkl")
joblib.dump(scaler_audio, "scaler_audio.pkl")
joblib.dump(scaler_video, "scaler_video.pkl")

print("Early fusion model saved")

#@title sanity checks for early fusion
print("Text:", X_text.shape)
print("Audio:", X_audio.shape)
print("Video:", X_video.shape)
print("Labels:", y.shape)
print("Fused:", X_fused.shape)

import numpy as np

print("NaNs in fused:", np.isnan(X_fused).sum())
print("Infs in fused:", np.isinf(X_fused).sum())

import numpy as np

print("Train label distribution:", np.bincount(y_train_f))
print("Val label distribution:", np.bincount(y_val_f))

train_acc = fusion_model.score(X_train_f, y_train_f)
val_acc   = fusion_model.score(X_val_f, y_val_f)

print("Train acc:", train_acc)
print("Val acc:", val_acc)

idx = 0
sample = X_val_f[idx].reshape(1, -1)
pred = fusion_model.predict(sample)

print("Predicted:", label_encoder.inverse_transform(pred))
print("Actual:", label_encoder.inverse_transform([y_val_f[idx]]))

import numpy as np

coef = np.abs(fusion_model.coef_).mean(axis=0)

print("Text weight:", coef[:X_text.shape[1]].mean())
print("Audio weight:", coef[X_text.shape[1]:X_text.shape[1]+X_audio.shape[1]].mean())
print("Video weight:", coef[-X_video.shape[1]:].mean())
