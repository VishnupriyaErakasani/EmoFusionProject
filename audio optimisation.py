#@title Audio feature exraction optimisation
import librosa
import numpy as np

def extract_mfcc_optimized(audio_path, n_mfcc=40, max_len=300):
    try:
        y, sr = librosa.load(audio_path, sr=16000)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        features = np.vstack([mfcc, delta, delta2])  # (120, time)

        # pad / truncate
        if features.shape[1] < max_len:
            pad = max_len - features.shape[1]
            features = np.pad(features, ((0,0),(0,pad)))
        else:
            features = features[:, :max_len]

        return features.flatten()

    except:
        return None

from sklearn.preprocessing import StandardScaler

scaler_audio = StandardScaler()
X_audio_scaled = scaler_audio.fit_transform(X_audio)

print("Audio feature shape:", X_audio_scaled.shape)
print("Any NaNs:", np.isnan(X_audio_scaled).any())
print("Any infs:", np.isinf(X_audio_scaled).any())

np.save("audio_features_optimized.npy", X_audio_scaled)

#@title audio model optimisation and comparision
audio_features = {}   # key = utterance_id

import os

print(os.listdir("/content"))

import numpy as np

X_audio = np.load("audio_features.npy")
y_audio = np.load("audio_labels.npy")

print("Audio features shape:", X_audio.shape)
print("Audio labels shape:", y_audio.shape)

X_audio = np.load("audio_features_optimized.npy")
y_audio = np.load("audio_labels.npy")

from sklearn.model_selection import train_test_split

X_train_audio, X_val_audio, y_train_audio, y_val_audio = train_test_split(
    X_audio,
    y_audio,
    test_size=0.2,
    random_state=42,
    stratify=y_audio
)

print(X_train_audio.shape, X_val_audio.shape)
#@title audio model optimisation and comparision
audio_features = {}   # key = utterance_id

import os

print(os.listdir("/content"))

import numpy as np

X_audio = np.load("audio_features.npy")
y_audio = np.load("audio_labels.npy")

print("Audio features shape:", X_audio.shape)
print("Audio labels shape:", y_audio.shape)

X_audio = np.load("audio_features_optimized.npy")
y_audio = np.load("audio_labels.npy")

from sklearn.model_selection import train_test_split

X_train_audio, X_val_audio, y_train_audio, y_val_audio = train_test_split(
    X_audio,
    y_audio,
    test_size=0.2,
    random_state=42,
    stratify=y_audio
)

print(X_train_audio.shape, X_val_audio.shape)

#@title audio model comparision optimisation
from sklearn.model_selection import train_test_split

X_train_audio, X_val_audio, y_train_audio, y_val_audio = train_test_split(
    X_audio,
    y_audio,
    test_size=0.2,
    random_state=42,
    stratify=y_audio
)

print(X_train_audio.shape, X_val_audio.shape)

from sklearn.model_selection import train_test_split

X_train_audio, X_val_audio, y_train_audio, y_val_audio = train_test_split(
    X_audio_scaled,      # or X_audio if not scaled
    y_audio,
    test_size=0.2,
    random_state=42,
    stratify=y_audio
)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import time

model = LogisticRegression(max_iter=300, class_weight="balanced")

start = time.time()
model.fit(X_train_audio, y_train_audio)
train_time = time.time() - start

y_pred = model.predict(X_val_audio)

print("Logistic Regression Results")
print("Accuracy:", accuracy_score(y_val_audio, y_pred))
print("F1-score:", f1_score(y_val_audio, y_pred, average="weighted"))
print("Training Time (s):", round(train_time, 2))

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=200, random_state=42)

start = time.time()
model.fit(X_train_audio, y_train_audio)
train_time = time.time() - start

y_pred = model.predict(X_val_audio)

print("Random Forest Results")
print("Accuracy:", accuracy_score(y_val_audio, y_pred))
print("F1-score:", f1_score(y_val_audio, y_pred, average="weighted"))
print("Training Time (s):", round(train_time, 2))

from sklearn.neural_network import MLPClassifier

model = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42)

start = time.time()
model.fit(X_train_audio, y_train_audio)
train_time = time.time() - start

y_pred = model.predict(X_val_audio)
print("MLP Results")
print("Accuracy:", accuracy_score(y_val_audio, y_pred))
print("F1-score:", f1_score(y_val_audio, y_pred, average="weighted"))
print("Training Time (s):", round(train_time, 2))
