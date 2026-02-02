# @title Audio Preprocessing
!pip install librosa soundfile --quiet

import os

for root, dirs, files in os.walk("/root"):
    if "MELD" in root or "meld" in root.lower():
        print(root)

for root, dirs, files in os.walk("/root"):
    for d in dirs:
        if "train" in d.lower():
            print(os.path.join(root, d))

import os

TRAIN_SPLITS_PATH = "/root/.cache/kagglehub/datasets/zaber666/meld-dataset/versions/1/MELD-RAW/MELD.Raw/train/train_splits"

print("train_splits exists:", os.path.exists(TRAIN_SPLITS_PATH))
print("Contents:", os.listdir(TRAIN_SPLITS_PATH))

VIDEO_MP4_PATH = "/root/.cache/kagglehub/datasets/zaber666/meld-dataset/versions/1/MELD-RAW/MELD.Raw/train/train_splits"

AUDIO_WAV_OUT = "/content/audio_wav"
os.makedirs(AUDIO_WAV_OUT, exist_ok=True)

import subprocess
from tqdm import tqdm

mp4_files = [f for f in os.listdir(VIDEO_MP4_PATH) if f.endswith(".mp4")]
failed_audio = []

for file in tqdm(mp4_files):
    mp4_path = os.path.join(VIDEO_MP4_PATH, file)
    wav_path = os.path.join(AUDIO_WAV_OUT, file.replace(".mp4", ".wav"))

    if os.path.exists(wav_path):
        continue

    try:
        subprocess.run(
            ["ffmpeg", "-i", mp4_path, "-ac", "1", "-ar", "16000", wav_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
    except:
        failed_audio.append(file)

print("Total MP4 files:", len(mp4_files))
print("Failed audio extractions:", len(failed_audio))

print(df.columns)

import pandas as pd

TRAIN_CSV = "/root/.cache/kagglehub/datasets/zaber666/meld-dataset/versions/1/MELD-RAW/MELD.Raw/train/train_sent_emo.csv"

df_av = pd.read_csv(TRAIN_CSV)

print(df_av.columns)
print(df_av.head())

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df_av["Emotion_Label"] = label_encoder.fit_transform(df_av["Emotion"])

row = df_av.iloc[0]
filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.wav"
print(filename)

import os
print(os.path.exists(os.path.join(AUDIO_WAV_OUT, filename)))

filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.wav"

import librosa
import numpy as np

def extract_mfcc(audio_path, n_mfcc=40, max_len=300):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # Pad or truncate
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]

        return mfcc.flatten()
    except Exception as e:
        return None

from tqdm import tqdm
import numpy as np
import os

audio_features = []
audio_labels = []
failed_mfcc = []

for _, row in tqdm(df_av.iterrows(), total=len(df_av)):
    filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.wav"
    wav_path = os.path.join(AUDIO_WAV_OUT, filename)

    if not os.path.exists(wav_path):
        failed_mfcc.append(filename)
        continue

    mfcc_feat = extract_mfcc(wav_path)

    if mfcc_feat is None:
        failed_mfcc.append(filename)
        continue

    audio_features.append(mfcc_feat)
    audio_labels.append(row["Emotion_Label"])

audio_features = np.array(audio_features)
audio_labels = np.array(audio_labels)

print("Audio feature matrix shape:", audio_features.shape)
print("Audio labels shape:", audio_labels.shape)
print("Failed MFCC files:", len(failed_mfcc))

np.save("audio_features.npy", audio_features)
np.save("audio_labels.npy", audio_labels)

print("Audio features and labels saved successfully")

#@title Audio model training
import numpy as np

X_audio = np.load("audio_features.npy")
y_audio = np.load("audio_labels.npy")

print("X_audio shape:", X_audio.shape)
print("y_audio shape:", y_audio.shape)

from sklearn.model_selection import train_test_split

X_train_audio, X_val_audio, y_train_audio, y_val_audio = train_test_split(
    X_audio,
    y_audio,
    test_size=0.2,
    random_state=42,
    stratify=y_audio
)

print(X_train_audio.shape, X_val_audio.shape)

from sklearn.linear_model import LogisticRegression

audio_model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    n_jobs=-1
)

audio_model.fit(X_train_audio, y_train_audio)

from sklearn.metrics import accuracy_score, classification_report

y_pred_audio = audio_model.predict(X_val_audio)

print("Audio Accuracy:", accuracy_score(y_val_audio, y_pred_audio))
print(classification_report(y_val_audio, y_pred_audio))

sample_audio_feat = X_val_audio[0].reshape(1, -1)
pred = audio_model.predict(sample_audio_feat)

print("Predicted label:", pred[0])
print("Predicted emotion:", label_encoder.inverse_transform(pred))
