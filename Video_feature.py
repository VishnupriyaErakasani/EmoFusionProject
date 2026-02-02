# @title Video Preprocessing
import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Check if path exists
VIDEO_MP4_PATH = (
    "/root/.cache/kagglehub/datasets/zaber666/meld-dataset/"
    "versions/1/MELD-RAW/MELD.Raw/train/train_splits"
)

print("Video path exists:", os.path.exists(VIDEO_MP4_PATH))
print("Sample files:", os.listdir(VIDEO_MP4_PATH)[:5])

#Cnn model initialising
cnn_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

print("CNN model ready")

# extracting video features
def extract_video_feature(video_path, max_frames=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        count += 1

    cap.release()

    if len(frames) == 0:
        return None

    frames = np.array(frames, dtype=np.float32)
    frames = preprocess_input(frames)

    features = cnn_model.predict(frames, verbose=0)
    return np.mean(features, axis=0)  # aggregate frames

#Processing video frames
video_features = []
video_labels = []
failed_videos = []

for _, row in tqdm(df_av.iterrows(), total=len(df_av)):
    filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
    video_path = os.path.join(VIDEO_MP4_PATH, filename)

    if not os.path.exists(video_path):
        failed_videos.append(filename)
        continue

    feat = extract_video_feature(video_path)

    if feat is None:
        failed_videos.append(filename)
        continue

    video_features.append(feat)
    video_labels.append(row["Emotion_Label"])

#convert and verify
video_features = np.array(video_features)
video_labels = np.array(video_labels)

print("Video feature matrix shape:", video_features.shape)
print("Video labels shape:", video_labels.shape)
print("Failed videos:", len(failed_videos))

#saving video preprocessing
np.save("video_features.npy", video_features)
np.save("video_labels.npy", video_labels)

print("Video features saved successfully")

# @title video model training
video_features   # numpy array, shape (N, feature_dim)
df               # dataframe containing Emotion_Label
label_encoder    # already fitted

import numpy as np

X_video = np.load("video_features.npy")
y_video = np.load("video_labels.npy")

print("X_video shape:", X_video.shape)
print("y_video shape:", y_video.shape)

from sklearn.model_selection import train_test_split

X_train_video, X_val_video, y_train_video, y_val_video = train_test_split(
    X_video,
    y_video,
    test_size=0.2,
    random_state=42,
    stratify=y_video
)

print(X_train_video.shape, X_val_video.shape)

from sklearn.linear_model import LogisticRegression

video_model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    n_jobs=-1
)

video_model.fit(X_train_video, y_train_video)

from sklearn.metrics import accuracy_score, classification_report

y_pred_video = video_model.predict(X_val_video)

print("Video Accuracy:", accuracy_score(y_val_video, y_pred_video))
print(classification_report(y_val_video, y_pred_video))

sample_video_feat = X_val_video[0].reshape(1, -1)
pred = video_model.predict(sample_video_feat)

print("Predicted label:", pred[0])
print("Predicted emotion:", label_encoder.inverse_transform(pred))

import joblib

joblib.dump(video_model, "video_model.pkl")
print("Video model saved successfully")

np.save("X_val_video.npy", X_val_video)
np.save("y_val_video.npy", y_val_video)

video_model = joblib.load("video_model.pkl")

X_val_video = np.load("X_val_video.npy")
y_val_video = np.load("y_val_video.npy")
