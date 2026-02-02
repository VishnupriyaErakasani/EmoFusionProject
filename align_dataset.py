import os

dataset_path = "/root/.cache/kagglehub/datasets/zaber666/meld-dataset/versions/1"
#dataset_path="/kaggle/input/meld-dataset"

print("Top-level files/folders in MELD dataset:")
print(os.listdir(dataset_path))

# Check subfolders
for folder in ["train", "dev", "test"]:
    folder_path = os.path.join(dataset_path, folder)
    if os.path.exists(folder_path):
        print(f"\nContents of {folder} folder:")
        print(os.listdir(folder_path))

import os

dataset_path = "/root/.cache/kagglehub/datasets/zaber666/meld-dataset/versions/1"
#dataset_path="/kaggle/input/meld-dataset"

# List top-level
print("Top-level folders/files:")
print(os.listdir(dataset_path))

# Explore MELD-RAW
meld_raw_path = os.path.join(dataset_path, "MELD-RAW")
print("\nContents of MELD-RAW:")
print(os.listdir(meld_raw_path))

import os
import pandas as pd

# Train CSV path
#train_csv="/kaggle/input/meld-dataset/MELD-RAW/MELD.Raw/train/train_sent_emo.csv"
train_csv="/root/.cache/kagglehub/datasets/zaber666/meld-dataset/versions/1/MELD-RAW/MELD.Raw/train/train_sent_emo.csv"



# Load train CSV
train_df = pd.read_csv(train_csv)

# Preview
print("Train dataset preview:")
print(train_df.head())

# Verify shape and labels
print("\nTrain shape:", train_df.shape)
print("Unique emotion labels:", train_df['Emotion'].unique())
print("Unique sentiment labels:", train_df['Sentiment'].unique())

import os

os.listdir("/content")

import os
os.listdir(path)

import os

#BASE_PATH = "/kaggle/input/meld-dataset"
BASE_PATH="/root/.cache/kagglehub/datasets/zaber666/meld-dataset/versions/1"
RAW_PATH = os.path.join(BASE_PATH, "MELD-RAW", "MELD.Raw")

print("Train folder:", os.listdir(os.path.join(RAW_PATH, "train")))

import pandas as pd

train_csv = os.path.join(RAW_PATH, "train", "train_sent_emo.csv")
df = pd.read_csv(train_csv)

print("Dataset shape:", df.shape)
df.head()

df = df[['Utterance', 'Emotion']]

print(df.shape)
df.head()
