import pandas as pd
import os

# Replace this with your printed dataset path
DATA_PATH = r"C:\Users\ekris\.cache\kagglehub\datasets\zaber666\meld-dataset\versions\1"

train_df = pd.read_csv(os.path.join(DATA_PATH, r"C:\Users\ekris\.cache\kagglehub\datasets\zaber666\meld-dataset\versions\1\MELD-RAW\MELD.Raw\train\train_sent_emo.csv"))
test_df = pd.read_csv(os.path.join(DATA_PATH, r"C:\Users\ekris\.cache\kagglehub\datasets\zaber666\meld-dataset\versions\1\MELD-RAW\MELD.Raw\test_sent_emo.csv"))
val_df = pd.read_csv(os.path.join(DATA_PATH, r"C:\Users\ekris\.cache\kagglehub\datasets\zaber666\meld-dataset\versions\1\MELD-RAW\MELD.Raw\dev_sent_emo.csv"))

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Validation shape:", val_df.shape)
print(train_df.head())
