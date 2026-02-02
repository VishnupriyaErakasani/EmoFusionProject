# Text probabilities
y_prob_text = text_model.predict_proba(X_val_text)

# Audio probabilities
y_prob_audio = audio_model.predict_proba(X_val_audio)

# Video probabilities
y_prob_video = video_model.predict_proba(X_val_video)

print(y_prob_text.shape, y_prob_audio.shape, y_prob_video.shape)

y_prob_fused = (
    y_prob_text +
    y_prob_audio +
    y_prob_video
) / 3

import numpy as np

y_pred_late = np.argmax(y_prob_fused, axis=1)

from sklearn.metrics import accuracy_score, classification_report

print("Late Fusion Accuracy:", accuracy_score(y_val_text, y_pred_late))
print(classification_report(y_val_text, y_pred_late))

w_text, w_audio, w_video = 0.5, 0.25, 0.25

y_prob_fused_w = (
    w_text * y_prob_text +
    w_audio * y_prob_audio +
    w_video * y_prob_video
)

y_pred_w = np.argmax(y_prob_fused_w, axis=1)

print("Weighted Late Fusion Accuracy:",
      accuracy_score(y_val_text, y_pred_w))

print("Pred distribution:", np.bincount(y_pred_late))
print("True distribution:", np.bincount(y_val_text))
