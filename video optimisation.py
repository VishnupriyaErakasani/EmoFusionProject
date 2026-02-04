#@title Video feature optimisation
print("Video features shape:", X_video.shape)
print("Video labels shape:", y_video.shape)

print("Any NaNs in video features:", np.isnan(X_video).any())
print("Unique labels:", np.unique(y_video))

from sklearn.preprocessing import StandardScaler

scaler_video = StandardScaler()
X_video_scaled = scaler_video.fit_transform(X_video)

from sklearn.model_selection import train_test_split

X_train_vid, X_val_vid, y_train_vid, y_val_vid = train_test_split(
    video_features,
    video_labels,
    test_size=0.2,
    random_state=42,
    stratify=video_labels
)

print("Train shape:", X_train_vid.shape)
print("Val shape:", X_val_vid.shape)

from sklearn.linear_model import LogisticRegression

video_lr = LogisticRegression(
    max_iter=500,
    class_weight="balanced",
    n_jobs=-1
)

start = time.time()
video_lr.fit(X_train_vid, y_train_vid)
train_time = time.time() - start

y_pred_lr = video_lr.predict(X_val_vid)

print("Logistic Regression (Video)")
print("Accuracy:", accuracy_score(y_val_vid, y_pred_lr))
print("F1-score:", f1_score(y_val_vid, y_pred_lr, average="weighted"))
print(classification_report(y_val_vid, y_pred_lr))

joblib.dump(video_lr, "video_lr_model.pkl")

from sklearn.ensemble import RandomForestClassifier

video_rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

start = time.time()
video_rf.fit(X_train_vid, y_train_vid)
train_time = time.time() - start

y_pred_rf = video_rf.predict(X_val_vid)

print("Random Forest – Video")
print("Accuracy:", accuracy_score(y_val_vid, y_pred_rf))
print("F1-score:", f1_score(y_val_vid, y_pred_rf, average="weighted"))
print(classification_report(y_val_vid, y_pred_rf))

joblib.dump(video_rf, "video_rf_model.pkl")

from sklearn.neural_network import MLPClassifier

video_mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation="relu",
    solver="adam",
    max_iter=300,
    random_state=42
)

start = time.time()
video_mlp.fit(X_train_vid, y_train_vid)
train_time = time.time() - start

y_pred_mlp = video_mlp.predict(X_val_vid)

print("MLP – Video")
print("Accuracy:", accuracy_score(y_val_vid, y_pred_mlp))
print("F1-score:", f1_score(y_val_vid, y_pred_mlp, average="weighted"))
print(classification_report(y_val_vid, y_pred_mlp))

joblib.dump(video_mlp, "video_mlp_model.pkl")

from sklearn.svm import SVC

video_svm = SVC(
    kernel="rbf",
    probability=True,
    class_weight="balanced",
    C=1.0,
    gamma="scale"
)

start = time.time()
video_svm.fit(X_train_vid, y_train_vid)
train_time = time.time() - start

y_pred_svm = video_svm.predict(X_val_vid)

print("SVM (RBF) – Video")
print("Accuracy:", accuracy_score(y_val_vid, y_pred_svm))
print("F1-score:", f1_score(y_val_vid, y_pred_svm, average="weighted"))
print(classification_report(y_val_vid, y_pred_svm))

joblib.dump(video_svm, "video_svm_model.pkl")
