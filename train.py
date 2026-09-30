import os
import pandas as pd
import numpy as np
import pickle
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

print("Loading params.yaml...")
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

RANDOM_SEED = params["random_seed"]
MODEL_PARAMS = params["model"]

print("Loading training dataset...")
df_train = pd.read_csv("data/splits/train.csv")
feature_cols = ["weekly_self_study_hours", "attendance_percentage", "class_participation"]
X_train = df_train[feature_cols].copy()
y_train = df_train["grade"].copy()

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

preprocessing_pipeline = Pipeline([
    ("scaler", StandardScaler())
])
X_train_scaled = preprocessing_pipeline.fit_transform(X_train)

model = RandomForestClassifier(
    random_state=RANDOM_SEED,
    n_estimators=MODEL_PARAMS["n_estimators"],
    max_depth=MODEL_PARAMS["max_depth"],
    min_samples_split=MODEL_PARAMS["min_samples_split"],
    class_weight=MODEL_PARAMS["class_weight"]
)
model.fit(X_train_scaled, y_train_encoded)

train_preds = model.predict(X_train_scaled)
train_accuracy = (train_preds == y_train_encoded).mean()
print(f"Train accuracy: {train_accuracy:.4f}")

os.makedirs("models", exist_ok=True)
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("models/preprocessing_pipeline.pkl", "wb") as f:
    pickle.dump(preprocessing_pipeline, f)
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
with open("models/feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

print(f"\nModel saved to: models/model.pkl")
print(f"Preprocessing pipeline saved to: models/preprocessing_pipeline.pkl")
print(f"Label encoder saved to: models/label_encoder.pkl")
print(f"Features used: {feature_cols}")
print(f"Target classes: {le.classes_}")
print(f"random_state={RANDOM_SEED}")