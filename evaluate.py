import os
import pickle
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("Loading params.yaml...")
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

RANDOM_SEED = params["random_seed"]

print("Loading test dataset...")
df_test = pd.read_csv("data/splits/test.csv")
feature_cols = ["weekly_self_study_hours", "attendance_percentage", "class_participation"]
X_test = df_test[feature_cols].copy()
y_test = df_test["grade"].copy()

print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

print("Loading trained model...")
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/preprocessing_pipeline.pkl", "rb") as f:
    preprocessing_pipeline = pickle.load(f)
with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

print("Generating predictions...")
X_test_scaled = preprocessing_pipeline.transform(X_test)
test_preds = model.predict(X_test_scaled)

y_test_encoded = le.transform(y_test)

print("Calculating metrics...")
test_accuracy = accuracy_score(y_test_encoded, test_preds)
precision = precision_score(y_test_encoded, test_preds, average='macro', zero_division=0)
recall = recall_score(y_test_encoded, test_preds, average='macro', zero_division=0)
f1 = f1_score(y_test_encoded, test_preds, average='macro', zero_division=0)

print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro): {recall:.4f}")
print(f"F1-Score (macro): {f1:.4f}")

cm = confusion_matrix(y_test_encoded, test_preds, labels=range(len(le.classes_)))
print(f"\nConfusion Matrix:")
print(f"  Classes: {le.classes_}")
for i, row in enumerate(cm):
    print(f"  Pred={le.classes_[i]}: {row}")

os.makedirs("reports", exist_ok=True)
metrics = {
    "accuracy": test_accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "n_classes": 5,
    "averaging_method": "macro",
    "classes": list(le.classes_)
}

import json
with open("reports/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nMetrics saved to: reports/metrics.json")
print("Evaluating with macro averaging for multiclass classification")