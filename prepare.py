import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

print("Loading params.yaml...")
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

RANDOM_SEED = params["random_seed"]
SAMPLE_SIZE = params["data"]["sample_size"]

print("Loading raw dataset...")
df_raw = pd.read_csv("student_performance.csv")
print(f"Input shape: {df_raw.shape}")

print("Validating required columns...")
required_cols = ["weekly_self_study_hours", "attendance_percentage", "class_participation", "grade"]
missing_cols = [col for col in required_cols if col not in df_raw.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

print("Checking missing values...")
missing = df_raw.isnull().sum().sum()
print(f"Total missing values: {missing}")

print("Checking duplicates...")
duplicates = df_raw.duplicated().sum()
print(f"Duplicate rows: {duplicates}")
if duplicates > 0:
    df_raw = df_raw.drop_duplicates()

valid_features = ["weekly_self_study_hours", "attendance_percentage", "class_participation"]
target = "grade"

df_sampled, _ = train_test_split(
    df_raw, test_size=(df_raw.shape[0] - SAMPLE_SIZE) / df_raw.shape[0], 
    random_state=RANDOM_SEED, stratify=df_raw["grade"]
)

df_processed = df_sampled[valid_features + [target]].copy()

os.makedirs("data/processed", exist_ok=True)
df_processed.to_csv("data/processed/processed.csv", index=False)

print(f"\nOutput shape: {df_processed.shape}")
print(f"Output saved to: data/processed/processed.csv")
print(f"Features: {valid_features}")
print(f"Target: {target}")
print(f"student_id excluded: True")
print(f"total_score excluded: True")
print(f"Stratified {SAMPLE_SIZE}-row sample with random_state={RANDOM_SEED}")