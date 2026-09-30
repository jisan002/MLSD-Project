import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

print("Loading params.yaml...")
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

RANDOM_SEED = params["random_seed"]
TEST_SIZE = params["data"]["test_size"]

print("Loading processed dataset...")
df = pd.read_csv("data/processed/processed.csv")
print(f"Input shape: {df.shape}")

feature_cols = ["weekly_self_study_hours", "attendance_percentage", "class_participation"]

train_df, test_df = train_test_split(
    df, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=df["grade"]
)

os.makedirs("data/splits", exist_ok=True)

train_df.to_csv("data/splits/train.csv", index=False)
test_df.to_csv("data/splits/test.csv", index=False)

print(f"\ntrain shape: {train_df.shape}")
print(f"test shape: {test_df.shape}")
print(f"Train class distribution:")
print(train_df["grade"].value_counts().sort_index())
print(f"\nTest class distribution:")
print(test_df["grade"].value_counts().sort_index())
print(f"\nOutputs saved to data/splits/")
print(f"Stratified 80/20 split with random_state={RANDOM_SEED}")