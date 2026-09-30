# Student Grade Prediction

## Project Overview

Predict student final grades (A, B, C, D, F) using study habits and attendance data.

## Features Used

- weekly_self_study_hours
- attendance_percentage
- class_participation

## Excluded Columns

- student_id (identifier, not predictive)
- total_score (target leakage - grade is derived from it)

## Pipeline Stages

1. **prepare.py** - Clean and sample the raw dataset
2. **split.py** - Create stratified train/test splits
3. **train.py** - Train RandomForestClassifier model
4. **evaluate.py** - Evaluate model on test data

## Usage

```bash
python prepare.py
python split.py
python train.py
python evaluate.py
```

## Results

- Train/Test split: 80/20 stratified
- Model: RandomForestClassifier with hyperparameter tuning
- Test Accuracy: 69.76%
- Metrics saved to: reports/metrics.json