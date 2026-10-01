# 🎓 Student Grade Prediction — Production ML Pipeline

> **An end-to-end, reproducible Machine Learning system for predicting student grades using DVC, Feast, and production-ready ML practices.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-purple?logo=dvc)](https://dvc.org/)
[![Feast](https://img.shields.io/badge/Feast-Feature%20Store-orange)](https://feast.dev/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)](https://www.docker.com/)

---

## 📌 Overview

**Student Grade Prediction** is an end-to-end Machine Learning project designed to demonstrate how a predictive model can be transformed into a **reproducible and production-oriented ML system**.

The system predicts a student's final grade:

> **A / B / C / D / F**

using three behavioral and academic features:

* 📚 Weekly self-study hours
* 🏫 Attendance percentage
* 🙋 Class participation

The project focuses not only on model training, but also on **data versioning, reproducibility, feature management, evaluation, and deployment**.

---

## 🎯 Project Objectives

The main objectives are to:

* Build a multiclass student-grade prediction model
* Create a reproducible ML pipeline
* Version datasets and ML artifacts with **DVC**
* Manage ML features using **Feast**
* Separate data processing, training, and evaluation
* Make experiments reproducible
* Prepare the model for API-based inference
* Containerize the application using Docker
* Follow production-oriented ML engineering practices

---

## 🧠 Problem Statement

Educational institutions collect various student-related information that can potentially help identify academic performance patterns.

This project explores whether a student's:

* study time,
* attendance, and
* class participation

can be used to predict their final grade category.

### Target

```text
A
B
C
D
F
```

### Input Features

| Feature                   | Description                                         |
| ------------------------- | --------------------------------------------------- |
| `weekly_self_study_hours` | Average hours spent studying independently per week |
| `attendance_percentage`   | Student attendance percentage                       |
| `class_participation`     | Level of participation in class                     |

### Features intentionally excluded

`student_id` and `total_score` are not used for prediction.

* `student_id` is an identifier rather than a meaningful predictive feature.
* `total_score` can introduce **target leakage**, since it is directly related to the student's final grade.

This keeps the prediction setup closer to a realistic inference scenario.

---

# 🏗️ System Architecture

```text
                 ┌─────────────────────┐
                 │   Raw Dataset       │
                 │    ~1,000,000 rows   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Data Sampling     │
                 │ Stratified ~50K     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Preprocessing     │
                 │ Cleaning / Encoding │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Feast Feature     │
                 │       Store         │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Model Training    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    Evaluation       │
                 │ Metrics / Reports   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Trained Model       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │     FastAPI         │
                 │   Prediction API    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │      Docker         │
                 │ Containerized App   │
                 └─────────────────────┘
```

---

# 🔄 DVC Pipeline

The project uses **DVC (Data Version Control)** to make the ML workflow reproducible.

```text
Raw Data
   │
   ▼
Preprocessing
   │
   ▼
Feature Engineering
   │
   ▼
Model Training
   │
   ▼
Evaluation
```

Each stage is represented as a reproducible pipeline step.

### Pipeline stages

| Stage                 | Purpose                       |
| --------------------- | ----------------------------- |
| `preprocess`          | Clean and prepare the dataset |
| `feature_engineering` | Prepare model/Feast features  |
| `train`               | Train the ML model            |
| `evaluate`            | Generate evaluation metrics   |

The pipeline can be reproduced using:

```bash
dvc repro
```

This allows the entire ML workflow to be rebuilt from version-controlled data and code.

---

# 📦 Dataset

The original synthetic dataset contains approximately:

```text
1,000,000 rows
```

To make experimentation more practical while preserving the target distribution, approximately:

```text
50,000 rows
```

are selected using **reproducible stratified sampling**.

### Original schema

```text
student_id
weekly_self_study_hours
attendance_percentage
class_participation
total_score
grade
```

### Modeling schema

```text
weekly_self_study_hours
attendance_percentage
class_participation
grade
```

The sampling process uses a fixed random seed so that the same dataset can be reproduced.

---

# 🧪 Machine Learning

The task is a **multiclass classification problem**.

### Input

```text
Study Hours
Attendance
Class Participation
```

### Output

```text
A / B / C / D / F
```

The model is trained using the processed feature dataset and evaluated using appropriate multiclass classification metrics.

### Evaluation

The project evaluates model performance using metrics such as:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

The evaluation stage produces reproducible metrics that can be tracked alongside the corresponding dataset and model version.

---

# 🗃️ Feature Store — Feast

**Feast** is used as the feature-store component of the project.

The feature store provides a consistent definition of model features between:

```text
Training
    ↓
Validation
    ↓
Inference
```

This helps reduce the risk of **training-serving skew**, where the features used during model training differ from those used during production inference.

Example feature set:

```text
weekly_self_study_hours
attendance_percentage
class_participation
```

---

# 🚀 API

The trained model can be exposed through a **FastAPI** service.

Example prediction request:

```json
{
  "weekly_self_study_hours": 15,
  "attendance_percentage": 92,
  "class_participation": 8
}
```

Example response:

```json
{
  "predicted_grade": "A"
}
```

The API provides a simple interface for integrating the trained model with other applications.

---

# 🐳 Docker

The application is designed to run inside a Docker container.

This provides:

* Reproducible runtime environments
* Dependency isolation
* Easier deployment
* Consistent development and production environments

Build the image:

```bash
docker build -t student-grade-prediction .
```

Run the container:

```bash
docker run -p 8000:8000 student-grade-prediction
```

The API can then be accessed through:

```text
http://localhost:8000
```

---

# 📁 Project Structure

```text
student-grade-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── student_grade_prediction.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── feature_repo/
│   ├── feature_store.yaml
│   ├── entities.py
│   ├── features.py
│   └── feature_views.py
│
├── models/
│   └── model.pkl
│
├── reports/
│   ├── metrics.json
│   └── confusion_matrix.png
│
├── api/
│   └── main.py
│
├── tests/
│   └── test_api.py
│
├── dvc.yaml
├── dvc.lock
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd student-grade-prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🔁 Reproduce the Pipeline

Initialize DVC if required:

```bash
dvc init
```

Reproduce the complete ML pipeline:

```bash
dvc repro
```

Check pipeline status:

```bash
dvc status
```

View pipeline:

```bash
dvc dag
```

---

# 🧪 Testing

Tests are written using **pytest**.

Run:

```bash
pytest
```

For more detailed output:

```bash
pytest -v
```

---

# 📊 Experiment Reproducibility

One of the key goals of this project is reproducibility.

The combination of:

```text
Git
+
DVC
+
Fixed Random Seeds
+
Versioned Data
+
Versioned Code
+
Tracked Parameters
```

makes it possible to reproduce previous experiments and understand exactly how a model was generated.

---

# 🔍 Why DVC?

Traditional Git repositories are not designed for large datasets and ML artifacts.

Instead of storing large datasets directly in Git, DVC allows us to version them separately while keeping lightweight metadata in the repository.

```text
Git
 ├── Code
 ├── Configuration
 └── DVC metadata

DVC
 ├── Dataset versions
 ├── Model artifacts
 └── Pipeline dependencies
```

This makes the ML project easier to reproduce and collaborate on.

---

# 🛠️ Technology Stack

| Category            | Technology           |
| ------------------- | -------------------- |
| Language            | Python               |
| Data Processing     | Pandas, NumPy        |
| Machine Learning    | Scikit-learn         |
| Data Versioning     | DVC                  |
| Feature Store       | Feast                |
| API                 | FastAPI              |
| Testing             | Pytest               |
| Containerization    | Docker               |
| Version Control     | Git & GitHub         |
| Experiment Tracking | DVC                  |
| Visualization       | Matplotlib / Seaborn |

---

# 📈 Future Improvements

Possible extensions include:

* Model comparison and hyperparameter tuning
* Automated model validation
* CI/CD with GitHub Actions
* Model monitoring
* Data drift detection
* Model performance monitoring
* Cloud deployment
* Automated retraining
* Production-grade feature serving

---

# 👨‍💻 Author

**Jisan**

BSc in Data Science
Department of Computer Science & Engineering

---

## ⭐ Project Goal

> **The goal is not simply to train a model. The goal is to build a reproducible ML system around the model.**

From:

```text
Data
 ↓
Versioning
 ↓
Preprocessing
 ↓
Features
 ↓
Training
 ↓
Evaluation
 ↓
API
 ↓
Deployment
```

This project demonstrates the transition from **Machine Learning experimentation → Machine Learning Engineering**.

---

## 📜 License

This project is intended for educational and academic purposes.
