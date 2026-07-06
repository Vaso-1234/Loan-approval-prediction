import pandas as pd
import numpy as np
import joblib
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =========================
# FILE PATHS
# =========================
DATA_PATH = "loan_data.csv"
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
REPORT_PATH = "evaluation_report.json"


# =========================
# LOAD DATA
# =========================
def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# =========================
# PREPROCESS
# =========================
def preprocess_data(df):
    print("\nBefore preprocessing:", df.shape)

    df = df.drop_duplicates()
    df = df.dropna()

    # keep only valid ranges
    df = df[
        (df["Age"] >= 18) & (df["Age"] <= 60) &
        (df["Income"] >= 8000) & (df["Income"] <= 300000) &
        (df["LoanAmount"] >= 500) & (df["LoanAmount"] <= 35000) &
        (df["CreditScore"] >= 390) & (df["CreditScore"] <= 850)
    ].copy()

    print("After preprocessing:", df.shape)
    return df


# =========================
# SPLIT + SCALE
# =========================
def split_data(df):
    X = df[["Age", "Income", "LoanAmount", "CreditScore"]]
    y = df["Approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


# =========================
# TRAIN MODEL
# =========================
def train_model(X_train_scaled, y_train):
    rf = RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    param_grid = {
        "n_estimators": [150, 200],
        "max_depth": [6, 8, 10, 12],
        "min_samples_split": [5, 10, 20],
        "min_samples_leaf": [2, 5, 10]
    }

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train_scaled, y_train)

    print("\nBest Parameters Found:")
    print(grid.best_params_)
    print(f"Best Cross-Validation F1 Score: {grid.best_score_:.4f}")

    return grid.best_estimator_, grid.best_params_, grid.best_score_


# =========================
# EVALUATE MODEL
# =========================
def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\n===== MODEL EVALUATION =====")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }

    return metrics


# =========================
# SAVE ARTIFACTS
# =========================
def save_artifacts(model, scaler, metrics):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    with open(REPORT_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nModel saved as: {MODEL_PATH}")
    print(f"Scaler saved as: {SCALER_PATH}")
    print(f"Evaluation report saved as: {REPORT_PATH}")


# =========================
# SANITY TESTS
# =========================
def run_sanity_tests(model, scaler):
    print("\n===== SANITY TESTS =====")

    test_cases = {
        "APPROVE SAMPLE": pd.DataFrame([{
            "Age": 25,
            "Income": 50000,
            "LoanAmount": 20000,
            "CreditScore": 650
        }]),

        "REJECT SAMPLE": pd.DataFrame([{
            "Age": 25,
            "Income": 9000,
            "LoanAmount": 20000,
            "CreditScore": 650
        }]),

        "LOW CREDIT SAMPLE": pd.DataFrame([{
            "Age": 25,
            "Income": 50000,
            "LoanAmount": 10000,
            "CreditScore": 400
        }])
    }

    for name, sample in test_cases.items():
        sample_scaled = scaler.transform(sample)
        pred = model.predict(sample_scaled)[0]
        prob = model.predict_proba(sample_scaled)[0]

        print(f"\n{name}")
        print("Prediction:", pred)
        print("Probabilities:", prob)


# =========================
# MAIN
# =========================
def main():
    df = load_data()
    df = preprocess_data(df)

    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = split_data(df)

    model, best_params, best_cv_score = train_model(X_train_scaled, y_train)
    metrics = evaluate_model(model, X_test_scaled, y_test)

    save_artifacts(model, scaler, metrics)
    run_sanity_tests(model, scaler)


if __name__ == "__main__":
    main()