# modelling.py (FINAL REVISI)
import argparse
import sys
import warnings
from pathlib import Path
from datetime import datetime

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Non-interaktif agar jalan di GitHub Actions
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")


def setup_mlflow():
    """Setup MLflow lokal"""
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("diabetes-basic")
    print("MLflow Tracking URI set to local (./mlruns)")


def train(data_path: Path, min_accuracy: float = 0.70):
    """Train Logistic Regression dengan MLflow autolog"""
    print("Starting training job for Diabetes Prediction")
    print("=" * 60)

    # Load data
    if not data_path.exists():
        raise FileNotFoundError(f" Dataset tidak ditemukan: {data_path}")

    df = pd.read_csv(data_path)
    if "Outcome" not in df.columns:
        raise ValueError(" Kolom target 'Outcome' tidak ditemukan di dataset")

    X = df.drop(columns=["Outcome"]).values
    y = df["Outcome"].astype(int).values

    print(f"Data shape: {df.shape}, Features: {X.shape[1]}")
    print(f"Target distribution: {np.bincount(y)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f" Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Setup MLflow
    setup_mlflow()

    # Aktifkan autolog MLflow
    mlflow.sklearn.autolog(
        log_model_signatures=True,
        log_input_examples=True,
        log_models=True,
        disable=False,
        exclusive=False,
        silent=True
    )

    # Mulai training
    run_name = f"logreg_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        print(f"🔄 Running MLflow experiment: {run_name}")

        # Train model
        model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)
        model.fit(X_train, y_train)

        # Evaluasi sederhana
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Model trained successfully with accuracy: {acc:.4f}")

        # Validasi performa
        if acc >= min_accuracy:
            print(f"Model passed validation ({acc:.4f} >= {min_accuracy})")
        else:
            print(f"Model did NOT meet minimum accuracy ({acc:.4f} < {min_accuracy})")

    print("=" * 60)
    print("Training selesai, artefak tersimpan di folder ./mlruns")
    print("Jalankan: `mlflow ui` untuk melihat hasil tracking.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train diabetes prediction model")
    parser.add_argument(
        "--data",
        type=str,
        default="../dataset_preprocessing/diabetes_preprocessed.csv",
        help="Path ke file CSV hasil preprocessing"
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.70,
        help="Nilai akurasi minimum untuk validasi model"
    )
    args = parser.parse_args()

    try:
        train(Path(args.data), args.min_accuracy)
        sys.exit(0)
    except Exception as e:
        print(f"Training gagal: {e}")
        sys.exit(1)
