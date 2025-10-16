# modelling.py (FINAL for Kriteria 3 - Basic)
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

import matplotlib
matplotlib.use("Agg")  # Non-GUI backend for CI/CD

warnings.filterwarnings("ignore")


def setup_mlflow():
    """Setup MLflow tracking environment"""
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("diabetes-basic")
    print("✅ MLflow Tracking set to local (./mlruns)")


def train(data_path: Path, min_accuracy: float = 0.70):
    """Train logistic regression model and log automatically using MLflow autolog"""
    print("🚀 Starting Diabetes Prediction Model Training")
    print("=" * 60)

    if not data_path.exists():
        raise FileNotFoundError(f"❌ Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    if "Outcome" not in df.columns:
        raise ValueError("❌ Target column 'Outcome' not found in dataset")

    X = df.drop(columns=["Outcome"]).values
    y = df["Outcome"].astype(int).values

    print(f"📊 Dataset shape: {df.shape}")
    print(f"📈 Features: {X.shape[1]}, Samples: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"📋 Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # Setup MLflow
    setup_mlflow()

    # Enable autolog
    mlflow.sklearn.autolog(
        log_model_signatures=True,
        log_input_examples=True,
        log_models=True,
        disable=False,
        exclusive=False,
        silent=True
    )

    run_name = f"logreg_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        print(f"🔄 Running MLflow experiment: {run_name}")

        model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Model trained successfully with accuracy: {acc:.4f}")

        if acc >= min_accuracy:
            print(f"🎯 Model passed validation ({acc:.4f} >= {min_accuracy})")
        else:
            print(f"⚠️ Model failed validation ({acc:.4f} < {min_accuracy})")

    print("=" * 60)
    print("🎉 Training complete. Check ./mlruns for MLflow artifacts.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train diabetes prediction model")
    parser.add_argument(
        "--data",
        type=str,
        default="../dataset_preprocessing/diabetes_preprocessed.csv",
        help="Path to the preprocessed CSV file"
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.70,
        help="Minimum accuracy threshold for validation"
    )

    args = parser.parse_args()
    try:
        train(Path(args.data), args.min_accuracy)
        sys.exit(0)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)
