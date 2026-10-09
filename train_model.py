"""Step 2 - Train the classifier on the collected landmarks.

Usage:
    python train_model.py

Outputs:
    model/sign_model.pkl        trained model
    model/confusion_matrix.png  evaluation figure (use it in your report/PPT)
"""
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report)
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join("data", "landmarks.csv")
MODEL_DIR = "model"


def main():
    df = pd.read_csv(DATA_PATH, dtype={"label": str})
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    print(f"Samples: {len(df)} | Classes: {sorted(set(y))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "sign_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
    ax.set_title(f"Confusion matrix (accuracy {acc * 100:.1f}%)")
    fig.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
    print("Model and confusion matrix saved in ./model")


if __name__ == "__main__":
    main()
