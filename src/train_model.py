"""
Train a baseline classifier that predicts a student's risk_level.

The script reads the synthetic dataset produced by ``generate_data.py``,
trains a Random Forest classifier with a stratified train/test split,
evaluates it with standard classification metrics, and persists all
artifacts (model, metrics, confusion matrix plot, feature importances)
to the ``results/`` directory.

The choice of Random Forest is intentional for a baseline:
    * it handles mixed-scale numeric features without manual scaling;
    * it works well on small tabular datasets;
    * it exposes feature_importances_, which is useful for an
      explainability-first educational use case.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Allow running headless (CI / scripts).
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
TEST_SIZE = 0.25

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "student_progress_sample.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

FEATURE_COLUMNS = [
    "attendance_rate",
    "assignment_score",
    "quiz_score",
    "project_score",
    "practice_hours",
    "ai_tool_usage_frequency",
    "previous_it_experience",
]
TARGET_COLUMN = "risk_level"
CLASS_ORDER = ["low", "medium", "high"]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            f"Run `python src/generate_data.py` first."
        )
    return pd.read_csv(path)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute headline classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_high_risk": float(
            recall_score(
                y_true,
                y_pred,
                labels=["high"],
                average="macro",
                zero_division=0,
            )
        ),
    }


def plot_confusion_matrix(
    cm: np.ndarray, labels: list[str], output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix — risk_level")

    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="black" if cm[i, j] < cm.max() / 2 else "white",
                fontsize=11,
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(
    importances: pd.Series, output_path: Path
) -> None:
    importances = importances.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(importances.index, importances.values, color="#4C78A8")
    ax.set_xlabel("Importance")
    ax.set_title("Feature importance — Random Forest")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # class_weight="balanced" helps because "high" risk is the minority class
    # but also the class teachers care most about.
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = evaluate(y_test.values, y_pred)
    report_text = classification_report(
        y_test, y_pred, labels=CLASS_ORDER, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)

    importances = pd.Series(
        model.feature_importances_, index=FEATURE_COLUMNS, name="importance"
    )

    # Persist artifacts ------------------------------------------------------
    with open(RESULTS_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(
        RESULTS_DIR / "classification_report.txt", "w", encoding="utf-8"
    ) as f:
        f.write(report_text)

    importances.sort_values(ascending=False).to_csv(
        RESULTS_DIR / "feature_importance.csv", header=True
    )

    plot_confusion_matrix(
        cm, CLASS_ORDER, RESULTS_DIR / "confusion_matrix.png"
    )
    plot_feature_importance(
        importances, RESULTS_DIR / "feature_importance.png"
    )

    # Console summary --------------------------------------------------------
    print("Training complete.")
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k:18s}: {v:.3f}")
    print("\nClassification report:")
    print(report_text)
    print(f"\nArtifacts saved to {RESULTS_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
