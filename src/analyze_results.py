"""
Post-training analysis of the synthetic student-progress dataset.

The script complements ``train_model.py`` by producing several
exploratory visualisations and a short, plain-text summary that can be
shared with non-technical readers (e.g. teachers or reviewers).

Outputs in ``results/``:
    * eda_distributions.png        — histograms of key numeric features
    * eda_risk_breakdown.png       — mean score per risk_level
    * eda_correlation_heatmap.png  — Pearson correlations between features
    * summary.md                   — Markdown summary of the analysis
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "student_progress_sample.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

NUMERIC_FEATURES = [
    "attendance_rate",
    "assignment_score",
    "quiz_score",
    "project_score",
    "practice_hours",
    "ai_tool_usage_frequency",
    "previous_it_experience",
    "final_result",
]
RISK_ORDER = ["low", "medium", "high"]


def plot_distributions(df: pd.DataFrame, output_path: Path) -> None:
    cols = [c for c in NUMERIC_FEATURES if c != "previous_it_experience"]
    n_cols = 3
    n_rows = int(np.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3.2 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        axes[i].hist(df[col], bins=25, color="#4C78A8", edgecolor="white")
        axes[i].set_title(col)
        axes[i].grid(alpha=0.2)
    for j in range(len(cols), len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle("Feature distributions", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_risk_breakdown(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    score_cols = [
        "attendance_rate",
        "assignment_score",
        "quiz_score",
        "project_score",
        "practice_hours",
    ]
    grouped = (
        df.groupby("risk_level")[score_cols]
        .mean()
        .reindex(RISK_ORDER)
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(score_cols))
    width = 0.25
    colors = {"low": "#3CB371", "medium": "#F4A261", "high": "#E76F51"}
    for i, risk in enumerate(RISK_ORDER):
        ax.bar(
            x + (i - 1) * width,
            grouped.loc[risk].values,
            width,
            label=risk,
            color=colors[risk],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(score_cols, rotation=20, ha="right")
    ax.set_ylabel("Mean value")
    ax.set_title("Average feature value per risk level")
    ax.legend(title="risk_level")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return grouped


def plot_correlation(df: pd.DataFrame, output_path: Path) -> None:
    corr = df[NUMERIC_FEATURES].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(NUMERIC_FEATURES)))
    ax.set_yticks(range(len(NUMERIC_FEATURES)))
    ax.set_xticklabels(NUMERIC_FEATURES, rotation=45, ha="right")
    ax.set_yticklabels(NUMERIC_FEATURES)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax)
    ax.set_title("Feature correlation (Pearson)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_summary(
    df: pd.DataFrame,
    grouped: pd.DataFrame,
    output_path: Path,
) -> None:
    metrics_path = RESULTS_DIR / "metrics.json"
    importance_path = RESULTS_DIR / "feature_importance.csv"

    metrics_block = ""
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        metrics_block = "\n".join(
            f"- **{k}**: {v:.3f}" for k, v in metrics.items()
        )

    importance_block = ""
    if importance_path.exists():
        importances = pd.read_csv(importance_path, index_col=0)
        importance_block = importances.head(5).to_markdown()

    risk_counts = df["risk_level"].value_counts().reindex(RISK_ORDER).fillna(0)

    lines = [
        "# Analysis Summary",
        "",
        "This file is generated automatically by `src/analyze_results.py` "
        "and summarises the synthetic dataset and the baseline model's "
        "behaviour.",
        "",
        "## Dataset overview",
        "",
        f"- Total students: **{len(df)}**",
        f"- Features used by the model: **{len(NUMERIC_FEATURES) - 1}** "
        "(excluding `final_result`, which leaks the target).",
        "",
        "### Risk level distribution",
        "",
        risk_counts.astype(int).to_markdown(),
        "",
        "### Mean feature value per risk level",
        "",
        grouped.round(2).to_markdown(),
        "",
        "## Baseline model metrics",
        "",
        metrics_block or "_Run `python src/train_model.py` first._",
        "",
        "## Top-5 most influential features",
        "",
        importance_block or "_Run `python src/train_model.py` first._",
        "",
        "## Reading guide",
        "",
        "- Higher `attendance_rate`, `project_score` and `practice_hours` "
        "are associated with **lower** predicted risk.",
        "- `ai_tool_usage_frequency` is included as an exploratory signal; "
        "its effect on risk should not be over-interpreted on synthetic data.",
        "- The model is a **demonstration baseline**. It is meant to "
        "illustrate methodology, not to be used in production.",
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            f"Run `python src/generate_data.py` first."
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    plot_distributions(df, RESULTS_DIR / "eda_distributions.png")
    grouped = plot_risk_breakdown(
        df, RESULTS_DIR / "eda_risk_breakdown.png"
    )
    plot_correlation(df, RESULTS_DIR / "eda_correlation_heatmap.png")
    write_summary(df, grouped, RESULTS_DIR / "summary.md")

    print("Analysis complete. Generated files:")
    for name in [
        "eda_distributions.png",
        "eda_risk_breakdown.png",
        "eda_correlation_heatmap.png",
        "summary.md",
    ]:
        print(f"  results/{name}")


if __name__ == "__main__":
    main()
