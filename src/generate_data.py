"""
Synthetic dataset generator for the SMILES-2026 application project.

The script produces a CSV file describing student progress in a vocational
IT education program. The data is fully synthetic and contains no personal
information. The relationships between features and the target variables
are designed to be plausible but not deterministic, so that a baseline ML
model can find a useful signal while still leaving room for noise.

Output:
    data/student_progress_sample.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_STUDENTS = 600

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "student_progress_sample.csv"


def _clip01(values: np.ndarray) -> np.ndarray:
    """Clip an array to the [0, 1] range."""
    return np.clip(values, 0.0, 1.0)


def generate_students(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate a synthetic student progress dataset.

    Features are sampled from simple, interpretable distributions and
    combined into a latent "academic strength" score that drives the
    final_result and risk_level targets.
    """
    student_id = np.array([f"S{idx:04d}" for idx in range(1, n + 1)])

    # Behavioural features
    attendance_rate = _clip01(rng.normal(loc=0.82, scale=0.12, size=n))
    practice_hours = np.clip(rng.normal(loc=45, scale=15, size=n), 5, 120)

    # Background feature (binary: previous IT experience yes/no)
    previous_it_experience = rng.binomial(n=1, p=0.4, size=n)

    # AI tool usage frequency on a 0..5 scale
    # (0 = never, 5 = uses AI tools almost daily during study)
    ai_tool_usage_frequency = rng.integers(low=0, high=6, size=n)

    # Academic performance: tied to attendance, practice and prior experience,
    # with independent gaussian noise so that scores are correlated but not
    # collinear.
    # An additive baseline (0.25) shifts the mean academic strength upward
    # so that the resulting class distribution resembles a realistic
    # vocational programme: most students pass, a smaller share is at
    # medium risk, and only a minority is at high risk.
    base_skill = (
        0.25
        + 0.40 * attendance_rate
        + 0.20 * (practice_hours / 120.0)
        + 0.10 * previous_it_experience
        + 0.05 * (ai_tool_usage_frequency / 5.0)
    )

    assignment_score = _clip01(base_skill + rng.normal(0, 0.10, size=n)) * 100
    quiz_score = _clip01(base_skill + rng.normal(0, 0.12, size=n)) * 100
    project_score = _clip01(base_skill + rng.normal(0, 0.09, size=n)) * 100

    # Final numeric result: weighted average + small noise.
    final_result = (
        0.35 * assignment_score
        + 0.25 * quiz_score
        + 0.40 * project_score
        + rng.normal(0, 3, size=n)
    )
    final_result = np.clip(final_result, 0, 100).round(1)

    # Risk level thresholds are chosen so that all three classes are
    # represented but the data set remains imbalanced toward "low" risk,
    # which matches real educational settings.
    # Thresholds are chosen so that "low" risk is the majority class
    # (matching real vocational programmes), "medium" is a meaningful
    # minority that teachers should monitor, and "high" is a small but
    # non-trivial group that needs early intervention.
    risk_level = np.where(
        final_result < 60, "high",
        np.where(final_result < 72, "medium", "low"),
    )

    df = pd.DataFrame(
        {
            "student_id": student_id,
            "attendance_rate": attendance_rate.round(3),
            "assignment_score": assignment_score.round(1),
            "quiz_score": quiz_score.round(1),
            "project_score": project_score.round(1),
            "practice_hours": practice_hours.round(1),
            "ai_tool_usage_frequency": ai_tool_usage_frequency,
            "previous_it_experience": previous_it_experience,
            "final_result": final_result,
            "risk_level": risk_level,
        }
    )
    return df


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = generate_students(N_STUDENTS, rng)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Generated synthetic dataset: {OUTPUT_PATH}")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    print("\nClass distribution (risk_level):")
    print(df["risk_level"].value_counts().to_string())
    print("\nFirst rows:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
