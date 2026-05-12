# AI-Assisted Analysis of Student Learning Progress in Vocational IT Education

An application project prepared for **SMILES-2026**.

## Short description

A small, fully reproducible machine-learning pipeline that simulates how a teacher in a vocational IT education programme could use a baseline classifier to identify students who may need additional support. The project uses a synthetic dataset (no personal data), a Random Forest classifier and a focus on **interpretability** rather than predictive performance alone.

## Motivation

I work as an IT educator / lecturer of special disciplines at **GAPOU "YAKSIT"** (State Autonomous Professional Educational Institution "YAKSIT", Yakutsk, Russia). Among the vocational professional-cycle modules (*МДК*) I teach are **MDK 01.01** Operating systems, **MDK 01.02** Databases, and **MDK 01.04** Operation of automated systems in a protected (secure) configuration. Those courses combine systems administration, data handling and information security — exactly the kind of environment where attendance, practical hours and project work are strong early signals of progress.

I am interested in how ML and AI can responsibly support teachers in vocational IT education. This project is intentionally small: its goal is not to "predict students", but to show a clean methodology — synthetic data, a transparent baseline, explainable feature importances, and an honest discussion of limitations and ethics.

The full reasoning behind the project is in [`SOLUTION.md`](./SOLUTION.md).

## Repository structure

```
.
├── README.md                              # this file
├── SOLUTION.md                            # full application report
├── requirements.txt                       # Python dependencies
├── .gitignore
├── data/
│   └── student_progress_sample.csv        # synthetic dataset (generated)
├── src/
│   ├── generate_data.py                   # synthetic dataset generator
│   ├── train_model.py                     # baseline model training
│   └── analyze_results.py                 # EDA + post-training analysis
├── notebooks/
│   └── SMILES_2026_application_project.ipynb
├── results/                               # metrics, plots, model artifacts
└── docs/                                  # extra documentation (optional)
```

## Dataset

The dataset is **synthetic** and is produced by `src/generate_data.py`. It does **not** contain any personal data or any real student records.

Each row describes one (synthetic) vocational IT student with the following fields:

| Field                       | Description                                                    |
| --------------------------- | -------------------------------------------------------------- |
| `student_id`                | Synthetic identifier (e.g. `S0042`)                            |
| `attendance_rate`           | Share of classes attended, in `[0, 1]`                         |
| `assignment_score`          | Average score on assignments (0–100)                           |
| `quiz_score`                | Average quiz score (0–100)                                     |
| `project_score`             | Score on the practical course project (0–100)                  |
| `practice_hours`            | Hours spent on self-study practice                             |
| `ai_tool_usage_frequency`   | How often the student uses AI tools, on a 0–5 scale            |
| `previous_it_experience`    | Binary flag: prior IT background (1) or none (0)               |
| `final_result`              | Numeric final result on a 0–100 scale (target proxy)           |
| `risk_level`                | Categorical target: `low` / `medium` / `high`                  |

## Methodology

1. **Synthetic data generation** with controlled, plausible relationships between features and the target.
2. **Exploratory data analysis (EDA)** — distributions, class balance, correlations, group statistics.
3. **Modelling** — `RandomForestClassifier` with `class_weight="balanced"` and a stratified train/test split (75% / 25%).
4. **Evaluation** — accuracy, macro precision / recall / F1, per-class metrics, confusion matrix, with an explicit focus on **recall for the `high` class**.
5. **Interpretability** — feature importances are persisted as both a CSV and a plot.

## How to run

The project is plain Python — no Docker, no GPU.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python src/generate_data.py        # creates data/student_progress_sample.csv
python src/train_model.py          # trains the model, writes results/*
python src/analyze_results.py      # writes EDA plots and results/summary.md

# 4. (Optional) Explore the analysis interactively
jupyter notebook notebooks/SMILES_2026_application_project.ipynb
```

All outputs (metrics, plots, the trained model and a Markdown summary) are written to the `results/` directory.

## Results

On the synthetic dataset (600 students, stratified 75/25 split) the baseline Random Forest reaches approximately:

| Metric              | Value  |
| ------------------- | ------ |
| Accuracy            | ~0.83  |
| Precision (macro)   | ~0.86  |
| Recall (macro)      | ~0.79  |
| F1 (macro)          | ~0.82  |
| Recall on `high`    | ~0.67  |

The most influential features are `assignment_score`, `project_score` and `quiz_score`, followed by behavioural features such as `attendance_rate` and `practice_hours`.

Plots and the full numerical report are stored in [`results/`](./results) and discussed in [`SOLUTION.md`](./SOLUTION.md).

## Technologies used

* Python 3.9+
* NumPy, pandas
* scikit-learn
* matplotlib
* Jupyter Notebook

## Author

**Tatiana Tron** — IT educator / lecturer of special disciplines at **GAPOU "YAKSIT"**, Yakutsk, Russia. Teaching context includes vocational modules **MDK 01.01** (Operating systems), **MDK 01.02** (Databases), and **MDK 01.04** (Operation of automated systems in a protected configuration).

Areas of interest: Information Security, AI / Data Analytics, Digital Learning, LLM tools and applied vocational IT education.

---

> The model in this project is a **demonstration baseline**. Any real use of similar tools in education requires real, anonymised data, careful evaluation, transparency for students and teachers, and a clear human-in-the-loop process.
