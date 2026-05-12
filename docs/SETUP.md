# Setup notes

This file contains a few extra notes that did not fit naturally into `README.md` or `SOLUTION.md`.

## Reproducibility

* All randomness is seeded with `random_state = 42`:
  * `src/generate_data.py` — synthetic data generation.
  * `src/train_model.py` — train/test split and model.
* On a fresh checkout, running the three scripts in order should produce
  the same `data/student_progress_sample.csv`, the same metrics in
  `results/metrics.json` and the same plots.

## Tested environment

The pipeline has been validated with:

* macOS 25.x, Python 3.9.6
* numpy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1, matplotlib 3.9.4

Other recent versions of Python (3.10 – 3.12) and the listed libraries
should work as well — `requirements.txt` uses loose upper bounds.

## Running individual steps

| Step           | Command                              | Output                                                          |
| -------------- | ------------------------------------ | --------------------------------------------------------------- |
| Generate data  | `python src/generate_data.py`        | `data/student_progress_sample.csv`                              |
| Train model    | `python src/train_model.py`          | `results/model.pkl`, `metrics.json`, `confusion_matrix.png`, …  |
| Analyse data   | `python src/analyze_results.py`      | `results/eda_*.png`, `results/summary.md`                       |
| Notebook       | `jupyter notebook notebooks/...`     | Interactive walkthrough                                         |

## Cleaning generated artifacts

To re-run the pipeline from scratch:

```bash
rm -rf results/*.png results/*.csv results/*.json results/*.pkl results/*.txt results/summary.md
rm -f  data/student_progress_sample.csv
```

`data/` and `results/` are kept in the repository so that reviewers can
see the artifacts without running the pipeline themselves.
