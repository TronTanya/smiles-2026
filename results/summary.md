# Analysis Summary

This file is generated automatically by `src/analyze_results.py` and summarises the synthetic dataset and the baseline model's behaviour.

## Dataset overview

- Total students: **600**
- Features used by the model: **7** (excluding `final_result`, which leaks the target).

### Risk level distribution

| risk_level   |   count |
|:-------------|--------:|
| low          |     287 |
| medium       |     252 |
| high         |      61 |

### Mean feature value per risk level

| risk_level   |   attendance_rate |   assignment_score |   quiz_score |   project_score |   practice_hours |
|:-------------|------------------:|-------------------:|-------------:|----------------:|-----------------:|
| low          |              0.85 |              79.3  |        78.8  |           78.08 |            47.74 |
| medium       |              0.8  |              67.24 |        67.5  |           67.46 |            42.9  |
| high         |              0.72 |              55.59 |        55.95 |           57.81 |            38.43 |

## Baseline model metrics

- **accuracy**: 0.833
- **precision_macro**: 0.858
- **recall_macro**: 0.792
- **f1_macro**: 0.816
- **recall_high_risk**: 0.667

## Top-5 most influential features

|                  |   importance |
|:-----------------|-------------:|
| assignment_score |    0.270375  |
| project_score    |    0.256361  |
| quiz_score       |    0.204662  |
| attendance_rate  |    0.10623   |
| practice_hours   |    0.0740885 |

## Reading guide

- Higher `attendance_rate`, `project_score` and `practice_hours` are associated with **lower** predicted risk.
- `ai_tool_usage_frequency` is included as an exploratory signal; its effect on risk should not be over-interpreted on synthetic data.
- The model is a **demonstration baseline**. It is meant to illustrate methodology, not to be used in production.
