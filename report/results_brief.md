# Results Brief — Single Source of Truth for Report
## ARES CIE3 — Numbers Teammates Must Not Deviate From
### Canonical Source: results/calculus_experiment.json — If any number in this brief disagrees with that file, the file wins.
### Commit: 95292ce692f25092e2381aea08f094f2326c3956 (app.py full + Caveats A-C) → 30f90f5 (evidence) → e555928 (README)
### Evidence: results/final_test_run.txt 42/42 PASS
### FIXED 2026-11-01: Previous brief said baseline 0.387223 optimized 0.3676 delta -5.1% — WRONG. Correct per JSON: baseline 0.39987259019990573 → optimized 0.387223 delta -0.01264959019990573 (-3.16%)

## 1. Calculus Experiment — results/calculus_experiment.json (CANONICAL — EXACT VALUES)

| Field | Value | JSON Path |
|-------|-------|-----------|
| baseline.cost (0.35/0.70) | 0.39987259019990573 | baseline.cost |
| optimized.optimal_cost (0.60/0.75) | 0.387223 | optimized.optimal_cost |
| cost_delta | -0.01264959019990573 | optimized - baseline |
| cost_delta_pct | -3.16% | -0.01264959/0.39987259*100 |
| baseline thresholds | review 0.35 block 0.70 | baseline.thresholds |
| optimized thresholds | review 0.60 block 0.75 | optimized.thresholds |
| baseline FPR | 0.06942277691107644 (6.94%) | baseline.rates.fpr |
| baseline FNR | 0.062352012628255724 (6.24%) | baseline.rates.fnr |
| baseline ReviewRate | 0.03737950029510132 (3.74%) | baseline.rates.review_rate |
| optimized FPR | 0.0343 (3.43%) | optimized.rates_at_optimum.fpr |
| optimized FNR | 0.0681 (6.81%) | optimized.rates_at_optimum.fnr |
| optimized ReviewRate | 0.0179 (1.79%) | optimized.rates_at_optimum.review_rate |
| FPR delta | -0.03512277691107644 (-50.6%) | 0.0343-0.0694 |
| FNR delta | +0.005747987371744276 (+9.2%) | FNR increased |
| Review delta | -0.01947950029510132 (-52.1%) | 0.0179-0.0373 |
| n_test | 5083 | held-out URLs |
| n_test_domains | 1497 | unseen registrable domains |
| prevalence | 0.7478 | phishing |
| seed | 42 | build_dataset + GroupShuffleSplit |
| test_size | 0.20 | GroupShuffleSplit |
| tau | 0.05 | sigmoid temperature |
| learning_rate | 0.05 | GD eta |
| iterations | 29 | GD steps |
| history_length | 30 | init + 29 steps |
| C_FP | 1.0 | |
| C_FN | 5.0 | |
| C_review | 0.5 | |

### Derivative-Check at Returned Point (0.60,0.75) — CANONICAL EXACT

| Field | Value | JSON Path |
|-------|-------|-----------|
| first_derivative_review | -0.153059 | optimized.derivative_check.first_derivative_review |
| first_derivative_block | 1.621676 | optimized.derivative_check.first_derivative_block |
| second_derivative_review | 0.405678 | optimized.derivative_check.second_derivative_review |
| second_derivative_block | 17.086192 | optimized.derivative_check.second_derivative_block |
| hessian_det | 6.923728 | optimized.derivative_check.hessian_det |
| boundary_hit | ['review_box_0.60', 'gap_0.15'] | optimized.derivative_check.boundary_hit |
| gradient_descent_iterations | 29 | optimized.derivative_check.gradient_descent_iterations |
| is_stationary | false | optimized.derivative_check.is_stationary |
| is_minimum | true | hessian_det>0 |

Interpretation: KKT constrained optimum, not interior stationary. Hessian det 6.923728>0 proves local minimum. Boundary active.

## 2. Baseline Metrics — results/baseline.json (CANONICAL)

| Field | Value |
|-------|-------|
| Model | XGBoost Calibrated |
| n_test | 5083 |
| n_test_domains | 1497 |
| ROC-AUC | 0.9929 |
| PR-AUC | 0.9976 |
| Brier | 0.0297 |
| TPR | 95.9% |
| FPR @0.50 | 4.99% |
| F1 | 0.9707 |
| Precision | 98.27% |
| TN | 1218 |
| FP | 64 |
| FN | 156 |
| TP | 3645 |

## 3. Test Status — Single Sentence for Report

"42/42 tests pass (calculus_threshold 8, feature_extractor 14, hardening 14, qr 6); two prior failures in test_qr.py were resolved by a preload workaround; platform linux Python 3.14.2 pytest-9.2.0 22.44s — evidence results/final_test_run.txt; commit 95292ce692f25092e2381aea08f094f2326c3956 → 30f90f5. Canonical JSON: baseline.cost 0.39987259019990573 → optimized.optimal_cost 0.387223, delta -0.01264959019990573 (-3.16%), FPR 6.94%→3.43% (-50.6%), FNR 6.24%→6.81% (+9.2%), Review 3.74%→1.79% (-52.1%), hessian_det 6.923728, boundary_hit review_box_0.60+gap_0.15."

## 4. Commit Hash

- Final app.py: 95292ce692f25092e2381aea08f094f2326c3956
- Evidence: 30f90f5
- README: e555928

## 5. No Deviation Rule

If any number disagrees with JSON, JSON wins. Previous error: brief said 0.387223/0.3676 -5.1% — WRONG. Correct: 0.39987259019990573 → 0.387223 -3.16% per JSON exact.

---
End of Results Brief — FIXED to JSON EXACT

