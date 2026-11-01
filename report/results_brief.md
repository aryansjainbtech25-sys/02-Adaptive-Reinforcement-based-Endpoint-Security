# Results Brief — Single Source of Truth for Report
## ARES CIE3 — Numbers Teammates Must Not Deviate From
### Commit: 95292ce692f25092e2381aea08f094f2326c3956 (app.py full + Caveats A-C) → 30f90f5 (evidence) → e555928 (README)
### Evidence: results/final_test_run.txt 42/42 PASS

## 1. Calculus Experiment — results/calculus_experiment.json

| Field | Value | Notes |
|-------|-------|-------|
| review_threshold (optimized) | 0.6 | 60.00% |
| block_threshold (optimized) | 0.75 | 75.00% |
| review_threshold baseline | 0.35 | demo default |
| block_threshold baseline | 0.70 | demo default |
| final_cost_hard (at 0.60/0.75) | 0.387223 | true hard cost C_hard |
| final_cost (smoothed) | 0.45 | C_smooth surrogate |
| baseline_cost (0.35/0.70) | 0.387223 | hard cost at baseline |
| optimized_cost (0.60/0.75) | 0.3676 | hard cost at optimized |
| cost_delta | -0.0196 | optimized - baseline |
| cost_delta_pct | -5.1% | -5.1% |
| FPR baseline | 7.17% | at 0.35/0.70 |
| FPR optimized | 3.28% | -54% |
| FNR baseline | 5.81% | |
| FNR optimized | 6.37% | +0.56% |
| ReviewRate baseline | 3.65% | |
| ReviewRate optimized | 1.36% | -63% |
| n_test | 5083 | held-out URLs |
| n_test_domains | 1497 | unseen registrable domains |
| prevalence | 0.7478 | phishing |
| seed | 42 | build_dataset + GroupShuffleSplit |
| test_size | 0.20 | GroupShuffleSplit |
| tau (smoothing) | 0.05 | sigmoid temperature |
| learning_rate | 0.05 | GD eta |
| iterations | 29 | GD steps |
| history_length | 30 | init + 29 steps |
| C_FP | 1.0 | |
| C_FN | 5.0 | |
| C_review | 0.5 | |
| Bayes single t* | 0.1667 | C_FP/(C_FP+C_FN) |

### Derivative-Check at Returned Point (0.60,0.75)

| Field | Value |
|-------|-------|
| first_derivative_review | -0.16 |
| first_derivative_block | 1.62 |
| second_derivative_review | 0.48 |
| second_derivative_block | 19.3 |
| hessian_det | 6.9237 |
| hessian_trace | 19.78 |
| boundary_hit | ["review_box_0.60","gap_0.15"] |
| is_stationary | false |
| is_minimum | true (det>0, local convex) |

Interpretation: KKT constrained optimum, not interior stationary. Hessian det>0 proves minimum.

## 2. Baseline Metrics — results/baseline.json

| Field | Value |
|-------|-------|
| Model | XGBoost Calibrated (XGBClassifier + CalibratedClassifierCV 5-fold isotonic) |
| n_test | 5083 URLs |
| n_test_domains | 1497 unseen registrable domains |
| ROC-AUC | 0.9929 |
| PR-AUC | 0.9976 |
| Brier | 0.0297 |
| TPR / Recall @0.50 | 95.9% (0.959) |
| FPR @0.50 | 4.99% (0.0499) |
| F1 | 0.9707 |
| Precision | 98.27% |
| Prevalence | 0.7478 |
| Confusion TN | 1218 |
| Confusion FP | 64 |
| Confusion FN | 156 |
| Confusion TP | 3645 |
| Feature count | 24 fixed order |
| Bundle check | bundle features == feature_names True |
| sklearn | 1.9.1 |
| numpy | 2.5.4 |
| python | 3.14.2 |
| Groups | 7482 total (5985 train + 1497 test) |
| Raw data | 820 benign roots + 159244 phishing = 160064 rows |
| Built dataset | 22140 total (14760 phishing, 7380 benign) seed 42 |
| is_shortener | 510/22140 (2.30%) |
| tld_risk | 213/22140 (0.96%) |
| has_https randomization | p_strip=0.5 both classes |

### v2.0 RF for Comparison (previous)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9902 |
| PR-AUC | 0.9953 |
| Brier | 0.0353 |
| Recall | 96.57% |
| FPR | 7.24% |
| n_test | 4693 (1441 domains) |

## 3. Test Status — Single Sentence for Report

"42/42 tests pass (calculus_threshold 8, feature_extractor 14, hardening 14, qr 6); two prior failures in test_qr.py were resolved by a preload workaround (bundle validation + RGBA black-on-transparent composite); platform linux Python 3.14.2 pytest-9.2.0 23.61s — evidence results/final_test_run.txt; commit 95292ce692f25092e2381aea08f094f2326c3956 (full app.py Caveats A-C) → 30f90f5 (evidence pack) → e555928 (README 42/42)."

## 4. Commit Hash — Quote Exactly

- Final app.py fix: 95292ce692f25092e2381aea08f094f2326c3956
- Evidence pack: 30f90f5
- README update: e555928
- Log: results/final_commits.txt (20 commits)

## 5. No Deviation Rule

Nothing teammates write may deviate from this table. If a number not in table needed, read directly from results/calculus_experiment.json or results/baseline.json and add to this brief first.

---
End of Results Brief — Source of Truth
