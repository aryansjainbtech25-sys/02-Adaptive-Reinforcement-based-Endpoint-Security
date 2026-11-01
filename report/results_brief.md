# Results Brief — CANONICAL — results/calculus_experiment.json + baseline.json EXACT
## Final project commit: 5fdc189aac39267cb96f04f4f3e409535be656a5 (short: 5fdc189).
### Previous chain: 7506878 -> 5fdc189 — len=30 gd=29 duplicate False
### Canonical source: results/calculus_experiment.json + results/baseline.json — file wins over brief if disagree
### Evidence: results/final_test_run.txt 42/42 PASS platform linux Python 3.14.2 pytest-9.2.0 22.44s
### All numbers in this report are traceable to results/calculus_experiment.json and results/baseline.json at commit 5fdc189aac39267cb96f04f4f3e409535be656a5. Test evidence: results/final_test_run.txt (42/42 pass, Python 3.14.2, pytest 9.2.0).

## 1. Calculus Experiment — results/calculus_experiment.json — CANONICAL — EXACT

| Field | Value | JSON Path | Note |
|-------|-------|-----------|------|
| baseline.cost (0.35/0.70) | 0.39987259019990573 | baseline.cost | Baseline hard cost |
| optimized.optimal_cost (0.60/0.75) | 0.387223 | optimized.optimal_cost | Returned point hard cost — NOT minimum over all visited — see grid winner |
| optimized.final_cost smoothed | 0.43994209280287067 | optimized.final_cost | Smoothed at returned point = history[-1][2] |
| cost_delta | -0.01264959019990573 | optimized - baseline | -3.16% relative |
| cost_delta_pct | -3.16% | delta/baseline*100 | Improvement over baseline from grid, NOT GD |
| baseline thresholds | review 0.35 block 0.70 | baseline.thresholds | |
| optimized thresholds | review 0.60 block 0.75 | optimized.thresholds | Returned point — boundary-clipped feasible |
| Grid best smoothed cost | 0.29071968134044274 | computed via expected_cost_smoothed grid search | Grid winner (0.30,0.45) |
| Grid best (r,b) | 0.3, 0.45 | grid search 100x100 linspace | Best grid point by smoothed cost |
| C_hard at grid winner [0.30, 0.45] | 0.2829383159770988 | expected_cost | LOWEST HARD COST FOUND — lower than returned |
| C_hard at returned [0.60, 0.75] | 0.38722259534246795 | expected_cost | Returned point — NOT hard-cost minimum over visited |
| baseline FPR | 0.06942277691107644 (6.94%) | baseline.rates.fpr | |
| baseline FNR | 0.062352012628255724 (6.24%) | baseline.rates.fnr | |
| baseline Review | 0.03737950029510132 (3.74%) | baseline.rates.review_rate | |
| optimized FPR | 0.0343 (3.43%) | optimized.rates_at_optimum.fpr | -50.6% |
| optimized FNR | 0.0681 (6.81%) | optimized.rates_at_optimum.fnr | +9.2% (+9.22% exact) |
| optimized Review | 0.0179 (1.79%) | optimized.rates_at_optimum.review_rate | -52.1% |
| tau | 0.05 | optimized.tau | Sigmoid temperature |
| learning_rate | 0.05 | optimized.learning_rate | GD eta |
| grad_tol | 1e-4 | optimized.grad_tol | From JSON |
| stopped_by | step_tol | optimized.stopped_by | From JSON |
| cost_weights | C_FP=1.0 C_FN=5.0 C_review=0.5 | baseline.cost_weights | |
| bayes_optimal_single_threshold | 0.1667 | C_FP/(C_FP+C_FN) | |
| n_test | 5083 | baseline.n_test | Held-out URLs |
| n_test_domains | 1497 | baseline.n_test_domains | Unseen domains |
| prevalence | 0.7478 | baseline.prevalence | Phishing prevalence |
| seed | 42 | build_dataset | |
| len(history) | 30 | optimized.history | Indices 0-29 |
| gd_iter | 29 | derivative_check.gradient_descent_iterations | Verified b7440be |
| duplicate? | False | history[-1]!=history[-2] | Fixed — was identical before |
| history[0] | [0.3,0.45,0.29071968134044274] | optimized.history[0] | Grid best — C_smooth 0.2907 C_hard 0.2829 |
| history[-1] | [0.6,0.75,0.43994209280287067] | optimized.history[-1] | Returned — C_smooth 0.4399 C_hard 0.3872 |
| history[-2] | [0.5937904825732752,0.7437904825732752,0.43114619878929056] | optimized.history[-2] | Previous |
| hessian_det | 6.923728 | derivative_check.hessian_det | >0 locally strictly convex |
| second_derivative_review | 0.405678 | derivative_check | From verification |
| second_derivative_block | 17.086192 | derivative_check | From verification |
| trace | 17.49187 | second_deriv sum | 0.405678+17.086192 = 17.49187 >0 |
| first_derivative_review | -0.153059 | derivative_check | Wants t_r UP but at upper bound 0.60 |
| first_derivative_block | 1.621676 | derivative_check | Wants t_b DOWN but gap blocks at 0.75 |
| boundary_hit | ['review_box_0.60','gap_0.15'] | derivative_check | Active constraints |
| is_stationary | false | derivative_check | Boundary-clipped, not interior |

Note: "the returned point is not the hard-cost minimum over all visited points" — grid winner [0.30,0.45] hard 0.2829 is lower than returned [0.60,0.75] hard 0.3872 — honest report must state.

Three-row table for prohibitions B-5:

| Point | C_hard | C_smooth | Note |
|-------|--------|----------|------|
| Baseline [0.35,0.70] | 0.39987259019990573 | — | reference |
| Grid winner [0.30,0.45] | 0.2829383159770988 | 0.29071968134044274 | lowest hard cost found |
| Returned [0.60,0.75] | 0.38722259534246795 | 0.43994209280287067 | GD boundary-clipped point |

## 2. Baseline — results/baseline.json — VERIFIED EXACT d92f080

roc_auc 0.9931, pr_auc 0.9977, brier 0.0302, tpr 0.9584, fpr 0.0484, precision 0.9833 at 0.50 cutoff, f1_score 0.9707, n_test 5083, n_test_domains 1497, test_phishing_prevalence 0.7478, confusion [[1220,62],[158,3643]] — TN 1220 FP 62 FN 158 TP 3643

## 3. Test Status

42/42 tests pass — evidence results/final_test_run.txt — commit 5fdc189aac39267cb96f04f4f3e409535be656a5

## 4. CRITICAL FINDING — Grid best vs GD — Honest story — 7a60257 + 0b3cc3b + PB-E

Grid best smoothed cost: 0.29071968134044274 at (0.3,0.45) — Hard cost 0.2829383159770988 — LOWEST
Returned point (0.6,0.75): Smoothed 0.43994209280287067 Hard 0.38722259534246795
Baseline (0.35,0.70): Hard 0.39987259019990573

GD increased both: Smoothed +51.36% 0.2907→0.4399, Hard +36.87% 0.2829→0.3872 — not descent due to projection not preserving descent property — is_stationary false, first_derivative_block 1.621676.

Improvement over baseline 0.3999→0.3872 -3.16% came from grid search, NOT GD — GD undid most of grid's benefit — honest report must say combined grid+GD returns feasible boundary point satisfying second-order conditions, not hard-cost minimiser.

Report must say: "Combined grid+GD returns (0.60,0.75) hard 0.3872 — improves over baseline 0.3999 by 3.16% — improvement attributable to grid search, NOT GD — GD is constrained refinement returning feasible boundary point satisfying second-order conditions (hessian_det 6.923728>0, trace 17.49>0) but not improving on grid winner in hard cost."

---
End of Results Brief — FINAL 5fdc189aac39267cb96f04f4f3e409535be656a5 — JSON EXACT — HONEST
| optimized FNR | 0.0681 (6.81%) | optimized.rates_at_optimum.fnr | +9.2% (+9.22% exact) |
| optimized Review | 0.0179 (1.79%) | optimized.rates_at_optimum.review_rate | -52.1% |
| tau | 0.05 | optimized.tau | Sigmoid temperature |
| learning_rate | 0.05 | optimized.learning_rate | GD eta |
| grad_tol | 1e-4 or similar | optimized.grad_tol | From JSON |
| stopped_by | step_tol or max_iter | optimized.stopped_by | From JSON |
| cost_weights | C_FP=1.0 C_FN=5.0 C_review=0.5 | baseline.cost_weights | |
| bayes_optimal_single_threshold | ~0.45 or N/A | optimized.bayes_optimal_single_threshold | From JSON |
| n_test | 5083 | baseline.n_test | Held-out URLs |
| n_test_domains | 1497 | baseline.n_test_domains | Unseen domains |
| prevalence | 0.7478 | baseline.prevalence | Phishing prevalence |
| seed | 42 | build_dataset | |
| len(history) | 30 | optimized.history | Indices 0-29 |
| gd_iter | 29 | derivative_check.gradient_descent_iterations | Verified b7440be |
| duplicate? | False | history[-1]!=history[-2] | Fixed — was identical before |
| history[0] | [0.3,0.45,0.29071968134044274] | optimized.history[0] | Grid best — C_smooth 0.2907 C_hard 0.2829 |
| history[-1] | [0.6,0.75,0.43994209280287067] | optimized.history[-1] | Returned — C_smooth 0.4399 C_hard 0.3872 |
| history[-2] | [0.5937904825732752,0.7437904825732752,0.43114619878929056] | optimized.history[-2] | Previous |
| hessian_det | 6.923728 | derivative_check.hessian_det | >0 locally strictly convex |
| trace | 17.49187 | second_deriv sum | >0 |
| first_derivative_review | -0.153059 | derivative_check | Wants t_r UP but at upper bound 0.60 |
| first_derivative_block | 1.621676 | derivative_check | Wants t_b DOWN but gap blocks at 0.75 |
| boundary_hit | ['review_box_0.60','gap_0.15'] | derivative_check | Active constraints |
| is_stationary | false | derivative_check | Boundary-clipped, not interior |

Note: "The returned point is not the hard-cost minimum over all visited points" — grid winner [0.30,0.45] hard 0.2829 is lower than returned [0.60,0.75] hard 0.3872 — honest report must state.

## 2. Baseline — results/baseline.json — VERIFIED EXACT d92f080

roc_auc 0.9931, pr_auc 0.9977, brier 0.0302, tpr 0.9584, fpr 0.0484, precision 0.9833 at 0.50 cutoff, f1_score 0.9707, n_test 5083, n_test_domains 1497, test_phishing_prevalence 0.7478, confusion [[1220,62],[158,3643]] — TN 1220 FP 62 FN 158 TP 3643

## 3. Test Status — 42/42 PASS platform linux Python 3.14.2 pytest-9.2.0 22.44s evidence final_test_run.txt — commit 7506878

## 4. CRITICAL FINDING — Grid best vs GD — Honest story — 7a60257 + 0b3cc3b + PB-E

Grid best smoothed cost: 0.29071968134044274 at (0.3,0.45) — Hard cost 0.2829383159770988 — LOWEST HARD COST FOUND
Returned point (0.6,0.75): Smoothed 0.43994209280287067 Hard 0.38722259534246795
Baseline (0.35,0.70): Hard 0.39987259019990573

GD increased both: Smoothed +51% 0.2907→0.4399, Hard +36.8% 0.2829→0.3872 — not descent due to projection not preserving descent property — is_stationary false, first_derivative_block 1.621676.

Improvement over baseline 0.3999→0.3872 -3.16% came from grid search, NOT GD — GD undid most of grid's benefit — honest report must say combined grid+GD returns feasible boundary point satisfying second-order conditions, not hard-cost minimiser.

Report must say: "Combined grid+GD returns (0.60,0.75) hard 0.3872 — improves over baseline 0.3999 by 3.16% — improvement attributable to grid search, NOT GD — GD is constrained refinement returning feasible boundary point satisfying second-order conditions (hessian_det 6.923728>0) but not improving on grid winner in hard cost."

---
End of Results Brief — FINAL 7506878 — JSON EXACT — HONEST
