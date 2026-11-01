# Results Brief — CANONICAL — results/calculus_experiment.json + baseline.json EXACT
## Commit: 6f8942e RB-1 to RB-7 + f0341ef P-1 to P-5

### 1. Calculus — results/calculus_experiment.json — FILE WINS
baseline.cost 0.39987259019990573 (0.35/0.70) → optimized.optimal_cost 0.387223 (0.60/0.75) delta -0.01264959019990573 -3.16% final_cost smoothed 0.439942
FPR 0.06942277691107644→0.0343 6.94%→3.43% -50.6%
FNR 0.062352012628255724→0.0681 6.24%→6.81% +9.2%
Review 0.03737950029510132→0.0179 3.74%→1.79% -52.1%
hessian_det 6.923728 first_derivative_review -0.153059 first_derivative_block 1.621676 second_derivative_review 0.405678 second_derivative_block 17.086192 boundary_hit review_box_0.60+gap_0.15 gd 29 history 30

### 2. Baseline — results/baseline.json — RB-4 VERIFIED EXACT
roc_auc 0.9931
pr_auc 0.9977
brier 0.0302
tpr 0.9584
fpr 0.0484
precision 0.9833 at 0.50 cutoff
f1_score 0.9707
n_test 5083
n_test_domains 1497
test_phishing_prevalence 0.7478
confusion [[1220, 62], [158, 3643]] — TN 1220 FP 62 FN 158 TP 3643

### 3. Test Status — 42/42 PASS platform linux Python 3.14.2 pytest-9.2.0 22.44s evidence final_test_run.txt

## 4. CRITICAL FINDING — Grid best vs GD — Honest story — 7a60257

Grid best smoothed cost: 0.29071968134044274 at (0.3,0.45) — Hard cost 0.2829383159770988 — LOWEST HARD COST FOUND
Returned point (0.6,0.75): Smoothed 0.43994209280287067 Hard 0.38722259534246795
Baseline (0.35,0.70): Hard 0.39987259019990573

GD increased both: Smoothed +51% 0.2907→0.4399, Hard +36.8% 0.2829→0.3872 — not descent due to projection not preserving descent property — is_stationary false, first_derivative_block 1.621676.

Improvement over baseline 0.3999→0.3872 -3.16% came from grid search, NOT GD — GD undid most of grid's benefit — honest report must say combined grid+GD returns feasible boundary point satisfying second-order conditions, not hard-cost minimiser.

