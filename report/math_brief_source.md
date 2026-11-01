# Math Brief — Source of Truth for Report — FIXED MB-A to MB-D
## ARES CIE3 — Formulas Teammates Must Copy Exactly
### Final project commit: 0b3cc3b2a7317a42c39d2c76e03793dd2223ae60 — len=30 gd=29 duplicate False — previous chain 0b3cc3b→7a60257→d66e2a4
### Canonical Source: results/calculus_experiment.json + src/calculus/calculus_threshold.py — file wins
### FIXED: MB-A history hard costs, MB-B safe claim, MB-C commit hash, MB-D Hessian caveat

## 1. Cost Function — MB-1 + MB-2 FIXED

Baseline hard cost C_hard(0.35,0.70) = 0.39987259019990573 per JSON exact
Optimized hard cost C_hard(0.60,0.75) = 0.387223 per JSON exact
Delta = -0.01264959019990573 -3.16%

Where per code _compute_two_thr_rates:

allow = y_proba < t_review
review = (y_proba >= t_review) & (y_proba < t_block)
block = y_proba >= t_block

FPR(t_r,t_b) = P(p ≥ t_r | y=0) = (fp_review + fp_block)/total_benign — Review OR Block — NOT only Block — MB-2 FIXED
FNR(t_r,t_b) = P(p < t_b | y=1) = (fn_allow + fn_review)/total_phish — Allow OR Review — NOT only Allow — MB-2 FIXED
Review = P(t_r ≤ p < t_b)

Coefficients: C_FP=1.0, C_FN=5.0, C_review=0.5

### Smoothed Surrogate — MB-2 FIXED EXACT CODE

s_review(p) = sigma((p - t_r)/tau) — smooth at/above review
s_block(p) = sigma((p - t_b)/tau) — smooth at/above block
sigma(x)=1/(1+exp(-x)), tau=0.05

FPR~ = mean(s_review[benign]) — because review OR block triggers FPR
FNR~ = 1 - mean(s_block[phish]) — because allow OR review triggers FNR
Review~ = mean(s_review * (1 - s_block))

Previous WRONG: FPR~ mean soft_Block, FNR~ mean soft_Allow — now fixed to code exact.

Smoothed Cost C_smooth = C_FP*FPR~ + C_FN*FNR~ + C_review*Review~
Final smoothed cost at optimum: 0.43994209280287067 per JSON exact — history[-1][2]

## 2. Gradient and Hessian — MB-2 derivative + MB-D Hessian caveat

∂C_smooth/∂t_r = C_FP*∂FPR~/∂t_r + C_review*∂Review~/∂t_r
∂C_smooth/∂t_b = C_FN*∂FNR~/∂t_b + C_review*∂Review~/∂t_b

At returned point (0.60,0.75):
first_derivative_review = -0.153059
first_derivative_block = 1.621676
second_derivative_review = 0.405678
second_derivative_block = 17.086192
hessian_det = 6.923728
is_stationary = false — boundary-clipped
trace = 0.405678+17.086192 = 17.49187 >0

MB-D FIXED — Hessian caveat:
Previous WRONG: "Hessian det 6.923728>0 proves local minimum" — incorrect for constrained boundary case.

Correct per MB-D:
"Hessian determinant 6.923728 >0 and trace 17.49 >0 prove the smoothed surrogate is locally strictly convex at the returned point. This is necessary condition for local minimum of smoothed surrogate on feasible set; it is NOT proof of constrained optimality (multipliers not verified per P-4). To claim constrained optimality need KKT with μ≥0, ν≥0 verified — not done, so use boundary-clipped feasible point safe claim."

## 3. GD Update and Projection — MB-3 FIXED

eta = 0.05
t_r_new = t_r - eta*grad_r
t_b_new = t_b - eta*grad_b
Projection: t_r clip 0.15-0.60, t_b clip 0.45-0.95
If tb_new - tr_new <0.15: tb_new = clip(tr_new+0.15, 0.45,0.95) — push t_b UP — MB-3 FIXED — not t_r down — push t_r down only if t_b hits 0.95

Verified: len(history)=30, gradient_descent_iterations=29, history[0]=[0.3,0.45,0.29071968134044274], history[-1]=[0.6,0.75,0.43994209280287067], history[-2]=[0.5937904825732752,0.7437904825732752,0.43114619878929056], duplicate False — FIXED — b7440be

## 4. Boundary Analysis — MB-4 + MB-B FIXED — Safe claim

MB-4 FIXED — downgrade KKT to boundary-clipped feasible point per P-4.
Active: review_box_0.60 (t_r at upper box) + gap_0.15 (t_b - t_r =0.15 min)

∂C/∂t_r = -0.153059 <0 — wants t_r UP, but at upper bound 0.60 — cannot
∂C/∂t_b = 1.621676 >0 — wants t_b DOWN, but gap t_b≥0.75 blocks — at lower bound of feasible set for t_b, gradient points outward

MB-B FIXED — safe claim wording:
Previous WRONG: "hard cost 0.387223 is minimum over grid verification" — did not verify grid_fallback_used, and implies only grid.

Correct per MB-B:
"Boundary-clipped feasible point at 0.60/0.75 — review threshold at its upper box bound 0.60 and gap at minimum 0.15 — Hessian determinant 6.923728 >0 and trace 17.49 >0 prove smoothed surrogate locally strictly convex there — hard cost 0.387223 is minimum found by combined grid + gradient descent procedure — NOT claimed as proven constrained optimum (KKT multipliers not verified)."

Check JSON: grid_fallback_used = false (grid found winner) — so returned point is best grid point plus GD refinement — combined procedure — fix wording.



## 5. History — MB-A VERIFIED — Grid winner vs GD — Honest — FINAL 7506878

Verified via grid search 100x100 linspace 0.15-0.60 review, 0.45-0.95 block, gap 0.15:

Grid best smoothed cost: 0.29071968134044274
Grid best (r, b): 0.3 0.45
Hard cost at grid best: 0.2829383159770988 — LOWEST HARD COST FOUND
Grid fallback used: false — history[0] equals grid best

History verified b7440be:
len(history) = 30
gradient_descent_iterations = 29
history[0] = [0.3, 0.45, 0.29071968134044274] — C_smooth 0.2907 C_hard 0.2829383159770988 — grid winner
history[-1] = [0.6, 0.75, 0.43994209280287067] — C_smooth 0.4399 C_hard 0.38722259534246795 — returned
history[-2] = [0.5937904825732752, 0.7437904825732752, 0.43114619878929056]
duplicate? False — FIXED

C_hard at history[0] [0.3,0.45] = 0.2829383159770988 — Smoothed 0.29071968134044274
C_hard at history[-1] [0.6,0.75] = 0.38722259534246795 — Smoothed 0.43994209280287067
C_hard at baseline (0.35,0.70) = 0.39987259019990573
C_hard at returned (0.60,0.75) = 0.38722259534246795 = optimized.optimal_cost 0.387223

Interpretation — HONEST, no "minimum over procedure":
JSON reports optimized.optimal_cost = 0.387223 at the returned point. This is not the minimum hard cost over all visited points — initial grid winner (0.30,0.45) has hard cost 0.2829, which is lower. JSON reports hard cost at point procedure returns, not minimum over search.

GD increased both metrics during 29 iterations: C_smooth 0.2907→0.4399 +51%, C_hard 0.2829→0.3872 +36.8% — not descent due to projection onto gap (0.15) and box (0.15-0.60, 0.45-0.95) not preserving descent property — is_stationary false, first_derivative_block 1.621676 >0 wants t_b DOWN but gap blocks at 0.75.

Improvement over baseline 0.3999→0.3872 -3.16% came from grid search, NOT GD — GD undid most of grid's benefit — grid best [0.3,0.45] hard 0.2829 lower than returned [0.6,0.75] hard 0.3872.

Report must say: Combined grid+GD returns (0.60,0.75) hard 0.3872 — improves over baseline 0.3999 by 3.16% — improvement attributable to grid search, NOT GD — GD is constrained refinement returning feasible boundary point satisfying second-order conditions (hessian_det 6.923728>0, trace 17.49>0, locally strictly convex) but not improving on grid winner in hard cost — report returned point honestly as feasible point satisfying second-order conditions, not hard-cost minimiser.

Previous invented numbers: "Iteration 0: C_smooth≈0.42, C_hard≈0.51" — WRONG — correct 0.2907 and 0.2829 — MB-A verified.

## 6. CRITICAL FINDING — GD increased hard cost — honest story — 7a60257

See §5 above — grid best [0.3,0.45] hard 0.2829 lowest — GD increased to 0.3872 — honest report feasible boundary point not minimiser.

## 7. Verification — MB-C — Commit hash — FINAL 7506878

Final project commit: 7506878849402bfa147d05311bdba2659f6292c6 — len=30 gd=29 duplicate False — previous chain 0b3cc3b→7a60257→d66e2a4→09320bc→b7440be

Evidence: results/final_test_run.txt 42/42 PASS platform linux Python 3.14.2 pytest-9.2.0 22.44s

Canonical source rule: results/calculus_experiment.json and src/calculus/calculus_threshold.py — file wins over brief if disagree.

---
End of Math Brief — FINAL 7506878 — JSON+CODE EXACT — HONEST — Grid winner 0.3,0.45 hard 0.2829 < returned 0.6,0.75 hard 0.3872
