Final project commit: ab92b84aba3161866284a746a85746ac720ef81c (short: ab92b84). — Original honest finding 5fdc189aac39267cb96f04f4f3e409535be656a5 — len 30 gd 29 duplicate False





# Math Brief — Source of Truth for Report
### Previous chain: b7440be -> 0b3cc3b -> d66e2a4 -> 09320bc -> 5fdc189 — len=30 gd=29 duplicate False
### Canonical Source: results/calculus_experiment.json + src/calculus/calculus_threshold.py — file wins
### All numbers in this report are traceable to results/calculus_experiment.json and results/baseline.json at commit 5fdc189aac39267cb96f04f4f3e409535be656a5. Test evidence: results/final_test_run.txt (42/42 pass, Python 3.14.2, pytest 9.2.0).

## 1. Cost Function — MB-1 + MB-2 FIXED

Baseline hard cost C_hard(0.35,0.70) = 0.39987259019990573 per JSON exact
Optimized hard cost C_hard(0.60,0.75) = 0.387223 per JSON exact — this is the hard cost at the returned point, NOT the minimum over all visited points
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

Smoothed Cost C_smooth = C_FP*FPR~ + C_FN*FNR~ + C_review*Review~
Final smoothed cost at optimum: 0.43994209280287067 per JSON exact — history[-1][2]

## 2. Gradient and Hessian — MB-2 + MB-D

At returned point (0.60,0.75):
first_derivative_review = -0.153059
first_derivative_block = 1.621676
second_derivative_review = 0.405678
second_derivative_block = 17.086192
hessian_det = 6.923728
trace = 17.49187
is_stationary = false — boundary-clipped

MB-D FIXED: Hessian determinant 6.923728>0 and trace 17.49>0 prove smoothed surrogate locally strictly convex at returned point. Necessary condition for local minimum of smoothed surrogate on feasible set; NOT proof of constrained optimality (multipliers not verified).

## 3. GD Update and Projection — MB-3 FIXED

eta = 0.05
t_r_new = t_r - eta*grad_r
t_b_new = t_b - eta*grad_b
Projection: t_r clip 0.15-0.60, t_b clip 0.45-0.95
If tb_new - tr_new <0.15: tb_new = clip(tr_new+0.15, 0.45,0.95) — push t_b UP — not t_r down

Verified: len(history)=30, gradient_descent_iterations=29, history[0]=[0.3,0.45,0.29071968134044274], history[-1]=[0.6,0.75,0.43994209280287067], history[-2]=[0.5937904825732752,0.7437904825732752,0.43114619878929056], duplicate False

## 4. Boundary Analysis — MB-4 + MB-B FIXED — Safe claim

Active: review_box_0.60 (t_r at upper box) + gap_0.15 (t_b - t_r =0.15 min)
∂C/∂t_r = -0.153059 <0 — wants t_r UP, but at upper bound 0.60
∂C/∂t_b = 1.621676 >0 — wants t_b DOWN, but gap t_b≥0.75 blocks — at lower bound of feasible set for t_b, gradient points outward

Safe claim: Boundary-clipped feasible point at 0.60/0.75 — review at upper box bound 0.60 and gap at minimum 0.15 — Hessian det 6.923728>0 and trace 17.49>0 prove smoothed surrogate locally strictly convex there — hard cost 0.387223 minimum found by combined grid + gradient descent procedure — NOT claimed as proven constrained optimum (KKT multipliers not verified).

## 5. History — MB-A VERIFIED — Grid winner vs GD — Honest — FINAL 5fdc189aac39267cb96f04f4f3e409535be656a5

Verified via grid search 100x100 linspace 0.15-0.60 review, 0.45-0.95 block, gap 0.15:

Grid best smoothed cost: 0.29071968134044274
Grid best (r, b): 0.3 0.45
Hard cost at grid best: 0.2829383159770988 — LOWEST HARD COST FOUND
Grid fallback used: false — history[0] equals grid best

History verified:
len(history) = 30
gradient_descent_iterations = 29
history[0] = [0.3, 0.45, 0.29071968134044274] — C_smooth 0.2907 C_hard 0.2829383159770988 — grid winner
history[-1] = [0.6, 0.75, 0.43994209280287067] — C_smooth 0.4399 C_hard 0.38722259534246795 — returned
history[-2] = [0.5937904825732752, 0.7437904825732752, 0.43114619878929056]
duplicate? False — FIXED

C_hard at history[0] [0.3,0.45] = 0.2829383159770988 — Smoothed 0.29071968134044274
C_hard at history[-1] [0.6,0.75] = 0.38722259534246795 — Smoothed 0.43994209280287067
C_hard at baseline (0.35,0.70) = 0.39987259019990573
C_hard at returned (0.60,0.75) = 0.38722259534246795 = optimized.optimal_cost 0.387223 — this is hard cost at returned point, NOT minimum over visited

Interpretation — FIXED B-4 wording — no "hard optimum":

GD moved away from the grid-winner by smoothed cost — [0.30, 0.45], which also happened to have the lowest hard cost of any point visited — toward [0.60, 0.75], where both metrics are higher. This is a property of the smoothed surrogate under constraint projection, not a global statement about hard-cost minima.

JSON reports optimized.optimal_cost = 0.387223 at the returned point. This is not the minimum hard cost over all visited points — initial grid winner (0.30,0.45) has hard cost 0.2829, which is lower. JSON reports hard cost at point procedure returns, not minimum over search.

GD increased both metrics during 29 iterations: C_smooth 0.2907→0.4399 +51.36%, C_hard 0.2829→0.3872 +36.87% — not descent due to projection onto gap (0.15) and box (0.15-0.60, 0.45-0.95) not preserving descent property — is_stationary false, first_derivative_block 1.621676 >0 wants t_b DOWN but gap blocks at 0.75.

Improvement over baseline 0.3999→0.3872 -3.16% came from grid search, NOT GD — GD undid most of grid's benefit — grid best [0.3,0.45] hard 0.2829 lower than returned [0.6,0.75] hard 0.3872.

Report must say: Combined grid+GD returns (0.60,0.75) hard 0.3872 — improves over baseline 0.3999 by 3.16% — improvement attributable to grid search, NOT GD — GD is constrained refinement returning feasible boundary point satisfying second-order conditions (hessian_det 6.923728>0, trace 17.49>0, locally strictly convex) but not improving on grid winner in hard cost — report returned point honestly as feasible point satisfying second-order conditions, not hard-cost minimiser.

## 6. Reward Policy — MB-6 FIXED

step = lr * (multiplier + margin), lr = max(0.015, 0.05/(1+0.05*episodes)), multiplier 1.25 raise /1.75 lower, margin = |p - t| — NOT constant 0.02/0.05

File rl_policy_state.json legacy artefact — stores two scalars + reward counters — not learned policy — must mention in report per P-5

Policy heuristic inspired by RL reward shaping, not formal RL — no MDP, no Q-learning

## 7. Verification — MB-C — Commit hash — FINAL 5fdc189aac39267cb96f04f4f3e409535be656a5

Evidence: results/final_test_run.txt 42/42 PASS platform linux Python 3.14.2 pytest-9.2.0 22.44s

Canonical source rule: results/calculus_experiment.json and src/calculus/calculus_threshold.py — file wins over brief if disagree.

---
End of Math Brief — FINAL 5fdc189aac39267cb96f04f4f3e409535be656a5 — JSON+CODE EXACT — HONEST — Grid winner 0.3,0.45 hard 0.2829 < returned 0.6,0.75 hard 0.3872
