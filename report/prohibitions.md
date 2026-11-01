# Prohibitions — What NOT to Write in Report — FIXED PB-A to PB-D
## ARES CIE3 — Hard Rules — b7440be + 09320bc FINAL — JSON+CODE EXACT
### Canonical Source: results/calculus_experiment.json + results/baseline.json + src/calculus/calculus_threshold.py — file wins
### Final Commit: 09320bc — len=30 gd=29 duplicate False — previous b7440be→2f52c31→50943b5→d92f080→6f8942e→f0341ef
### Evidence: results/final_test_run.txt 42/42 PASS platform linux Python 3.14.2 pytest-9.2.0 22.44s

This document lists statements that must NOT appear in report v1.4 or viva_notes. Each prohibition has reason and correct alternative. All numbers must match JSON exact per verification.

PB-A FIXED — No self-contradiction — file wins — verified len=30 gd=29

PB-B VERIFIED — FNR +9.22% correct

PB-C FIXED — grid verification wording → combined grid+GD

PB-D ADDED — rl_policy_state.json vs calculus_experiment.json thresholds differ

---

### 1. Do NOT call the reward policy RL — P-2 + PB-A

**PROHIBITED:**
- "We use Reinforcement Learning (RL) for threshold adaptation"
- "Our RL agent learns optimal thresholds via Q-learning"
- "RL policy with Bellman equation"
- "Deep RL / DQN for endpoint security"
- "Adaptive learning" as synonym for RL — adaptive is fine, but do not imply RL

**WHY:** No MDP, no state transition, no Bellman, no Q-learning, no value function, no discount factor, no exploration. Policy is two scalars (t_r,t_b) adjusted by heuristic rewards.

**CORRECT:**
- "Feedback-driven heuristic threshold adaptation inspired by RL reward shaping, but not formal RL"
- "Heuristic policy with reward +1.0 correct, -1.5 false positive, -3.0 false negative"
- "Thresholds adjusted, classifier not retrained"

PB-A FIXED — Previous self-contradiction: "We ran gd_iter=29 actual from verification len=30 gd=29" later "If JSON says 30, use 30 — file wins" — self-contradictory, teammate confused.

Correct per b7440be verified output:
```
len(history) = 30
gradient_descent_iterations = 29
history[0]  = [0.3, 0.45, 0.29071968134044274]
history[-1] = [0.6, 0.75, 0.43994209280287067]
history[-2] = [0.5937904825732752, 0.7437904825732752, 0.43114619878929056]
duplicate? False
```
So final correct: gd_iter 29, len 30, no hedging — file is results/calculus_experiment.json — whatever it says wins — current file says 29/30 — use that — no "if JSON says 30" clause.

CORRECT FINAL: "GD ran 29 iterations (30 history entries indices 0-29), verified len=30 gd=29 duplicate False — file wins — results/calculus_experiment.json"

---

### 2. Do NOT claim GD converged — P-3 + PB-A + PB-C FIXED

**PROHIBITED:**
- "Gradient descent converged to global optimum"
- "GD converged in 29 iterations"
- "Converged thresholds 0.60/0.75"

**WHY:** GD on smoothed surrogate C_smooth did NOT converge to interior zero-gradient. Result is boundary-clipped feasible point on boundary (review_box_0.60 + gap_0.15), is_stationary=false, first_derivative_block=1.621676 !=0. Verified len=30 gd=29 per b7440be.

**CORRECT:**
- "GD ran 29 iterations (30 history entries indices 0-29), verified len=30 gd=29 duplicate False, reaching boundary-clipped feasible point at 0.60/0.75 on boundary"
- "hessian_det=6.923728>0 and trace 17.49>0 prove smoothed surrogate locally strictly convex — necessary condition, not proof of constrained optimality (multipliers not verified)"
- PB-C FIXED: Previous "Grid verification confirms 0.60/0.75 minimal hard cost per JSON" — not quite right — fix to "combined grid + gradient descent procedure finds 0.60/0.75 as minimum hard cost per JSON" — per MB-B
- "Hard cost 0.387223 minimum found by combined grid + gradient descent procedure — NOT grid only"

---

### 3. Do NOT claim false negatives decreased — P-3 + PB-B VERIFIED

**PROHIBITED:**
- "Calculus optimization reduced false negatives"
- "FNR decreased"
- "Both FPR and FNR improved"

**WHY:** FNR increased per JSON exact.

**CORRECT — PB-B VERIFIED:**
Baseline FNR = 0.062352012628255724 (6.24%)
Optimized FNR = 0.0681 (6.81%)
Delta = 0.0681-0.062352 = 0.005748
Relative = 0.005748/0.062352 = 9.22% ✅ — calculation correct

FPR 0.06942277691107644→0.0343 6.94%→3.43% -50.6%
FNR 0.062352012628255724→0.0681 6.24%→6.81% +9.2% (+9.22% exact)
Review 0.03737950029510132→0.0179 3.74%→1.79% -52.1%
Cost 0.39987259019990573→0.387223 -3.16% — FPR reduction outweighs FNR increase despite C_FN=5.0

Do NOT quote 0.3676, 0.387223 as baseline, -5.1%, 7.17%→3.28%, 5.81%→6.37% — correct per JSON exact above — b7440be verified

---

### 4. Do NOT quote precision_at_prevalence as if it reflects calculus-optimised thresholds

**PROHIBITED:**
- "Precision at prevalence 0.5 with optimized thresholds 0.60/0.75 is 98.27%"

**WHY:** Precision 98.27% is at classification cutoff 0.50 per baseline.json — but actual verified precision is 0.9833 at 0.50 per d92f080 — use 0.9833 exact.

**CORRECT:**
- "Precision 0.9833 at 0.50 classification cutoff (baseline.json verified d92f080) — NOT at calculus thresholds 0.60/0.75"
- "Policy thresholds 0.60/0.75 govern Allow/Review/Block, cost evaluated separately"

---

### 5. Do NOT invent sklearn/Numpy/Python versions

**PROHIBITED:**
- "sklearn 1.5.2" (old RF artifact)
- "numpy 1.26.0"
- "Python 3.10/3.11"

**WHY:** Actual from evidence:
- sklearn 1.9.1 (bundle features == feature_names True, Task 7 PASS)
- numpy 2.5.4
- Python 3.14.2 (from final_test_run.txt)
- pytest 9.2.0
- joblib_sklearn_1.9.1 loader

**CORRECT:** "sklearn 1.9.1, numpy 2.5.4, Python 3.14.2, pytest-9.2.0 — evidence final_test_run.txt 42/42 PASS"

---

### 6. Do NOT invent dataset counts + PB-D rl_policy_state.json vs calculus_experiment.json

**PROHIBITED:**
- "160064 total dataset" as final (raw before expansion)
- "4693 test URLs" (v2.0 RF, not v3.0 XGBoost)
- "820 benign" as final (raw before expansion to 7380)
- "7482 groups" as test only (total groups)

**WHY:** Correct per b7440be:
- Raw: 820 benign roots + 159244 phishing = 160064 rows
- Built: build_dataset(seed=42) → 22140 total, labels {1:14760 phishing, 0:7380 benign}, groups 7482 (5985 train + 1497 test)
- Split: GroupShuffleSplit 20% → test 5083 URLs, 1497 domains, prevalence 0.7478
- is_shortener 510/22140 (2.30%), tld_risk 213/22140 (0.96%)
- Confusion: TN 1220 FP 62 FN 158 TP 3643 per d92f080

**PB-D ADDED — rl_policy_state.json vs calculus_experiment.json:**
- File name results/rl_policy_state.json is legacy naming artefact — stores two scalars (thresholds) and reward counters — not learned policy — must mention in report per P-5
- PB-D: results/rl_policy_state.json may show different thresholds than results/calculus_experiment.json (because app writes state on feedback, and generate_state.py writes state on other runs)
- Report must say: "The thresholds stored in results/rl_policy_state.json reflect the current runtime state and may differ from the calculus-optimised (0.60, 0.75) if feedback has been recorded. The canonical source for the reported thresholds is results/calculus_experiment.json."
- Important: If examiner opens results/rl_policy_state.json and sees thresholds 0.55/0.70, they will wonder why differ from report's 0.60/0.75
- Prohibition: "Do NOT quote rl_policy_state.json thresholds as the calculus result — canonical is calculus_experiment.json"

**CORRECT:** Quote exactly from results_brief.md b7440be + d92f080 + 09320bc.

---

### P-4 — KKT Claim — DOWNGRADED per recommendation (a) — PB-C wording

**PROHIBITED (RISKY):**
- "KKT constrained optimum" unless you verify multipliers
- "Grid verification confirms 0.60/0.75 minimal hard cost per JSON" — PB-C — not quite right — fix to combined grid+GD

**WHY:** To claim KKT need Lagrangian, stationarity, primal/dual feasibility, complementary slackness — must check signs — not verified.

**CORRECT (SAFE):**
- "Boundary-clipped feasible point at 0.60/0.75 — review at upper box bound 0.60 and gap at minimum 0.15 — Hessian determinant 6.923728>0 and trace 17.49>0 prove smoothed surrogate locally strictly convex there — hard cost 0.387223 minimum found by combined grid + gradient descent procedure — NOT claimed as proven constrained optimum (KKT multipliers not verified)"
- Do NOT let teammate write KKT optimum unless you verify multipliers yourself

---

### Additional Prohibitions (from T1-T8 + MB-A)

- Do NOT claim scale_pos_weight >1 (actual 0.556 <1)
- Do NOT claim has_https proves trust (transport only, randomized p_strip=0.5)
- Do NOT claim QR decoding requires internet (offline verified grep empty)
- Do NOT claim RGBA white-on-transparent PASS (FAILs, inverted QR rare edge case, black-on-transparent PASS)
- Do NOT claim generate_qr_image is safety filter (testing utility, warning)
- Do NOT claim multi-QR returns single (fail-closed raises QRDecodeError)
- Do NOT claim 26 tests (now 42/42)
- Do NOT quote 0.3676, 0.387223 as baseline, -5.1%, 7.17%→3.28%, 5.81%→6.37%, 3.65%→1.36% — correct per JSON exact: baseline 0.39987259019990573 → 0.387223 -3.16%, FPR 0.06942277691107644→0.0343, FNR 0.062352012628255724→0.0681, Review 0.03737950029510132→0.0179, hessian 6.923728, final_cost 0.43994209280287067, history[0] 0.2829383159770988/0.29071968134044274, history[-1] 0.38722259534246795/0.43994209280287067
- Do NOT claim C_smooth≈0.42 C_hard≈0.51 at iteration 0 — correct C_smooth 0.29071968134044274 C_hard 0.2829383159770988 — MB-A verified 09320bc
- Do NOT claim history[-1]==history[-2] duplicate — duplicate False fixed — b7440be verified
- Do NOT quote rl_policy_state.json thresholds as calculus result — PB-D

---

### Enforcement — PB-A to PB-D

- All numbers must match results/calculus_experiment.json EXACT — file wins — current verified len=30 gd=29 duplicate False — b7440be
- All numbers must match results/baseline.json EXACT — file wins — current verified roc_auc 0.9931 pr_auc 0.9977 brier 0.0302 confusion [[1220,62],[158,3643]] — d92f080
- All formulas must match src/calculus/calculus_threshold.py EXACT — file wins — FPR~ mean s_review[benign] NOT soft_Block — MB-2
- All math must match math_brief_source.md — b7440be + 09320bc FINAL — history hard costs computed
- Commit hash final: 09320bc — previous chain b7440be→2f52c31→50943b5→d92f080→6f8942e→f0341ef→c248245→e6d512e→95292ce
- Before writing report section, run: python -c "import json; c=json.load(open('results/calculus_experiment.json')); print(len(c['optimized']['history']), c['optimized']['derivative_check']['gradient_descent_iterations'], c['optimized']['history'][-1]==c['optimized']['history'][-2])"
- Check prohibitions list PB-A to PB-D

---
End of Prohibitions — FIXED PB-A to PB-D + P-1 to P-5 + RB-1 to RB-7 + MB-1 to MB-7 + MB-A to MB-D — JSON+CODE EXACT — 09320bc FINAL

## PB-E — CRITICAL FINDING — Do NOT claim GD improved cost over grid winner

PROHIBITED:
- "GD optimisation reduced hard cost from 0.3999 to 0.3872"
- "GD improved on grid best"
- "Gradient descent achieved 3.16% improvement"

WHY: Grid best [0.3,0.45] hard cost 0.2829 is LOWER than returned [0.6,0.75] hard 0.3872 — GD increased hard cost +36.8% and smoothed +51% — improvement over baseline 0.3999→0.3872 came from grid search, NOT GD — GD is constrained refinement returning feasible boundary point.

CORRECT:
- "Combined grid+GD returns (0.60,0.75) hard 0.3872 — improves over baseline 0.3999 by 3.16% — decomposing: grid found (0.30,0.45) best grid point smoothed 0.2907 hard 0.2829 lowest found — during 29 GD iterations both smoothed (0.2907→0.4399) and hard (0.2829→0.3872) increased — GD ended at boundary-clipped feasible point — improvement over baseline attributable to grid search; GD is constrained refinement returning feasible boundary point but not improving on grid winner in hard cost — report returned point honestly as feasible point satisfying second-order conditions, not hard-cost minimiser."

