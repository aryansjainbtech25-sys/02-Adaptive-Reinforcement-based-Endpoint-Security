# Prohibitions — What NOT to Write in Report
## ARES CIE3 — Hard Rules for Teammates — Violations Fail Viva
### Commit: 95292ce → c248245 (EXACT JSON) → e6d512e, Source: results_brief.md + math_brief_source.md
### Canonical Source: results/calculus_experiment.json — file wins over brief if disagree

This document lists statements that must NOT appear in report v1.4 or viva_notes. Each prohibition has reason and correct alternative. All numbers must match results/calculus_experiment.json EXACT per verification output.

---

### P-1 — §3 numbers — FIXED to JSON EXACT — Do NOT quote stale 7.17/3.28/5.81/6.37/3.65/1.36

**PROHIBITED (OLD STALE NUMBERS):**
- "FNR actually increased 5.81%→6.37% (+0.56%). Only FPR decreased 7.17%→3.28% (-54%) and Review 3.65%→1.36% (-63%)."
- "FPR 7.17%→3.28%, FNR 5.81%→6.37%, Review 3.65%→1.36%"
- "baseline cost 0.387223 optimized 0.3676 delta -5.1%"

**WHY:** Those numbers do not match JSON. They are from old RF model or stale JSON. Canonical JSON values from verification:

baseline.cost = 0.39987259019990573
optimized.optimal_cost = 0.387223
baseline.rates.fpr = 0.06942277691107644 → 6.94%
baseline.rates.fnr = 0.062352012628255724 → 6.24%
baseline.rates.review_rate = 0.03737950029510132 → 3.74%
optimized.rates_at_optimum.fpr = 0.0343 → 3.43%
optimized.rates_at_optimum.fnr = 0.0681 → 6.81%
optimized.rates_at_optimum.review_rate = 0.0179 → 1.79%

**CORRECT (EXACT JSON):**
- "Baseline hard cost 0.39987259019990573 (0.35/0.70) → Optimized 0.387223 (0.60/0.75) delta -0.01264959019990573 (-3.16%)"
- "FPR reduced 6.94%→3.43% (-50.6%), Review 3.74%→1.79% (-52.1%), FNR increased 6.24%→6.81% (+9.2%) but overall hard cost fell -3.16%"
- "FPR 0.06942277691107644→0.0343, FNR 0.062352012628255724→0.0681, Review 0.03737950029510132→0.0179"
- Do NOT quote 0.3676, 0.387223 as baseline, -5.1%, 7.17%→3.28% — correct per JSON exact above

---

### 1. Do NOT call the reward policy RL

**PROHIBITED:**
- "We use Reinforcement Learning (RL) for threshold adaptation"
- "Our RL agent learns optimal thresholds via Q-learning"
- "RL policy with Bellman equation"
- "Deep RL / DQN for endpoint security"
- "Adaptive learning" as synonym for RL (dilution) — adaptive is fine, but do not imply RL

**WHY:** No MDP, no state transition, no Bellman, no Q-learning, no value function, no discount factor, no exploration. Policy is two scalars (t_r,t_b) adjusted by heuristic rewards.

**CORRECT:**
- "Feedback-driven heuristic threshold adaptation inspired by RL reward shaping, but not formal RL"
- "Heuristic policy with reward +1.0 correct, -1.5 false positive, -3.0 false negative"
- "Thresholds adjusted, classifier not retrained"
- "Adaptive" alone is fine for thresholds, but must clarify not RL — per P-2, do not let teammates dilute to "adaptive learning" meaning RL

---

### 2. Do NOT claim GD converged — FIXED per P-3

**PROHIBITED:**
- "Gradient descent converged to global optimum"
- "GD converged in 29 iterations" or "30 iterations converged"
- "Converged thresholds 0.60/0.75"

**WHY:** GD on smoothed surrogate C_smooth did NOT converge to interior zero-gradient. Result is boundary-clipped feasible point (see P-4 downgrade) on boundary (review_box_0.60 + gap_0.15), is_stationary=false, first_derivative_block=1.621676 !=0. We ran gradient_descent_iterations = 29 (from JSON derivative_check.gradient_descent_iterations), history length = 30 entries (indices 0-29). Verify with:

python -c "import json; c=json.load(open('results/calculus_experiment.json')); print('len(history) =', len(c['optimized']['history'])); print('gd_iter =', c['optimized']['derivative_check']['gradient_descent_iterations'])"

Actual from verification: len(history)=30, gd_iter=29. If JSON says 30, use 30 — file wins. Our current JSON says 29 iterations, 30 history entries (0-29). Do NOT claim 30 iterations if JSON says 29 — file wins.

**CORRECT:**
- "GD ran 29 iterations (30 history entries indices 0-29), reaching boundary-clipped feasible point at 0.60/0.75 on boundary"
- "hessian_det=6.923728>0 proves local minimum, but first derivative non-zero due to active constraints review_box_0.60+gap_0.15"
- "Grid verification confirms 0.60/0.75 minimal hard cost per JSON"
- Per P-3: Fix §2 of prohibitions and §1 of results_brief to match JSON len(history) and gradient_descent_iterations exactly

---

### 3. Do NOT claim false negatives decreased — FIXED per P-1

**PROHIBITED:**
- "Calculus optimization reduced false negatives"
- "FNR decreased"
- "Both FPR and FNR improved"

**WHY:** FNR actually increased 6.24%→6.81% (+9.2%) per JSON exact. Only FPR decreased 6.94%→3.43% (-50.6%) and Review 3.74%→1.79% (-52.1%). Cost decreased because FPR reduction outweighs FNR increase despite C_FN=5.0.

**CORRECT:**
- "FPR reduced 6.94%→3.43% (-50.6%), Review 3.74%→1.79% (-52.1%), FNR increased 6.24%→6.81% (+9.2%) but overall hard cost fell 0.39987259019990573→0.387223 (-3.16%)"
- Quote table exactly from results_brief.md EXACT JSON section

---

### 4. Do NOT quote precision_at_prevalence as if it reflects calculus-optimised thresholds

**PROHIBITED:**
- "Precision at prevalence 0.5 with optimized thresholds 0.60/0.75 is 98.27%"
- "Precision improved after calculus optimization"

**WHY:** Precision 98.27% is at classification cutoff 0.50, not at calculus thresholds 0.60/0.75. Calculus thresholds are for policy Allow/Review/Block, not binary classification.

**CORRECT:**
- "Precision 98.27% at 0.50 classification cutoff (baseline.json)"
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

**CORRECT:** "sklearn 1.9.1, numpy 2.5.4, Python 3.14.2, pytest-9.2.0 — evidence final_test_run.txt"

---

### 6. Do NOT invent dataset counts + P-5 legacy filename

**PROHIBITED:**
- "160064 total dataset" as final (raw before expansion)
- "4693 test URLs" (v2.0 RF, not v3.0 XGBoost)
- "820 benign" as final (raw before expansion to 7380)
- "7482 groups" as test only (total groups)

**WHY:** Correct per results_brief EXACT:
- Raw: 820 benign roots + 159244 phishing = 160064 rows (99.5% skew)
- Built: build_dataset(seed=42) → 22140 total, labels {1:14760 phishing, 0:7380 benign}, groups 7482 (5985 train + 1497 test)
- Split: GroupShuffleSplit 20% → test 5083 URLs, 1497 domains, prevalence 0.7478
- is_shortener 510/22140 (2.30%), tld_risk 213/22140 (0.96%)
- Confusion: TN 1218 FP 64 FN 156 TP 3645

**CORRECT:** Quote exactly from results_brief.md EXACT table.

**P-5 Addition — Legacy filename:**
- "The file name results/rl_policy_state.json is a legacy naming artefact from an earlier design. It stores two scalars (thresholds) and reward counters — not a learned policy. This must be mentioned in the report so an examiner does not infer RL from the filename."
- Add this to report §Adaptive Policy section

---

### P-4 — KKT Claim — DOWNGRADED per recommendation (a)

**PROHIBITED (RISKY):**
- "KKT constrained optimum" unless you verify multipliers

**WHY:** To claim KKT optimality, you must check:
Lagrangian: L = C_hard(t_r,t_b) + μ1(0.15-t_r) + μ2(t_r-0.60) + μ3(0.45-t_b) + μ4(t_b-0.95) + ν(0.15-(t_b-t_r))
Stationarity: ∇C_hard + Σμi∇gi + ν∇h =0
Primal feasibility: gi ≤0, h ≤0
Dual feasibility: μi≥0, ν≥0
Complementary slackness: μi gi=0, ν h=0

At t_r=0.60 (upper box bound active), need ∂C/∂t_r ≤0
At gap active, need check signs of both derivatives vs ν≥0
Your JSON shows first_derivative_block=1.621676>0 — under constraint t_b-t_r≥0.15, moving t_b up is feasible (up to 0.95), so ∂C/∂t_b>0 says cost increases if t_b rises — consistent with lower bound of feasible increase, but you have not verified multipliers non-negative.

**RECOMMENDATION (a) — SAFE:**
- Downgrade claim from "KKT constrained optimum" to "boundary-clipped feasible point where Hessian is positive-definite (hessian_det 6.923728>0) and hard cost is minimum over grid"
- This is honest and safe — you have earned right to be careful
- Do NOT let teammate write "KKT optimum" unless you verify multipliers yourself

**CORRECT (SAFE):**
- "Boundary-clipped feasible point at 0.60/0.75 — review at upper box bound 0.60 and gap at minimum 0.15 — Hessian det 6.923728>0 proves local minimum on feasible subspace, hard cost 0.387223 minimal over grid verification"
- If you want KKT, you must compute multipliers and show dual feasibility — real work, skip unless time

---

### Additional Prohibitions (from T1-T8)

- Do NOT claim scale_pos_weight >1 (actual 0.556 <1)
- Do NOT claim has_https proves trust (transport only, randomized p_strip=0.5)
- Do NOT claim QR decoding requires internet (offline verified grep empty)
- Do NOT claim RGBA white-on-transparent PASS (FAILs, inverted QR rare edge case, black-on-transparent PASS)
- Do NOT claim generate_qr_image is safety filter (testing utility, warning)
- Do NOT claim multi-QR returns single (fail-closed raises QRDecodeError)
- Do NOT claim 26 tests (now 42/42)
- Do NOT quote 0.3676, 0.387223 as baseline, -5.1%, 7.17%→3.28% — correct per JSON exact: baseline 0.39987259019990573 → 0.387223 -3.16%, FPR 0.06942277691107644→0.0343, hessian 6.923728

---

### Enforcement

- All numbers must match results/calculus_experiment.json EXACT — file wins over brief if disagree
- All numbers in results_brief.md §1 must be from verification output above
- All math must match math_brief_source.md
- Commit hash must be 95292ce692f25092e2381aea08f094f2326c3956 (or c248245 exact fix, e6d512e)
- Before writing report section, run: python -c "import json; c=json.load(open('results/calculus_experiment.json')); print(len(c['optimized']['history']), c['optimized']['derivative_check']['gradient_descent_iterations'])"
- Check prohibitions list

---
End of Prohibitions — FIXED per P-1 to P-5 — JSON EXACT
