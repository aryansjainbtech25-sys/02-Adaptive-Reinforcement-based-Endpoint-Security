Final project commit: 5fdc189aac39267cb96f04f4f3e409535be656a5 (short: 5fdc189).

# Prohibitions — What NOT to Write in Report
### Previous chain: b7440be -> 0b3cc3b -> d66e2a4 -> 09320bc -> 5fdc189 — len=30 gd=29 duplicate False
### All numbers in this report are traceable to results/calculus_experiment.json and results/baseline.json at commit 5fdc189aac39267cb96f04f4f3e409535be656a5. Test evidence: results/final_test_run.txt (42/42 pass, Python 3.14.2, pytest 9.2.0).

This document lists statements that must NOT appear in report v1.4 or viva_notes. Each prohibition has reason and correct alternative.

---

### 1. Do NOT call the reward policy RL

**PROHIBITED:**
- "We use Reinforcement Learning (RL) for threshold adaptation"
- "Our RL agent learns optimal thresholds via Q-learning"
- "RL policy with Bellman equation"
- "Deep RL / DQN for endpoint security"

**WHY:** No MDP, no state transition, no Bellman, no Q-learning, no value function, no discount factor, no exploration. Policy is two scalars (t_r,t_b) adjusted by heuristic rewards.

**CORRECT:**
- "Feedback-driven heuristic threshold adaptation inspired by RL reward shaping, but not formal RL"
- "Heuristic policy with reward +1.0 correct, -1.5 false positive, -3.0 false negative"
- "File name rl_policy_state.json is legacy artefact from earlier design"

---

### 2. Do NOT claim GD converged

**PROHIBITED:**
- "Gradient descent converged to global optimum"
- "GD converged in 29 iterations"
- "Converged thresholds 0.60/0.75"

**WHY:** GD on smoothed surrogate C_smooth did NOT converge to interior zero-gradient. Result is boundary-clipped feasible point on boundary (review_box_0.60 + gap_0.15), is_stationary=false, first_derivative_block=1.621676 !=0. Verified len=30 gd=29 duplicate False.

**CORRECT:**
- "GD ran 29 iterations (30 history entries indices 0-29), reaching boundary-clipped feasible point at 0.60/0.75"
- "hessian_det=6.923728>0 and trace 17.49187>0 prove smoothed surrogate locally strictly convex — necessary condition, not proof of constrained optimality"
- "Combined grid + gradient descent procedure finds 0.60/0.75 as minimum hard cost returned — NOT grid only"

---

### 3. Do NOT claim false negatives decreased

**PROHIBITED:**
- "Calculus optimization reduced false negatives"
- "FNR decreased from 5.81% to 6.37%"
- "Both FPR and FNR improved"

**WHY:** FNR actually increased. Verified:

| Point | C_hard | C_smooth | Note |
|-------|--------|----------|------|
| Baseline [0.35,0.70] | 0.39987259019990573 | — | reference |
| Grid winner [0.30,0.45] | 0.2829383159770988 | 0.29071968134044274 | lowest hard cost found |
| Returned [0.60,0.75] | 0.38722259534246795 | 0.43994209280287067 | GD boundary-clipped point |

FPR reduced 0.06942277691107644 (6.94%)→0.0343 (3.43%) -50.6%, Review 0.03737950029510132 (3.74%)→0.0179 (1.79%) -52.1%, FNR increased 0.062352012628255724 (6.24%)→0.0681 (6.81%) +9.22%

Cost decreased 0.39987259019990573→0.387223 -3.16% because FPR reduction outweighs FNR increase despite C_FN=5.0, but improvement attributable to grid search, NOT GD.

**CORRECT:**
- Use three-row table above — cannot be misread
- "FPR reduced 6.94%→3.43% (-50.6%), Review 3.74%→1.79% (-52.1%), FNR slightly increased 6.24%→6.81% (+9.22%) but overall hard cost fell 0.3998→0.3872 (-3.16%) attributable to grid search"

---

### 4. Do NOT quote precision_at_prevalence as if it reflects calculus-optimised thresholds

**PROHIBITED:**
- "Precision at prevalence 0.5 with optimized thresholds 0.60/0.75 is 98.27%"

**WHY:** Precision 0.9833 at 0.50 cutoff per d92f080 baseline.json verified, NOT at calculus thresholds.

**CORRECT:**
- "Precision 0.9833 at 0.50 classification cutoff (baseline.json)"
- "Policy thresholds 0.60/0.75 govern Allow/Review/Block, not classification"

---

### 5. Do NOT invent sklearn/Numpy/Python versions

**PROHIBITED:** "sklearn 1.5.2", "numpy 1.26.0", "Python 3.10/3.11"

**WHY:** Actual: sklearn 1.9.1, numpy 2.5.4, Python 3.14.2, pytest-9.2.0 — evidence final_test_run.txt

**CORRECT:** "sklearn 1.9.1, numpy 2.5.4, Python 3.14.2, pytest-9.2.0"

---

### 6. Do NOT invent dataset counts + PB-D rl_policy_state.json

**PROHIBITED:** "160064 total dataset" as final, "4693 test URLs", "820 benign" as final, "7482 groups" as test only

**WHY:** Correct: Raw 820 benign roots + 159244 phishing = 160064, Built 22140 total (14760 phishing, 7380 benign), groups 7482 (5985 train + 1497 test), test 5083 URLs, 1497 domains, prevalence 0.7478, is_shortener 510/22140 (2.30%), tld_risk 213/22140 (0.96%), confusion TN 1220 FP 62 FN 158 TP 3643 per d92f080

**PB-D:** results/rl_policy_state.json may show different thresholds than results/calculus_experiment.json because app writes state on feedback. Report must say: "Thresholds stored in results/rl_policy_state.json reflect current runtime state and may differ from calculus-optimised (0.60,0.75) if feedback recorded. Canonical source for reported thresholds is results/calculus_experiment.json." Do NOT quote rl_policy_state.json thresholds as calculus result.

---

### PB-E — B-5 FINAL — Do NOT claim calculus optimization reduced hard cost without grid vs GD caveat — FINAL 5fdc189aac39267cb96f04f4f3e409535be656a5

**PROHIBITED:**
- "Calculus optimization reduced the hard cost"
- "GD optimisation reduced hard cost from 0.3999 to 0.3872"
- "Gradient descent achieved 3.16% improvement"
- "Calculus optimization reduced hard cost from 0.3999 to 0.3872"

**WHY:** Grid best [0.3,0.45] hard cost 0.2829 is LOWER than returned [0.6,0.75] hard 0.3872 — GD increased hard cost from 0.2829 (grid winner) to 0.3872 (returned) — relative increase 36.87% (0.3872/0.2829-1), and smoothed cost from 0.2907 to 0.4399 — relative increase 51.36% (0.4399/0.2907-1). Neither metric improved during GD. Improvement over baseline 0.3999→0.3872 -3.16% came from grid search, NOT GD.

Three-row table — use exactly:

| Point | C_hard | C_smooth | Note |
|-------|--------|----------|------|
| Baseline [0.35,0.70] | 0.39987259019990573 | — | reference |
| Grid winner [0.30,0.45] | 0.2829383159770988 | 0.29071968134044274 | lowest hard cost found |
| Returned [0.60,0.75] | 0.38722259534246795 | 0.43994209280287067 | GD boundary-clipped point |

**CORRECT — HONEST:**
- "Combined grid+GD returns (0.60,0.75) hard 0.3872 — improves over baseline 0.3999 by 3.16% — improvement attributable to grid search, NOT GD — GD is constrained refinement returning feasible boundary point satisfying second-order conditions (hessian_det 6.923728>0, trace 17.49>0) but not improving on grid winner in hard cost (grid winner 0.30,0.45 hard 0.2829 lowest)."
- "GD step increased hard cost from 0.2829 (grid winner [0.30,0.45]) to 0.3872 (returned [0.60,0.75]) — relative increase 36.87%. GD step increased smoothed cost from 0.2907 to 0.4399 — relative increase 51.36%. Neither metric improved during GD."
- "Do NOT say 'calculus optimization reduced the hard cost' without acknowledging that grid-search contribution dominates and GD step increased hard cost."

---

### Additional Prohibitions

- Do NOT claim scale_pos_weight >1 (actual 0.556 <1)
- Do NOT claim has_https proves trust (transport only, randomized p_strip=0.5)
- Do NOT claim QR decoding requires internet (offline verified)
- Do NOT claim RGBA white-on-transparent PASS (FAILs)
- Do NOT claim generate_qr_image is safety filter
- Do NOT claim multi-QR returns single (fail-closed raises QRDecodeError)
- Do NOT claim 26 tests (now 42/42)
- Do NOT quote 0.3676, -5.1%, 7.17%→3.28%, 5.81%→6.37% — correct per JSON exact: baseline 0.39987259019990573 → 0.387223 -3.16%, FPR 0.06942277691107644→0.0343, FNR 0.062352012628255724→0.0681, Review 0.03737950029510132→0.0179, hessian 6.923728, second_deriv 0.405678/17.086192 trace 17.49187, final_cost 0.43994209280287067, history[0] 0.2829383159770988/0.29071968134044274, history[-1] 0.38722259534246795/0.43994209280287067
- Do NOT claim C_smooth≈0.42 C_hard≈0.51 at iteration 0 — correct C_smooth 0.29071968134044274 C_hard 0.2829383159770988
- Do NOT claim history[-1]==history[-2] duplicate — duplicate False fixed
- Do NOT quote rl_policy_state.json thresholds as calculus result

---

### Enforcement

- All numbers must match results/calculus_experiment.json EXACT — file wins — verified len=30 gd=29 duplicate False — 5fdc189aac39267cb96f04f4f3e409535be656a5
- All numbers must match results/baseline.json EXACT — file wins — verified roc_auc 0.9931 pr_auc 0.9977 brier 0.0302 confusion [[1220,62],[158,3643]]
- All formulas must match src/calculus/calculus_threshold.py EXACT — file wins — FPR~ mean s_review[benign] NOT soft_Block
- All math must match math_brief_source.md — FINAL 5fdc189aac39267cb96f04f4f3e409535be656a5 — history hard costs computed
- Commit hash final: 5fdc189aac39267cb96f04f4f3e409535be656a5
- Before writing report section, run: python -c 'import json; c=json.load(open("results/calculus_experiment.json")); print(len(c["optimized"]["history"]), c["optimized"]["derivative_check"]["gradient_descent_iterations"], c["optimized"]["history"][-1]==c["optimized"]["history"][-2])'
- Expected: 30 29 False

---
End of Prohibitions — FINAL 5fdc189aac39267cb96f04f4f3e409535be656a5 — JSON+CODE EXACT — HONEST
