# Prohibitions — What NOT to Write in Report
## ARES CIE3 — Hard Rules for Teammates — Violations Fail Viva
### Commit: 95292ce → 9f1a792, Source: results_brief.md + math_brief_source.md

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
- "Thresholds adjusted, classifier not retrained"

---

### 2. Do NOT claim GD converged

**PROHIBITED:**
- "Gradient descent converged to global optimum"
- "GD converged in 29 iterations"
- "Converged thresholds 0.60/0.75"

**WHY:** GD on smoothed surrogate C_smooth did NOT converge to interior zero-gradient. Result is KKT constrained optimum on boundary (review_box_0.60 + gap_0.15), is_stationary=false, first_derivative_block=1.62 !=0. We stopped after 29 iterations, history 30 entries. Not converged, but sufficient to reach near grid optimum.

**CORRECT:**
- "GD ran 29 iterations (30 history entries), reaching KKT constrained optimum at 0.60/0.75 on boundary"
- "hessian_det=6.9237>0 proves local minimum, but first derivative non-zero due to active constraints"
- "Grid verification confirms 0.60/0.75 minimal hard cost"

---

### 3. Do NOT claim false negatives decreased

**PROHIBITED:**
- "Calculus optimization reduced false negatives"
- "FNR decreased from 5.81% to 6.37%"
- "Both FPR and FNR improved"

**WHY:** FNR actually increased 5.81%→6.37% (+0.56%). Only FPR decreased 7.17%→3.28% (-54%) and Review 3.65%→1.36% (-63%). Cost decreased because C_FN=5.0 but FPR reduction outweighs FNR increase.

**CORRECT:**
- "FPR reduced 7.17%→3.28% (-54%), Review 3.65%→1.36% (-63%), FNR slightly increased 5.81%→6.37% (+0.56%) but overall hard cost fell 0.387223→0.3676 (-5.1%)"
- Quote table exactly from results_brief.md

---

### 4. Do NOT quote precision_at_prevalence as if it reflects calculus-optimised thresholds

**PROHIBITED:**
- "Precision at prevalence 0.5 with optimized thresholds 0.60/0.75 is 98.27%"
- "Precision improved after calculus optimization"

**WHY:** Precision 98.27% is at classification cutoff 0.50, not at calculus thresholds 0.60/0.75. Calculus thresholds are for policy Allow/Review/Block, not for binary classification. Precision_at_prevalence 0.5 vs 0.6/0.75 confusion — baseline metrics at 0.50 cutoff cannot be mixed with policy thresholds.

**CORRECT:**
- "Precision 98.27% at 0.50 classification cutoff (baseline.json)"
- "Policy thresholds 0.60/0.75 govern Allow/Review/Block, not classification — cost evaluated separately"
- Keep baseline table (0.50 cutoff) and calculus table (0.60/0.75) separate, do not cross-quote

---

### 5. Do NOT invent sklearn/Numpy/Python versions

**PROHIBITED:**
- "sklearn 1.5.2" (old RF artifact)
- "numpy 1.26.0"
- "Python 3.10/3.11"
- Any version not in final_test_run.txt or baseline.json

**WHY:** Actual versions from evidence:
- sklearn 1.9.1 (bundle features == feature_names validated, Task 7 PASS)
- numpy 2.5.4
- Python 3.14.2 (from final_test_run.txt platform linux Python 3.14.2 pytest-9.2.0)
- joblib_sklearn_1.9.1 loader

**CORRECT:** Quote exactly:
- "sklearn 1.9.1, numpy 2.5.4, Python 3.14.2, pytest-9.2.0 — evidence final_test_run.txt"
- "Bundle loader joblib_sklearn_1.9.1"

---

### 6. Do NOT invent dataset counts

**PROHIBITED:**
- "160064 total dataset" as final (that's raw before expansion)
- "4693 test URLs" (that's v2.0 RF, not v3.0 XGBoost)
- "820 benign" as final count (raw, before expansion to 7380)
- "7482 groups" as test only (it's total groups)
- Any count not in results_brief.md

**WHY:** Correct counts from evidence:
- Raw: 820 benign roots + 159244 phishing = 160064 rows (99.5% skew)
- Built: build_dataset(seed=42) → 22140 total, labels {1:14760 phishing, 0:7380 benign}, groups 7482 (5985 train + 1497 test)
- Split: GroupShuffleSplit 20% → test 5083 URLs, 1497 domains, prevalence 0.7478
- Features: 24 fixed order, is_shortener 510/22140 (2.30%), tld_risk 213/22140 (0.96%)
- Confusion: TN 1218 FP 64 FN 156 TP 3645

**CORRECT:** Quote exactly from results_brief.md table, do not mix raw vs built vs test counts.

---

### Additional Prohibitions (from T1-T8)

- Do NOT claim scale_pos_weight >1 (actual 0.556 <1)
- Do NOT claim has_https proves trust (it's transport only, randomized p_strip=0.5)
- Do NOT claim QR decoding requires internet (offline verified grep empty)
- Do NOT claim RGBA white-on-transparent PASS (it FAILs, inverted QR rare edge case, black-on-transparent PASS)
- Do NOT claim generate_qr_image is safety filter (it's testing utility, warning documented)
- Do NOT claim multi-QR returns single (fail-closed: raises QRDecodeError, UI prompts choice)
- Do NOT claim 26 tests (now 42/42)
- Do NOT quote 0.3676, 0.387223 baseline, -5.1%, 7.17%→3.28% — correct per JSON: baseline 0.39987259019990573 → 0.387223 -3.16%, FPR 0.06942277691107644→0.0343, hessian 6.923728, first_derivative_block 1.621676

---

### Enforcement

- All numbers must match results_brief.md exactly
- All math must match math_brief_source.md exactly
- Commit hash must be 95292ce692f25092e2381aea08f094f2326c3956 (or 30f90f5 evidence, 9f1a792 final)
- Before writing report section, check this prohibitions list

---
End of Prohibitions
- Do NOT quote 0.3676, 0.387223 as baseline, -5.1% — correct: baseline 0.39987259019990573 → 0.387223 -3.16% per JSON exact
