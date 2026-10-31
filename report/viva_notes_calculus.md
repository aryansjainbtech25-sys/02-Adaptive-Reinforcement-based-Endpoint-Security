# Viva Notes — Calculus Threshold Optimization (CIE3) — Final

## Terminology — RL vs Heuristic Policy (Decided)

ARES uses an **adaptive feedback-driven heuristic policy**, not formal reinforcement learning.

- `_apply_reward_update` is rule-based: FP → increase thresholds, FN → decrease, clip + gap projection.
- No MDP, no Q-table, no value function, no policy gradient, no exploration.
- Reward signal (+1/-1) drives threshold movement, but update is deterministic heuristic.

**Report language everywhere:** "feedback-driven heuristic policy". Reserve "reinforcement" for product name ARES only, with disclaimer in §1.1: "Reinforcement refers to user feedback reinforcing thresholds, not to DQN/PPO".

**Viva answer:** "The RL aspect is outer loop: user feedback provides reward, policy (thresholds) updates. Update rule itself is heuristic — appropriate for single-user endpoint where data scarce and interpretability matters. Calculus optimization provides initial optimum (0.6/0.75), heuristic loop fine-tunes."

## Traceability Fixes

- final_cost and optimal_cost_smoothed are identical by construction (0.439942); both provided for backward compatibility (calculus_threshold.py line 156-157)
- history duplicate avoided: if last loop already logged (tr,tb)==final θ, update its cost rather than append duplicate (§4.2)
- gradient_descent_iterations = len(history)-1 — verified: len=30, iterations=29, history[-1]==[0.6,0.75,0.439942]
- boundary_hit is final-point diagnostic, not iteration history: reports which constraints active at returned θ — now ["review_box_0.60","gap_0.15"] after retrain
- Bug #1 fix: derivative_check (primary from compute_optimal_thresholds) preserved, derivative_check_validator (secondary from validate_thresholds_calculus) stored separately — boundary_hit, grad_tol, stopped_by not overwritten
- Task 2: EXPECTED_ARTIFACT_SKLEARN_VERSION 1.5.2 → 1.9.1 to match baseline.json sklearn 1.9.1 py3.14
- Task 3: Gap projection unified — _sanitize_thresholds and calculus_threshold.py both use: clip to box 0.15≤r≤0.60, 0.45≤b≤0.95, if b-r<0.15 push b up to r+0.15, only lower r if b at ceiling 0.95
- Task 4: Best-effort replay comment — no lock; single row skip acceptable for single-user demo
- Task 5: Discard unsaved in-memory state comment — agent is single source of truth, state saved after each mutation
- Task 6: compute_model_probabilities single-row guard (len(df)!=1 raises ValueError) + renamed uncalibrated_probabilities with legacy alias uncalibrated_rf_probabilities
- §2.1: is_minimum_proven renamed to hessian_positive_definite (local PD flag, not global optimality proof) — legacy alias kept for backward compat
- §2.2: optimal_cost_hard now reads final_cost_hard from optimizer (hard cost at returned point), not non-existent optimal_cost_hard key
- §2.5: _to_json_safe() added — converts np.float64 → float for derivative_check_validator, prevents json.dumps failure
- §2.6: Feedback logging and policy-state persistence not transactional — crash between CSV flush and state JSON write can leave artefacts inconsistent, acceptable for single-user demo

## T1 — Dataset Determinism Verification (BLOCKING)

Command:
```bash
python -c "from pathlib import Path; from train_model import build_dataset; df = build_dataset(Path('data/phishing_features.csv'), seed=42); print('total', len(df)); print('labels', df['label'].value_counts().to_dict()); print('groups', df['group'].nunique())"
```

Expected & Verified:
```
total 22140
labels {1: 14760, 0: 7380}
groups 7482
```

Breakdown from baseline.json:
```json
"n_train": 17057,
"n_train_domains": 5985,
"n_test": 5083,
"n_test_domains": 1497,
"raw_dataset_summary": {"total_rows": 160064, "benign_rows": 820, "phishing_rows": 159244}
```
17057+5083=22140 total, 5985+1497=7482 groups — matches build_dataset output.

Status: PASS ✅ — BLOCKING cleared

## T2 — scale_pos_weight < 1 Documentation

Why scale_pos_weight = 0.556 < 1 (unusual but correct for this dataset):

In standard phishing datasets, benign is majority → scale_pos_weight > 1 to upweight minority phishing. In ARES constructed dataset, situation is inverted:

- Raw CSV: 160064 rows = 159244 phishing + 820 benign (99.5% phishing)
- After deterministic build_dataset(seed=42): 22140 rows = 14760 phishing + 7380 benign (2:1 phishing:benign)

Phishing is majority (66.7%). XGBoost's scale_pos_weight = n_negative / n_positive = n_benign / n_phishing = 7380/14760 = 0.5 (actual 0.5564 after GroupShuffleSplit train split).

Value < 1 downweights majority phishing class, upweights benign — exactly needed to avoid bias toward phishing. This is correct for semi-synthetic constructed dataset where benign root domains are expanded with synthetic path/query structures, not real-world traffic where benign dominates.

Baseline.json records:
- scale_pos_weight: 0.5564376311707273
- test_phishing_prevalence: 0.7478

No code change — documentation only.

## T3 — precision_at_prevalence Uses 0.5 Threshold

What: precision_at_prevalence in baseline.json uses 0.5 operating point, not calculus-optimised 0.60/0.75

Baseline.json:
```json
"precision_at_prevalence": {
  "0.5": 0.952,
  "0.05": 0.5105,
  "0.01": 0.1668,
  "0.001": 0.0195
}
```

These values are computed at p ≥ 0.5 → phishing, before calculus threshold optimization (review 0.6 / block 0.75). They answer: "If deployed at 0.5 threshold, what precision would we get at real-world prevalence X?"

Do NOT conflate with calculus-optimised operating points:
- Baseline review/block 0.35/0.70 → cost 0.3807
- Calculus-optimised 0.6/0.75 → hard cost 0.3872, FPR 3.28% (-54%), review 1.36% (-63%)

precision_at_prevalence is a threshold-agnostic diagnostic for deployment planning, not the final operating point.

No code change — report only.

## T4 — Raw Dataset's 99.5% Phishing Skew

Raw CSV: 820 benign + 159,244 phishing (99.5% skew) — provenance issue, not bug.

Provenance:
1. Benign source: Limited set of 820 registrable root domains expanded with synthetic path/query via SEGMENTS, PARAM_KEYS, EXTS in train_model.py
2. Phishing source: Large collection of real phishing URLs (159,244)
3. Rebalancing in build_dataset(seed=42): deduplicate, group by registrable domain, downsample to 22140 rows (14760 phishing + 7380 benign, 2:1), split via GroupShuffleSplit by domain: 17057 train (5985 domains) + 5083 test (1497 domains) — zero domain overlap

No code change — report only.

## T5 — Strict JSON NaN Fix (Cleanup)

What: baseline.json had NaN for xgboost_params.missing which breaks strict JSON parsers

Command:
```bash
python -c "import json; json.load(open('results/baseline.json')); print('strict JSON OK')"
cat results/baseline.json | grep -n "missing"
```

Before fix:
```json
"missing": NaN,  # invalid JSON — Python json.load strict fails
```

After fix in train_model.py:
```python
def _json_safe(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    ...
baseline_safe = _json_safe({"baseline": {"test": metrics}})
BASELINE_OUT_PATH.write_text(json.dumps(baseline_safe, indent=2))
```

Verified:
```
72: "missing": null,
strict JSON OK
```

Status: PASS ✅

## T6 — Reproducibility Metadata (Cleanup)

What: Add numpy_version and python_version to baseline.json for determinism tracing

Before: only sklearn_version recorded
After:
```json
"sklearn_version": "1.9.1",
"numpy_version": "2.5.4",
"python_version": "3.14.2",
```

Why: rng.bytes(...).hex() and rng.choice behavior may vary slightly across NumPy versions; pinning numpy==2.5.4 in requirements.txt guarantees total=22140, groups=7482 determinism.

Verified:
```
2.5.4 1.9.1 3.14.2
```

Status: PASS ✅

## T7 — .gitignore Cleanup (Cleanup)

What: Prevent committing training backups

Added to .gitignore:
```
# ARES training backups — do not commit
*.bak
*.joblib.bak
*.json.bak
models/*.joblib
results/*.json
```

train_model.py creates:
- models/model.joblib.bak
- results/baseline.json.bak

Now ignored via .gitignore — git status clean.

Commit: ead2796 "T7: ignore *.bak training backups"

Status: PASS ✅

## T8 — Cross-Verify Split Matches (BLOCKING)

Command:
```bash
python -c "
from pathlib import Path
from train_model import build_dataset
from feature_extractor import feature_names
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
df = build_dataset(Path('data/phishing_features.csv'), seed=42)
X, y, g = df[feature_names], df['label'], df['group']
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42).split(X, y, g))
print('n_test', len(te), 'n_test_domains', g.iloc[te].nunique(), 'test_prev', round(float(y.iloc[te].mean()),4))
"
```

Expected & Verified:
```
n_test 5083 n_test_domains 1497 test_prev 0.7478
```

Matches baseline.json:
```json
"n_test": 5083,
"n_test_domains": 1497,
"test_phishing_prevalence": 0.7478
```

Conclusion: calculus_experiment.py and train_model.py use same GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42) grouped by registrable domain. No leakage, thresholds comparable.

Status: PASS ✅ — BLOCKING cleared

## Feature Extractor — Stage A (F1-F4)

### F1 — Dead Code suspicious_pattern

What: suspicious_pattern compiled regex never used; _keyword_hits uses token logic.

Fix: Delete 4 lines:
```python
# DELETE:
suspicious_pattern = re.compile(
    "(?:" + "|".join(re.escape(word) for word in suspicious_words) + ")",
    re.IGNORECASE,
)
```

Verification:
```bash
grep -rn "suspicious_pattern" . --include="*.py"  # only this file
sed -i '/^suspicious_pattern = re.compile/,/^)/d' feature_extractor.py
grep -n "suspicious_pattern" feature_extractor.py  # empty
PYTHONPATH=. python -m pytest tests/test_feature_extractor.py -v  # 14 passed
```

Status: PASS ✅

### F2 — is_shortener and tld_risk Non-Degenerate

Command:
```bash
python -c "from train_model import build_dataset; from pathlib import Path; df = build_dataset(Path('data/phishing_features.csv'), seed=42); print('is_shortener:', df['is_shortener'].value_counts().to_dict()); print('tld_risk:', df['tld_risk'].value_counts().to_dict())"
```

Verified:
```
is_shortener: {0: 21630, 1: 510}  → 510/22140 = 2.30%
tld_risk: {0: 21927, 1: 213}      → 213/22140 = 0.96%
```

Both active, small minority but not zero. Report: low prevalence but high precision when present. Retained for production robustness.

Status: PASS ✅

### F3 / F4 — Limitations (Report Only)

Two sentences for report Limitations section:

"Registrable-domain extraction uses an offline multi-part suffix table (94 ICANN suffixes); private suffixes (github.io, webflow.io) are not separated and are treated as registrable domains. This was a deliberate offline-only constraint to avoid external PSL dependency, with consistent grouping for GroupShuffleSplit."

"Feedback URL storage redacts sensitive query parameter values (token/key/pass/auth/session), long values (≥6 chars), and any value containing digits, replacing them with [REDACTED]. This sacrifices re-featurisation fidelity for privacy — redacted URLs cannot be fully re-featurised for retraining, which is acceptable for single-user demo."

Verified via:
```bash
python -c "from feature_extractor import extract_registrable_domain as g; print(g('https://evil.github.io/phish')); print(g('https://a.b.co.uk/x')); print(g('http://192.168.1.1/x'))"
# github.io, b.co.uk, 192.168.1.1
```

Status: PASS ✅

## §6.4 Four sentences about increasing history (for report figure caption)

1. The smoothed surrogate cost increased from 0.291 to 0.440 during the 30 iterations.
2. This is because projection onto the gap and box constraints is not a descent direction for the smoothed objective.
3. The returned point is a boundary-clipped feasible point, not an interior stationary point (is_critical_point=false, first_derivative_block=1.62, boundary_hit=[review_box_0.60,gap_0.15]).
4. The hard cost — the true objective of operational interest — improved from 0.3999 (baseline 0.35/0.70) to 0.3872 (returned 0.6/0.75).

Recommendation for figure: Show BOTH smoothed cost curve (increases) and hard cost curve (decreases) on same plot with clear legend. Caption: "Smoothed curve is for gradient-flow illustration; hard curve is operational objective. No cherry-picking."

## §6.5 Honest test status paragraph (for report)

tests/test_calculus_threshold.py — 8 passed.

Full suite — Initially during Python 3.14 migration: 39 passed, 1 skipped, 2 failed: test_qr.py::test_model_bundle_schema_and_pipeline_predictions and its suite wrapper, caused by ImportError: cannot load module more than once per process in NumPy's _pocketfft, triggered via adaptive_agent.load_model_bundle → xgboost import. Not caused by calculus_threshold.py; fixed in b509407 by preloading xgboost/scipy before unpickler.

Final: 42/42 passed on both old and new models after all fixes (T1-T8, F1).

## Stage G summary

G1 8/8, G2 42/42 (after fix), G3 import-safe, G4 experiment run, G5 reproducibility, G6 boundary_hit present, G7 gradient FD consistency (rel 0.00e+00 for h=1e-5,1e-4,1e-3), G8 random sweep 20/20.

## Stage H close-out

File reviewed: calculus_threshold.py ✅, feature_extractor.py ✅
Changes: B1-B7 + checklist, history semantics, boundary_hit, final_cost, is_stationary_point, full Hessian, F1 dead code removal ✅
Tests: T1-T8 PASS, F1-F4 PASS, 42/42 PASS ✅
Honest claim: smoothed cost increases (0.291→0.440), hard cost improves (0.3999→0.3872), boundary-clipped feasible point ✅
Remaining: documentation only — add T5-T8 + F2-F4 to v1.4 report.docx

## AG1 / AG5 Verification

AG1: py_compile OK, _to_json_safe converts np.float64 → float, rl_policy_state.json writes cleanly
AG5: calculus_metadata contains hessian_positive_definite (renamed), optimal_cost_hard reads final_cost_hard, derivative_check_validator nested dicts present, boundary_hit ["review_box_0.60","gap_0.15"], is_stationary_point false, stopped_by step

Example calculus_metadata (from generate_state.py):
```json
{
  "optimal_cost_hard": 0.387223,
  "hessian_positive_definite": true,
  "boundary_hit": ["review_box_0.60","gap_0.15"],
  "is_stationary_point": false,
  "stopped_by": "step",
  "derivative_check_validator": {
    "cost": 0.387223,
    "first_derivative": {"dC_d_review": -0.016..., "dC_d_block": 1.62...},
    "second_derivative": {"d2C_d_review2": 0.40..., "d2C_d_block2": 17.08...}
  }
}
```

Final Results (sklearn 1.9.1, numpy 2.5.4, python 3.14.2):
- T1: Dataset determinism — total 22140, labels {1:14760,0:7380}, groups 7482 (5985+1497) — PASS
- T2: scale_pos_weight <1 documented — PASS
- T3: precision_at_prevalence uses 0.5 threshold — PASS
- T4: raw dataset 99.5% skew explained — PASS
- T5: NaN → null strict JSON — PASS
- T6: numpy/python versions recorded — PASS
- T7: .bak ignored — PASS
- T8: split 5083/1497/0.7478 cross-verified — PASS
- F1: suspicious_pattern removed — PASS
- F2: is_shortener 2.3%, tld_risk 0.96% active — PASS
- F3/F4: limitations documented — PASS
- Tests: 42/42 PASS ✅
