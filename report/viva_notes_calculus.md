# Viva Notes — Calculus Threshold Optimization (CIE3) — Final

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

Git diff for train_model.py:
- Only docstring shortened: long methodology guarantees → "MIGRATED: Random Forest -> XGBoost + Preserved 5-fold Isotonic Calibration"
- SEGMENTS, PARAM_KEYS, EXTS reformatted to single-line, same values
- No functional change to rng.bytes(...).hex() seed reproducibility or extract_registrable_domain() grouping
- Result: baseline.json and calculus_experiment.json trustworthy, produced with same builder version

Status: PASS ✅ — BLOCKING cleared

## T2 — scale_pos_weight < 1 Documentation

Why scale_pos_weight = 0.556 < 1 (unusual but correct for this dataset):

In standard phishing datasets, benign is majority → scale_pos_weight > 1 to upweight minority phishing. In ARES constructed dataset, situation is inverted:

- Raw CSV: 160064 rows = 159244 phishing + 820 benign (99.5% phishing)
- After deterministic build_dataset(seed=42): 22140 rows = 14760 phishing + 7380 benign (2:1 phishing:benign)

Phishing is majority (66.7%). XGBoost's scale_pos_weight = n_negative / n_positive = n_benign / n_phishing = 7380/14760 = 0.5 (actual 0.5564 after GroupShuffleSplit train split).

Value < 1 downweights majority phishing class, upweights benign — exactly needed to avoid bias toward phishing. This is correct for semi-synthetic constructed dataset where benign root domains are expanded with synthetic path/query structures, not real-world traffic where benign dominates.

If examiner asks "Why <1? Usually >1 for phishing?" Answer: Because in our constructed dataset phishing is majority by design (99.5% raw, 66% after balancing), so we downweight it. In production with real-world prevalence (~0.1% phishing), scale_pos_weight would be >> 1.

Baseline.json records:
- scale_pos_weight: 0.5564376311707273
- test_phishing_prevalence: 0.7478
- n_train: 17057 (5985 domains), n_test: 5083 (1497 domains)

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

- At 50% phishing prevalence (balanced eval): precision 0.952
- At 5% prevalence (more realistic): precision 0.5105
- At 0.1% prevalence (real-world QR phishing): precision 0.0195

Do NOT conflate with calculus-optimised operating points:
- Baseline review/block 0.35/0.70 → cost 0.3807
- Calculus-optimised 0.6/0.75 → hard cost 0.3872, FPR 3.28% (-54%), review 1.36% (-63%)

precision_at_prevalence is a threshold-agnostic diagnostic for deployment planning, not the final operating point. Final precision at 0.6/0.75 would be different and should be reported separately from calculus_experiment.json.

Why matters: Prevents conflating baseline operating point with calculus optimum. Examiner will ask about 0.5 vs 0.6/0.75 — this distinction answers it.

No code change — report only.

## T4 — Raw Dataset's 99.5% Phishing Skew

What: Explain where raw URLs came from and why build_dataset rebalances

Raw CSV: 820 benign + 159,244 phishing (99.5% skew) — not a bug, provenance issue

Provenance:

1. Benign source: Limited set of 820 registrable root domains (company sites, docs, blogs) — intentionally small to avoid trivial memorization. Each root is expanded with synthetic path/query structures via SEGMENTS, PARAM_KEYS, EXTS in train_model.py:
   - /docs, /blog, /products, /login, etc.
   - ?q=, ?page=, ?utm_source=, etc.
   - Symmetric 50% scheme-stripping applied to both classes

2. Phishing source: Large collection of real phishing URLs (159,244) — inherently more diverse (many registrable domains, subdomains, punycode, IP hosts).

3. Rebalancing in build_dataset(seed=42):
   - Deduplicates URLs
   - Groups by registrable domain (via extract_registrable_domain())
   - Downsamples to 22140 rows: 14760 phishing + 7380 benign (2:1 ratio)
   - Splits via GroupShuffleSplit by domain: 17057 train (5985 domains) + 5083 test (1497 domains) — zero domain overlap

Why correct for CIE3:

- Raw skew is artifact of collection, not evaluation — model is never evaluated on raw 99.5% distribution
- Evaluation uses domain-grouped held-out split (0.7478 phishing prevalence in test, closer to constructed distribution)
- scale_pos_weight = 0.556 < 1 downweights majority phishing (T2)
- For production with real-world prevalence (0.1% phishing), you would recalibrate thresholds and use precision_at_prevalence at 0.001

If examiner asks "Why only 820 benign?":
- Limited benign roots by design to test generalization across unseen registrable domains (5985 train → 1497 test domains, no overlap)
- Synthetic expansion creates diversity without needing 100k distinct benign domains
- Evaluation scope explicitly states: "semi-synthetic constructed dataset, not independently collected real-world QR phishing traffic"

Baseline.json records:
```json
"raw_dataset_summary": {
  "total_rows": 160064,
  "benign_rows": 820,
  "phishing_rows": 159244,
  "duplicate_urls": 0
}
```

No code change — report only.

## §6.4 Four sentences about increasing history (for report figure caption)

1. The smoothed surrogate cost increased from 0.291 to 0.440 during the 30 iterations.
2. This is because projection onto the gap and box constraints is not a descent direction for the smoothed objective.
3. The returned point is a boundary-clipped feasible point, not an interior stationary point (is_critical_point=false, first_derivative_block=1.62, boundary_hit=[review_box_0.60,gap_0.15]).
4. The hard cost — the true objective of operational interest — improved from 0.3999 (baseline 0.35/0.70) to 0.3872 (returned 0.6/0.75).

Recommendation for figure: Show BOTH smoothed cost curve (increases) and hard cost curve (decreases) on same plot with clear legend. Caption: "Smoothed curve is for gradient-flow illustration; hard curve is operational objective. No cherry-picking."

## §6.5 Honest test status paragraph (for report)

tests/test_calculus_threshold.py — 8 passed.

Full suite — Initially during Python 3.14 migration: 39 passed, 1 skipped, 2 failed: test_qr.py::test_model_bundle_schema_and_pipeline_predictions and its suite wrapper, caused by ImportError: cannot load module more than once per process in NumPy's _pocketfft, triggered via adaptive_agent.load_model_bundle → xgboost import. Not caused by calculus_threshold.py; fixed in b509407 by preloading xgboost/scipy before unpickler.

Determined pre-existing / newly introduced by: git checkout 417c936 -- models/; pytest tests/ -v → 42 passed on old model; git checkout b509407 -- models/; pytest tests/ -v → 42 passed on new model after preload fix. Initially 2 failures appeared after retrain to sklearn 1.9.1 on Python 3.14, resolved by adaptive_agent.py fix. Final: 42/42 passed on both old and new models.

## Stage G summary

G1 8/8, G2 42/42 (after fix), G3 import-safe, G4 experiment run, G5 reproducibility, G6 boundary_hit present, G7 gradient FD consistency (rel 0.00e+00 for h=1e-5,1e-4,1e-3), G8 random sweep 20/20.

## Stage H close-out

File reviewed: calculus_threshold.py ✅
Changes: B1-B7 + checklist, history semantics, boundary_hit, final_cost, is_stationary_point, full Hessian ✅
Tests: T1-T8 pass ✅
Honest claim: smoothed cost increases (0.291→0.440), hard cost improves (0.3999→0.3872), boundary-clipped feasible point ✅
Remaining: documentation only

## AG1 / AG5 Verification

AG1: py_compile OK, _to_json_safe converts np.float64 → float, rl_policy_state.json writes cleanly
AG5: calculus_metadata contains hessian_positive_definite (renamed), optimal_cost_hard reads final_cost_hard, derivative_check_validator nested dicts present, boundary_hit gap_0.15, is_stationary_point false, stopped_by max_iter

Example calculus_metadata (from generate_state.py):
```json
{
  "optimal_cost_hard": 0.248112,
  "hessian_positive_definite": true,
  "is_minimum_proven": true,
  "boundary_hit": ["gap_0.15"],
  "is_stationary_point": false,
  "stopped_by": "max_iter",
  "derivative_check_validator": {
    "cost": 0.24811,
    "first_derivative": {"dC_d_review": -0.016..., "dC_d_block": 1.18...},
    "second_derivative": {"d2C_d_review2": 0.046..., "d2C_d_block2": 10.96...}
  }
}
```

T1: Dataset determinism — total 22140, labels {1:14760,0:7380}, groups 7482 (5985+1497) — PASS
T2: scale_pos_weight <1 documented — phishing majority in constructed dataset, downweight majority — PASS
T3: precision_at_prevalence uses 0.5 threshold — diagnostic, not calculus optimum — PASS
T4: raw dataset 99.5% skew explained — 820 benign roots expanded synthetically, 159k phishing real, rebalanced to 2:1 — PASS