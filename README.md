# ARES — Adaptive Reinforcement-based Endpoint Security
### Offline QR-Code Phishing (Quishing) Risk Analyzer with Calibrated XGBoost & Calculus-Optimized Threshold Policy

ARES is an educational and research prototype developed at **R V University** that evaluates URLs embedded inside QR codes (**quishing**) using offline lexical and structural analysis. It decodes QR images without visiting or fetching the destination URL, extracts **24 lexical, host, and structural URL features**, estimates phishing probability using a **5-fold Isotonic-Calibrated XGBoost (`XGBClassifier` + `CalibratedClassifierCV`)** — migrated from Random Forest in v3.0 — and recommends an advisory endpoint action (`Allow`, `Review`, or `Block`) governed by a **feedback-driven + calculus-optimized threshold controller**.

> **Important Security Disclaimer:** ARES is an offline lexical URL analyzer, **not** a full Endpoint Detection and Response (EDR) agent, browser sandbox, or live threat-intelligence scanner. A low estimated phishing probability (`Allow` recommendation) **does not guarantee** that a website is safe, and `https://` indicates encrypted network transport rather than domain legitimacy. Never use ARES as the sole basis for trusting an unknown link.

## Contributors

Developed as part of an AIML Semester 3 academic project at **R V University**:

| Sl. No. | Name | USN | Role |
|---:|---|---:|---|
| 1 | Aryan S Jain | `1RUA25CSE0087` | XGBoost Migration + Calculus Module + Adaptive Agent |
| 2 | Ashwin Chandrashekar Naik | `1RUA25CSE0090` | Feature Extractor + QR Utils |
| 3 | Ashwin Rajesha Bhat | `1RUA25CSE0091` | Streamlit App + UI/UX |
| 4 | Ashwin Subramaniam | `1RUA25CSE0092` | Experiments, Baseline Metrics & Report |

## Application Screenshot

<p align="center">
  <img src="ares-dashboard.png" alt="ARES dashboard showing QR-code phishing detection, sample QR gallery, and adaptive threshold policy" width="950"/>
</p>
<p align="center"><em>ARES dashboard — QR-code phishing detection and adaptive risk assessment.</em></p>

---

## What's New in v3.0 — XGBoost + Calculus Migration (AIML Semester 3)

- **Model Migration:** Random Forest (100 trees, `scikit-learn==1.9.1`) → **XGBoost (`XGBClassifier`, 400 estimators, max_depth=8, learning_rate=0.05, subsample=0.9, colsample=0.9, scale_pos_weight balanced)** with `tree_method=hist`. ROC-AUC **0.9902 → 0.9929**, PR-AUC **0.9953 → 0.9976**, Brier **0.0353 → 0.0297**, FPR **7.24% → 4.99%**.
- **Calibration Preserved:** 5-fold `GroupKFold` Isotonic Calibration, `GroupShuffleSplit` by registrable domain (zero leakage verified via `verify_no_group_leakage`), dual probability reporting (calibrated + uncalibrated base XGB).
- **Calculus Component (NEW — `calculus_threshold.py`):** Defines expected cost `C(t_r,t_b)=C_FP·FPR + C_FN·FNR + C_review·ReviewRate + λ||t-t_0||²`. Sigmoid-smoothed surrogate `FPR̃=mean(sigmoid((p-θ)/τ))` fixes stepwise derivative issue: `d/dθ sigmoid = -s(1-s)/τ ≠0`. Computes first derivative `dC/dt≈[C(t+ε)-C(t-ε)]/(2ε)`, second derivative `d²C/dt²≈[C(t+ε)-2C(t)+C(t-ε)]/ε²`. Optimum at `dC/dt=0`, verified `d²C/dt²>0`. Real experiment on **5083 held-out URLs**: Baseline **0.35/0.70** cost **0.3872** → Optimized **0.60/0.75** cost **0.3676** (**-5.1%**), **FPR 7.17%→3.28% (-54%)**, **Review 3.65%→1.36% (-63%)**.
- **New Files:** `calculus_threshold.py`, `experiments/calculus_experiment.py`, `results/calculus_experiment.json`

## System Architecture & Key Capabilities

1. **Bomb-Safe, Multi-Variant QR Decoding (`qr_utils.py`):**
   - Verifies image dimensions from headers before decoding and rejects images exceeding `25,000,000` pixels.
   - Composites transparent RGBA PNGs onto white background for dark modules readability.
   - Sweeps native resolution (`scale=1`, including `3000px` boundary with `20px` quiet-zone), nearest-neighbor upscaling `2x–10x` for small `21–69px` dataset matrices, Otsu/adaptive binarization, inverted polarities.
   - Detects multiple QR codes (`detectAndDecodeMulti`): fails closed in CLI mode and prompts explicit choice in Streamlit UI.

2. **Centralized URL Validation & Domain Parsing (`feature_extractor.py`):**
   - Single canonical parser (`canonical_split`, `parse_host_structure`, `extract_registrable_domain`) across validation, feature extraction, UI anatomy, and domain-grouped training.
   - Normalizes backslashes `\`, strips control characters, supports IPv4/IPv6 literals, explicit ports, IDN Punycode `xn--`, rejects plain-text `Hello ARES` or non-HTTP schemes `javascript:`, `mailto:`, `WIFI:`.

3. **Dual Probability Reporting (`adaptive_agent.py`):**
   - Computes both **Isotonic-Calibrated Phishing Probability** (mean of 5 calibrated folds) and **Uncalibrated Base XGBoost Probability** (mean of 5 uncalibrated base fold estimators).
   - Validates `model.classes_ [0,1]`, feature order `24` features, finite probabilities summing to `1.0`.
   - Includes pure-NumPy engine `PureNumpyCalibratedClassifier` for backwards compatibility, native XGBoost unpickler allowing `xgboost.sklearn.XGBClassifier`. Handles version skew (artifact `1.5.2` for XGB).

4. **Feedback-Driven + Calculus-Optimized Threshold Policy (`adaptive_agent.py` + `calculus_threshold.py`):**
   - **Two Policies:** (1) Heuristic RL: Review default `35%`, Block `70%`, reward `+1.0` correct, `-1.5` FP, `-3.0` FN, file locking + atomic `.tmp` replacement. (2) **Calculus GD:** `θ_{k+1}=θ_k - η∇J̃`, `η=0.05, τ=0.05`, grid init + 29 iterations, `dC/dt≈0, d²C/dt²>0` proof, KKT boundary optimum handling.
   - Redacts sensitive query params (`token`, `session`, `password`, `key`) before `results/feedback.csv`, rejects duplicate/contradictory verdicts.
   - User feedback adjusts thresholds only, does NOT retrain classifier.

## Requirements & Compatibility

- **Python:** `3.10`, `3.11`, or `3.12` recommended.
- **Dependencies (`requirements.txt`):**
  ```
  pandas>=2.0.0
  numpy>=1.26.0
  scikit-learn==1.5.2 # matches models/model.joblib — was 1.9.1 for RF v2.0
  xgboost==3.0.2 # NEW in v3.0
  joblib>=1.3.0
  opencv-python-headless>=4.8.0
  Pillow>=10.0.0
  streamlit>=1.38.0
  pytest>=8.0.0
  matplotlib>=3.8.0
  ```

## Installation & Launch

### Option A: One-Command Launcher
```cmd
run.bat  # Windows — checks python + models/model.joblib, creates .venv, installs, compiles, runs streamlit
chmod +x run.sh
./run.sh # macOS/Linux
```

### Option B: Manual
```bash
git clone https://github.com/<username>/ARES.git
cd ARES
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m py_compile app.py adaptive_agent.py feature_extractor.py calculus_threshold.py qr_utils.py train_model.py

# Tests — expect 26 passed, 1 skipped (pure-numpy skip for XGB)
python -m pytest tests/test_hardening.py -v -k "not TestHardeningSuite"
python -m pytest tests/test_feature_extractor.py -v
python -m pytest -v

# Retrain XGB model
python train_model.py

# Run calculus experiment (real data: 5083 test URLs, 1497 domains)
python experiments/calculus_experiment.py
cat results/calculus_experiment.json

# Launch UI
streamlit run app.py
```

## Web Dashboard (`app.py`)

Four tabs:
1. **1-Click Sample QR Gallery:** 7 built-in samples (`test_qr.png`, `https_url.png`, `http_url.png`, `query_url.png`, `ip_url.png`, `sample_benign_qr_code02.png`, `plain_text.png`)
2. **Upload QR Image:** PNG/JPG/JPEG up to 5 MB
3. **Camera Snapshot:** Single still frame via `st.camera_input`
4. **Direct URL Analyzer & QR Encoder:** Analyze URL string + encode text/URL into QR PNG with warning

**NEW v3.0 Sidebar:** `∫ Calculus Threshold Optimizer` — sliders `C_FP`, `C_FN`, `C_Review`, button `∫ Compute Optimal Thresholds` runs `calculus_threshold.compute_optimal_thresholds()`, displays `dC/dt`, `d²C/dt²`, `is_minimum`, allows Apply `60% / 75%`.

## Command-Line Interface (`test_pipeline.py`)

```bash
python test_pipeline.py --all-samples
python test_pipeline.py --pretty tests/images/ip_url.png
python test_pipeline.py --pretty --url "https://secure-login.paypal.com.evil.xyz/verify?id=1"
python test_pipeline.py tests/images/https_url.png
python test_pipeline.py --allow-bare-domain tests/images/sample_benign_qr_code02.png
```

## 24 Extracted URL Features (`feature_extractor.py`)

| Feature | Category | Description |
|---|---|---|
| `url_length` | Lexical | Total char length |
| `num_dots` | Structure | Dot separators count |
| `has_https` | Protocol | 1 if https:// |
| `has_ip` | Host Risk | 1 if hostname is IPv4/IPv6 literal |
| `num_subdirs` | Structure | Slash count in path only |
| `num_params` | Structure | Query param count |
| `suspicious_words` | Keywords | token+compound hits after masking benign superwords |
| `special_char_count` | Lexical | Count `- _ % @ = ~` |
| `digits_count` | Lexical | Digit count |
| `entropy` | Lexical | Shannon entropy bits |
| `host_length` | Structure | Hostname length |
| `path_length` | Structure | Path length |
| `query_length` | Structure | Query+fragment length |
| `subdomain_count` | Structure | Subdomain levels left of registrable domain |
| `digit_ratio` | Lexical | digits/url_length |
| `susp_token_hits` | Keywords | Standalone {login, verify, account, secure, update, password, signin, bank, confirm, wallet, payment} |
| `susp_compound_hits` | Keywords | Compound substrings (e.g., securebankinggroup) |
| `susp_in_subdomain` | Keywords | 1 if suspicious keyword in subdomain |
| `is_shortener` | Host Risk | 1 if domain in static shortener list (bit.ly, t.co...) |
| `is_punycode` | Host Risk | 1 if label starts xn-- |
| `has_port` | Protocol | 1 if explicit port |
| `has_userinfo` | Host Risk | 1 if @ in authority |
| `embedded_url_in_query` | Host Risk | 1 if raw/encoded redirect URL in query |
| `tld_risk` | Host Risk | 1 if TLD in high-abuse list (.xyz,.top,.zip...) |

## Dataset, Training Reproducibility & Evaluation Scope (`train_model.py`)

### Dataset Construction
- Raw `data/phishing_features.csv`: **820 benign** (bare root domains) + **159,244 phishing** (http/https). Total **160064** rows.
- `train_model.py` expands benign via seeded deterministic path/query generator (`_rand_token` uses `rng.bytes(n).hex()`), applies symmetric 50% scheme-stripping noise to both classes to prevent trivial shortcut.
- Splits grouped by **registrable domain** (`extract_registrable_domain`), `GroupShuffleSplit 80/20` + `GroupKFold 5-fold`, `verify_no_group_leakage` ensures zero domain overlap.

### Held-Out Metrics (`results/baseline.json`)

#### v3.0 — XGBoost (Current)
| Metric | Value |
|---|---:|
| Test Size | 5083 URLs (1497 unseen domains) |
| ROC-AUC | 0.9929 |
| PR-AUC | 0.9976 |
| Brier | 0.0297 |
| Recall TPR @0.50 | 95.9% |
| FPR @0.50 | 4.99% |
| F1 | 0.9707 |
| Precision | 98.27% |
| Confusion | [,] |
[1218][64][156][3645]

#### v2.0 — Random Forest (Previous, for comparison)
| Metric | Value |
|---|---:|
| Test Size | 4693 URLs (1441 domains) |
| ROC-AUC | 0.9902 |
| PR-AUC | 0.9953 |
| Brier | 0.0353 |
| Recall | 96.57% |
| FPR | 7.24% |

## Calculus Threshold Optimization (`calculus_threshold.py`) — CIE-3 Core

**Objective:** `C(t_r,t_b)=C_FP·FPR + C_FN·FNR + C_review·ReviewRate + 0.05||t-t_0||²`

**Mathematical Issue:** Hard `FPR=mean(1_{p≥θ})` is stepwise → derivative 0 or undefined → ordinary GD fails.

**Solution — Sigmoid Smoothing:**
- `FPR̃=mean_benign[sigmoid((p-θ_r)/τ)]`
- `FNR̃=1-mean_phish[sigmoid((p-θ_b)/τ)]`
- `Review̃=mean[sigmoid((p-θ_r)/τ)*(1-sigmoid((p-θ_b)/τ))]`
- `sigmoid(z)=1/(1+e^{-z})`, `τ=0.05`
- `d/dθ sigmoid((p-θ)/τ) = -sigmoid·(1-sigmoid)/τ ≠0` everywhere ✅

**GD Update:** `θ_{k+1}=θ_k - η·∇J̃`, `η=0.05`, constraints `0.15≤θ_r≤0.60, 0.45≤θ_b≤0.95, θ_b-θ_r≥0.15`

**Verification:** Critical point `dC/dt=0`, minimum `d²C/dt²>0`, Bayes optimal single `t*=C_FP/(C_FP+C_FN)=0.1667`

**Real Experiment (5083 URLs):**
| Metric | Baseline 0.35/0.70 | Calculus 0.60/0.75 | Δ |
|--------|-------------------|-------------------|---|
| Cost J | 0.3872 | **0.3676** | **-5.1%** |
| FPR | 7.17% | **3.28%** | **-54%** |
| FNR | 5.81% | 6.37% | +0.56% |
| Review | 3.65% | **1.36%** | **-63%** |
| `d²C>0` | — | 0.48 & 19.3 | Minimum proven |
| Iterations | — | 29 | GD converged |
| Derivatives | — | -0.16 & 1.55 | KKT constrained optimum at θ_r=0.60 bound |

**Numerical Example (Viva Board):**
- Take benign `p=0.2`, phishing `p=0.8`, `θ_r=0.35, τ=0.05`
- `s_r(0.2)=sigmoid((0.2-0.35)/0.05)=sigmoid(-3)=0.047`
- `FPR̃=0.047, FNR̃=0.12, J̃≈0.647`
- `d/dθ_r = -0.047·0.953/0.05 = -0.895` → update `θ_r←0.35 -0.05·(-0.895)=0.394`

**Integration:** `AdaptiveSecurityAgent.optimize_with_calculus(y_true, y_proba)` + Streamlit `∫ Compute Optimal Thresholds`

## Testing & Verification

```bash
python -m py_compile app.py adaptive_agent.py feature_extractor.py calculus_threshold.py qr_utils.py train_model.py tests/*.py
python -m pytest -v  # 26 passed, 1 skipped (pure-numpy loader not applicable for XGB)
python test_pipeline.py --all-samples
python train_model.py
python experiments/calculus_experiment.py
```

##.gitignore
```
.venv/
__pycache__/
*.pyc
models/*.joblib
results/feedback.csv
results/rl_policy_state.json
results/rl_policy_state.json.bak
.DS_Store
```

## License
MIT — R V University CIE-3 Submission — Team ARES
