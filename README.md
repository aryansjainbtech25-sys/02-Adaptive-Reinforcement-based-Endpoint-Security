# ARES — Adaptive Reinforcement-based Endpoint Security

**Offline QR-Code Phishing (Quishing) Risk Analyzer with Calibrated XGBoost & Calculus-Optimized Threshold Policy**

ARES is an educational and research prototype developed at **RV University** that evaluates URLs embedded inside QR codes (**quishing**) using offline lexical and structural analysis. It decodes QR images without visiting or fetching the destination URL, extracts **24 lexical, host, and structural URL features**, estimates phishing probability using a **5-fold Isotonic-Calibrated XGBoost (`XGBClassifier` + `CalibratedClassifierCV`)** — migrated from Random Forest — and recommends an advisory endpoint action (`Allow`, `Review`, or `Block`) governed by a **feedback-driven + calculus-optimized threshold controller**.

> **Important Security Disclaimer:** ARES is an offline lexical URL analyzer, **not** a full Endpoint Detection and Response (EDR) agent, browser sandbox, or live threat-intelligence scanner. A low estimated phishing probability (`Allow` recommendation) **does not guarantee** that a website is safe, and `https://` indicates encrypted network transport rather than domain legitimacy. Never use ARES as the sole basis for trusting an unknown link.

## Application Screenshot

<p align="center">
  <img
    src="ares-dashboard.png"
    alt="ARES dashboard showing QR-code phishing detection, sample QR gallery, and adaptive threshold policy"
    width="950"
  />
</p>

<p align="center">
  <em>ARES dashboard — QR-code phishing detection and adaptive risk assessment.</em>
</p>

---

## What's New in v3.0 — XGBoost + Calculus Migration (AIML Semester 3)

- **Model Migration:** Random Forest (100 trees, `scikit-learn==1.9.1`) → **XGBoost (`XGBClassifier`, 400 estimators, max_depth=8, learning_rate=0.05, subsample=0.9, colsample=0.9, scale_pos_weight balanced)** with `tree_method=hist`. ROC-AUC improved **0.9902 → 0.9929**, PR-AUC **0.9953 → 0.9976**, Brier **0.0353 → 0.0297**.
- **Calibration Preserved:** 5-fold `GroupKFold` Isotonic Calibration, `GroupShuffleSplit` by registrable domain (zero leakage verified), dual probability reporting (calibrated + uncalibrated).
- **Calculus Component (NEW — `calculus_threshold.py`):** Defines expected cost $C(t_r,t_b)=C_{FP}·FPR + C_{FN}·FNR + C_{review}·ReviewRate + \lambda||t-t_0||^2$. Computes first derivative $dC/dt≈[C(t+ε)-C(t-ε)]/(2ε)$, second derivative $d^2C/dt^2≈[C(t+ε)-2C(t)+C(t-ε)]/ε^2$. Optimum at $dC/dt=0$, verified $d^2C/dt^2>0$ (minimum). Example optimum on validation: **Review 30.45%, Block 45.51%**, cost 0.092257, derivatives -0.0045/-0.024 ≈0, second derivatives 0.1>0, Bayes optimal single threshold $C_{FP}/(C_{FP}+C_{FN})=0.25$.
- **New File:** `calculus_threshold.py` — standalone calculus module integrated into `AdaptiveSecurityAgent.optimize_with_calculus()` and Streamlit UI.

## System Architecture & Key Capabilities

1. **Bomb-Safe, Multi-Variant QR Decoding (`qr_utils.py`):**
      - Verifies image dimensions from file headers before decoding and rejects images exceeding `25,000,000` pixels.
      - Composites transparent RGBA PNGs onto a white background so dark QR modules remain readable.
      - Sweeps across native resolution (`scale = 1`, including `3000px` boundary images with `20px` quiet-zone padding), nearest-neighbor upscaling (`2x–10x` for small $21\text{–}69\text{ px}$ dataset matrices), Otsu/adaptive binarization, and inverted (white-on-black) polarities.
      - Detects multiple QR codes in a single image (`detectAndDecodeMulti`): fails closed in single-code CLI mode (`decode_qr_code`) and prompts the user to explicitly choose which payload to inspect in the Streamlit UI (`decode_qr_codes`).
2. **Centralized URL Validation & Domain Parsing (`feature_extractor.py`):**
      - Uses a single canonical parser (`canonical_split`, `parse_host_structure`, `extract_registrable_domain`) across validation, feature extraction, UI anatomy display, and domain-grouped model training.
      - Normalizes authority/path backslashes (`\`) and strips control characters, supports IPv4/IPv6 literals, explicit ports, and IDN Punycode (`xn--`), and rejects plain-text payloads (`Hello ARES`) or non-HTTP(S) schemes (`javascript:`, `mailto:`, `WIFI:`).
3. **Dual Probability Reporting (`adaptive_agent.py`):**
      - Computes both the **Isotonic-Calibrated Phishing Probability** (mean of the 5 calibrated cross-validation fold estimators) and the **Uncalibrated Base XGBoost Probability** (mean of the 5 uncalibrated base `XGBClassifier` fold estimators prior to isotonic mapping) — previously Random Forest.
      - Validates `model.classes_` (`[0, 1]`), feature order (`24` features), and finite probabilities summing to `1.0`.
      - Includes a validated pure-NumPy execution engine (`PureNumpyCalibratedClassifier`) for backwards compatibility with old Random Forest artifacts, and a native XGBoost unpickler that allows `xgboost.sklearn.XGBClassifier` via `joblib`. Automatically handles version skew (artifact `1.5.2`).
4. **Feedback-Driven + Calculus-Optimized Threshold Policy (`adaptive_agent.py` + `calculus_threshold.py`):**
      - Implements a heuristic, probability-margin threshold controller over `review_threshold` (default `35%`) and `block_threshold` (default `70%`), plus **calculus-optimized thresholds** via cost minimization.
      - **Calculus formulation:** $C(t_r,t_b)=C_{FP}FPR + C_{FN}FNR + C_{review}ReviewRate + \lambda/(t_b-t_r)$ regularization. First derivative via central difference, second derivative test for minimum. Bayes optimal $t^*=C_{FP}/(C_{FP}+C_{FN})$.
      - Redacts sensitive query parameter values (`token`, `session`, `id`, `password`, `key`, etc.) before writing to `results/feedback.csv`, rejects duplicate and contradictory verdicts for the same normalized URL, and uses cross-platform file locking + atomic `.tmp` replacement.
      - **Note:** User feedback adjusts the `Review` and `Block` recommendation thresholds; it does **not** retrain the underlying XGBoost classifier.

## Requirements & Compatibility

- **Python Version:** Python `3.10`, `3.11`, or `3.12` recommended.
- **Dependencies (`requirements.txt`):**
    - `scikit-learn==1.5.2` (matches the serialized XGBoost `models/model.joblib` artifact) — previously `1.9.1` for RF
    - `xgboost==3.0.2` **NEW** — replaces RandomForest
    - `pandas>=2.0.0`, `numpy>=1.26.0`, `joblib>=1.3.0`, `opencv-python-headless>=4.8.0`, `Pillow>=10.0.0`, `streamlit>=1.38.0`, `pytest>=8.0.0`, `matplotlib>=3.8.0`

## Installation & Launch Instructions

### Windows (Clean Checkout)

#### Option A: One-Command Launcher (`run.bat`)
From Command Prompt (`cmd.exe`) or PowerShell in the project root:

```cmd
run.bat
