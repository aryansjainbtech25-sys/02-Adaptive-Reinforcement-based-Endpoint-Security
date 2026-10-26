"""Run real calculus experiment on held-out test split from train_model.py"""
import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import json, numpy as np
from feature_extractor import feature_names
from adaptive_agent import load_model_bundle
from calculus_threshold import compute_optimal_thresholds, expected_cost, _compute_two_thr_rates

MODEL_PATH = PROJECT_DIR / "models" / "model.joblib"
BASELINE_PATH = PROJECT_DIR / "results" / "baseline.json"

# Build dataset same as train_model.py
from train_model import build_dataset
from sklearn.model_selection import GroupShuffleSplit

df = build_dataset(PROJECT_DIR / "data" / "phishing_features.csv", seed=42)
X, y, g = df[feature_names], df["label"], df["group"]
_, te = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42).split(X,y,g))
Xte, yte = X.iloc[te], y.iloc[te]

bundle = load_model_bundle(MODEL_PATH)
proba = bundle["model"].predict_proba(Xte)[:,1]

# Baseline 0.35 / 0.70
base_rates = _compute_two_thr_rates(yte.values, proba, 0.35, 0.70)
base_cost = expected_cost(yte.values, proba, 0.35, 0.70, 1.0, 5.0, 0.5)

# Calculus optimized with real data
result = compute_optimal_thresholds(yte.values, proba, cost_fp=1.0, cost_fn=5.0, cost_review=0.5)

output = {
    "baseline": {"review":0.35,"block":0.70,"cost":base_cost,"rates":base_rates},
    "optimized": result,
    "baseline_metrics": json.loads(BASELINE_PATH.read_text())["baseline"]["test"],
    "n_test": int(len(yte)),
}

def _to_py(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return o

# Convert recursively for JSON
import json as js
def _clean(d):
    if isinstance(d, dict):
        return {k: _clean(v) for k,v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_clean(x) for x in d]
    return _to_py(d)

cleaned = _clean(output)
Path(PROJECT_DIR / "results" / "calculus_experiment.json").write_text(js.dumps(cleaned, indent=2))
print(js.dumps(cleaned, indent=2))
