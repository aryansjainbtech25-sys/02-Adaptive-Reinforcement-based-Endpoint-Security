from pathlib import Path
import sys
import json
import numpy as np
import joblib

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from train_model import build_dataset
from sklearn.model_selection import GroupShuffleSplit
from calculus_threshold import _compute_two_thr_rates, expected_cost, compute_optimal_thresholds

MODEL_PATH = PROJECT_DIR / "models" / "model.joblib"
BASELINE_PATH = PROJECT_DIR / "results" / "baseline.json"

def load_model_bundle(path):
    return joblib.load(path)

def _to_py(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return o

def _clean(d):
    if isinstance(d, dict):
        return {k: _clean(v) for k,v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_clean(x) for x in d]
    return _to_py(d)

def main():
    df = build_dataset(PROJECT_DIR / "data" / "phishing_features.csv", seed=42)
    
    bundle = load_model_bundle(MODEL_PATH)
    
    # FIX: Use feature names from the SAVED bundle, not from current train_model.py
    # Old model was trained with older feature set, new train_model.py may have extra features
    if "feature_names" in bundle and bundle["feature_names"]:
        feature_names = bundle["feature_names"]
        print(f"Using bundle feature_names ({len(feature_names)}): {feature_names[:5]}...")
    else:
        try:
            from train_model import FEATURE_NAMES as feature_names
        except ImportError:
            feature_names = [c for c in df.columns if c not in ("label","group","domain","url")]
    
    # Ensure all bundle features exist in df, if not fallback
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        print(f"WARNING: Bundle features missing in dataset: {missing}")
        print(f"Falling back to intersection")
        feature_names = [c for c in feature_names if c in df.columns]
    
    X, y, g = df[feature_names], df["label"], df["group"]
    _, te = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42).split(X,y,g))
    Xte, yte = X.iloc[te], y.iloc[te]

    proba = bundle["model"].predict_proba(Xte)[:,1]

    base_rates = _compute_two_thr_rates(yte.values, proba, 0.35, 0.70)
    base_cost = expected_cost(yte.values, proba, 0.35, 0.70, 1.0, 5.0, 0.5)
    result = compute_optimal_thresholds(yte.values, proba, cost_fp=1.0, cost_fn=5.0, cost_review=0.5)

    output = {
        "baseline": {"review":0.35,"block":0.70,"cost":base_cost,"rates":base_rates},
        "optimized": result,
        "baseline_metrics": json.loads(BASELINE_PATH.read_text())["baseline"]["test"] if BASELINE_PATH.exists() else {},
        "n_test": int(len(yte)),
    }

    cleaned = _clean(output)
    out_path = PROJECT_DIR / "results" / "calculus_experiment.json"
    out_path.write_text(json.dumps(cleaned, indent=2))
    print(f"\nSaved to {out_path}")
    print(json.dumps(cleaned["optimized"]["derivative_check"], indent=2))
    return cleaned

if __name__ == "__main__":
    main()
