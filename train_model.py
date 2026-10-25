"""
train_model.py — Deterministic, Domain-Grouped Training & Evaluation Pipeline for ARES.
MIGRATED: Random Forest -> XGBoost + Preserved 5-fold Isotonic Calibration
"""
from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

from feature_extractor import (
    extract_registrable_domain,
    extract_url_features,
    feature_names,
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "phishing_features.csv"
MODEL_OUT_PATH = PROJECT_DIR / "models" / "model.joblib"
BASELINE_OUT_PATH = PROJECT_DIR / "results" / "baseline.json"

SEGMENTS = [
    "docs","blog","news","products","item","about","contact","support","help",
    "search","category","shop","cart","wiki","en","us","api","v2","static","assets",
    "user","profile","post","article","events","careers","press","downloads",
    "pricing","login","account","settings","menu","order","tickets",
]
PARAM_KEYS = ["q","page","id","ref","utm_source","utm_medium","lang","sort","v","t","sessionid","next"]
EXTS = ["", "", "", ".html", ".php", ".aspx", ".pdf", ".jpg"]

def validate_raw_dataset(df: pd.DataFrame) -> dict:
    required_cols = {"url", "label"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
    null_urls = int(df["url"].isna().sum())
    null_labels = int(df["label"].isna().sum())
    if null_labels > 0:
        raise ValueError(f"Dataset contains {null_labels} missing label values.")
    unique_labels = set(df["label"].unique().tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Dataset contains invalid labels outside {{0, 1}}: {unique_labels}")
    valid_df = df.dropna(subset=["url"]).copy()
    valid_df["url_clean"] = valid_df["url"].astype(str).str.strip()
    empty_urls = int((valid_df["url_clean"] == "").sum())
    if empty_urls > 0:
        raise ValueError(f"Dataset contains {empty_urls} empty URL strings.")
    n_benign = int((valid_df["label"] == 0).sum())
    n_phishing = int((valid_df["label"] == 1).sum())
    n_duplicate_urls = int(valid_df["url_clean"].duplicated().sum())
    return {
        "total_rows": int(len(df)),
        "null_urls": null_urls,
        "benign_rows": n_benign,
        "phishing_rows": n_phishing,
        "duplicate_urls": n_duplicate_urls,
    }

def _rand_token(rng: np.random.Generator) -> str:
    kind = int(rng.integers(0, 5))
    if kind == 0:
        exp = int(rng.integers(2, 9))
        return str(int(rng.integers(1, 10**exp)))
    if kind == 1:
        n_bytes = int(rng.integers(3, 9))
        return rng.bytes(n_bytes).hex()
    if kind == 2:
        k = int(rng.integers(1, 5))
        return "-".join(str(x) for x in rng.choice(SEGMENTS, size=k))
    if kind == 3:
        return str(rng.choice(SEGMENTS))
    return str(int(rng.integers(2015, 2027)))

def benign_variants(domain: str, rng: np.random.Generator, n: int = 8) -> list[str]:
    out = [domain]
    for _ in range(n):
        host = ("www." if float(rng.random()) < 0.4 else "") + domain
        depth = int(rng.choice([0, 1, 2, 3, 4, 5], p=[0.15, 0.25, 0.25, 0.15, 0.12, 0.08]))
        path = "/".join(_rand_token(rng) for _ in range(depth)) + str(rng.choice(EXTS))
        url = f"{host}/{path}" if (depth or float(rng.random()) < 0.5) else host
        if float(rng.random()) < 0.3:
            k = int(rng.integers(1, 5))
            q = "&".join(f"{rng.choice(PARAM_KEYS)}={_rand_token(rng)}" for _ in range(k))
            url = url.rstrip("/") + "/?" + q
        scheme = "https://" if float(rng.random()) < 0.8 else "http://"
        out.append(scheme + url)
    return out

def randomise_scheme(url: str, rng: np.random.Generator, p_strip: float = 0.5) -> str:
    if float(rng.random()) < p_strip:
        return url.removeprefix("https://").removeprefix("http://")
    return url

def build_dataset(csv_path: Path, seed: int = 42, phish_per_benign: float = 2.0, max_benign_domains: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    raw = pd.read_csv(csv_path)
    validate_raw_dataset(raw)
    domains = raw.loc[raw["label"] == 0, "url"].dropna().astype(str).str.strip().unique()
    if max_benign_domains is not None:
        domains = domains[:max_benign_domains]
    benign = [u for d in domains for u in benign_variants(d.removeprefix("http://").removeprefix("https://").strip("/"), rng)]
    phish_all = raw.loc[raw["label"] == 1, "url"].dropna().astype(str).str.strip().drop_duplicates().tolist()
    n_phish = min(len(phish_all), int(len(benign) * phish_per_benign))
    chosen_indices = rng.choice(len(phish_all), size=n_phish, replace=False)
    phish = [phish_all[int(i)] for i in chosen_indices]
    rows = []
    for label, urls in ((0, benign), (1, phish)):
        for u in urls:
            u_mod = randomise_scheme(u, rng)
            feats = extract_url_features(u_mod)
            feats.update(url=u_mod, label=int(label), group=extract_registrable_domain(u_mod))
            rows.append(feats)
    return pd.DataFrame(rows)

def prevalence_adjusted_precision(tpr: float, fpr: float, prevalence: float) -> float:
    denom = tpr * prevalence + fpr * (1.0 - prevalence)
    return (tpr * prevalence / denom) if denom > 0 else 0.0

def verify_no_group_leakage(groups: pd.Series, train_idx: np.ndarray, test_idx: np.ndarray, cv_folds: list[tuple[np.ndarray, np.ndarray]] | None = None) -> None:
    train_groups = set(groups.iloc[train_idx].dropna().unique())
    test_groups = set(groups.iloc[test_idx].dropna().unique())
    overlap = train_groups & test_groups
    if overlap:
        raise AssertionError(f"Domain leakage detected between train and test splits: {len(overlap)} shared domains.")
    if cv_folds is not None:
        train_slice_groups = groups.iloc[train_idx]
        for fold_num, (f_tr, f_val) in enumerate(cv_folds):
            g_ftr = set(train_slice_groups.iloc[f_tr].dropna().unique())
            g_fval = set(train_slice_groups.iloc[f_val].dropna().unique())
            fold_overlap = g_ftr & g_fval
            if fold_overlap:
                raise AssertionError(f"Domain leakage detected in calibration fold {fold_num}: {len(fold_overlap)} shared domains.")

def train_and_save(seed: int = 42, backup_existing: bool = True) -> dict:
    import joblib
    import sklearn
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (
        average_precision_score, brier_score_loss, confusion_matrix,
        f1_score, precision_score, recall_score, roc_auc_score,
    )
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit
    from xgboost import XGBClassifier

    raw_df = pd.read_csv(DATA_PATH)
    dataset_summary = validate_raw_dataset(raw_df)

    df = build_dataset(DATA_PATH, seed=seed)
    X, y, g = df[feature_names], df["label"], df["group"]

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed).split(X, y, g))
    Xtr, ytr, gtr = X.iloc[tr], y.iloc[tr], g.iloc[tr]
    Xte, yte, gte = X.iloc[te], y.iloc[te], g.iloc[te]

    folds = list(GroupKFold(n_splits=5).split(Xtr, ytr, gtr))
    verify_no_group_leakage(g, tr, te, folds)

    neg = int((ytr == 0).sum())
    pos = int((ytr == 1).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    base = XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist",
        verbosity=0,
    )
    model = CalibratedClassifierCV(base, method="isotonic", cv=folds)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Xtr, ytr)

    ref_xgb = XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist",
        verbosity=0,
    )
    ref_xgb.fit(Xtr, ytr)
    importances = ref_xgb.feature_importances_

    p = model.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    tpr = float(recall_score(yte, pred, pos_label=1))
    fpr = float(1.0 - recall_score(yte, pred, pos_label=0))
    prec = float(precision_score(yte, pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(yte, pred, pos_label=1, zero_division=0))
    test_prevalence = float(yte.mean())

    metrics = {
        "split": "GroupShuffleSplit (80/20) by registrable domain; 5-fold GroupKFold isotonic calibration",
        "model_type": "XGBoost (XGBClassifier) + isotonic calibration — migrated from RandomForest",
        "evaluation_scope": "Held-out domain-grouped evaluation on the semi-synthetic constructed dataset",
        "raw_dataset_summary": dataset_summary,
        "random_seed": seed,
        "sklearn_version": sklearn.__version__,
        "n_train": int(len(ytr)),
        "n_train_domains": int(gtr.nunique()),
        "n_test": int(len(yte)),
        "n_test_domains": int(gte.nunique()),
        "test_phishing_prevalence": round(test_prevalence, 4),
        "roc_auc": round(float(roc_auc_score(yte, p)), 4),
        "pr_auc": round(float(average_precision_score(yte, p)), 4),
        "brier": round(float(brier_score_loss(yte, p)), 4),
        "precision": round(prec, 4),
        "f1_score": round(f1, 4),
        "tpr": round(tpr, 4),
        "fpr": round(fpr, 4),
        "confusion_matrix": confusion_matrix(yte, pred, labels=[0, 1]).tolist(),
        "precision_at_prevalence": {str(pv): round(prevalence_adjusted_precision(tpr, fpr, pv), 4) for pv in (0.5, 0.05, 0.01, 0.001)},
        "xgboost_params": base.get_params(),
    }

    if backup_existing and MODEL_OUT_PATH.exists():
        shutil.copy2(MODEL_OUT_PATH, MODEL_OUT_PATH.with_suffix(".joblib.bak"))
    if backup_existing and BASELINE_OUT_PATH.exists():
        shutil.copy2(BASELINE_OUT_PATH, BASELINE_OUT_PATH.with_suffix(".json.bak"))

    bundle = {
        "model": model,
        "features": list(feature_names),
        "label_mapping": {0: "Benign", 1: "Phishing"},
        "feature_importances": [float(v) for v in importances],
        "feature_importance_source": "separately_fitted_uncalibrated_xgboost_on_training_split",
        "model_type": "xgboost",
    }
    MODEL_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_OUT_PATH)
    BASELINE_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_OUT_PATH.write_text(json.dumps({"baseline": {"test": metrics}}, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics

if __name__ == "__main__":
    train_and_save()