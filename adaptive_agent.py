from __future__ import annotations

from contextlib import contextmanager
import csv
import json
import os
import pickle
import warnings
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from feature_extractor import (
     extract_url_features,
    feature_names,
    parse_host_structure,
    redact_url_for_storage,
    suspicious_words,
)

EXPECTED_ARTIFACT_SKLEARN_VERSION = "1.9.1"


@contextmanager
def _exclusive_file_lock(lock_path: Path):
    """Cross-platform advisory file lock for safe concurrent CSV and JSON state updates."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if os.name == "nt":
            import msvcrt  # type: ignore

            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt  # type: ignore

                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
        except Exception:
            pass
        os.close(fd)


class _PureNumpyTree:
    """Lightweight execution engine for a serialized scikit-learn DecisionTree state dict."""

    def __init__(self, state: dict):
        self.state = state
        self.nodes = state["nodes"]
        self.values = state["values"]

    def predict_proba_single(self, x_vec: np.ndarray) -> np.ndarray:
        idx = 0
        nodes = self.nodes
        while nodes[idx]["left_child"] != -1:
            feat_idx = int(nodes[idx]["feature"])
            threshold = float(nodes[idx]["threshold"])
            if x_vec[feat_idx] <= threshold:
                idx = int(nodes[idx]["left_child"])
            else:
                idx = int(nodes[idx]["right_child"])
        val = self.values[idx][0].astype(np.float64)
        total = val.sum()
        return val / total if total > 0 else val


class PureNumpyRandomForest:
    """Pure-NumPy execution engine for a serialized scikit-learn RandomForestClassifier."""

    def __init__(self, state: dict):
        self.state = state
        self.classes_ = np.asarray(state.get("classes_", [0, 1]))
        self.n_features_in_ = int(state.get("n_features_in_", len(feature_names)))
        self.feature_names_in_ = np.asarray(
            state.get("feature_names_in_", feature_names)
        )
        self.estimators_ = state.get("estimators_", [])
        self._sklearn_version = state.get("_sklearn_version", "unknown")

    def predict_proba(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            arr = X.to_numpy(dtype=np.float64)
        else:
            arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        n_samples = arr.shape[0]
        n_classes = len(self.classes_)
        out = np.zeros((n_samples, n_classes), dtype=np.float64)

        for i in range(n_samples):
            row_probs = np.zeros(n_classes, dtype=np.float64)
            for est in self.estimators_:
                tree_obj = getattr(est, "tree_", None)
                if tree_obj is not None:
                    row_probs += _PureNumpyTree(tree_obj.state).predict_proba_single(
                        arr[i]
                    )
            out[i] = row_probs / max(len(self.estimators_), 1)
        return out

    def predict(self, X) -> np.ndarray:
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]


class PureNumpyCalibratedClassifier:
    """
    Pure-NumPy execution engine for a serialized scikit-learn CalibratedClassifierCV
    (with 5 fold RandomForest estimators and IsotonicRegression calibrators).
    """

    def __init__(self, state: dict):
        self.state = state
        self.classes_ = np.asarray(state.get("classes_", [0, 1]))
        self.n_features_in_ = int(state.get("n_features_in_", len(feature_names)))
        self.feature_names_in_ = np.asarray(
            state.get("feature_names_in_", feature_names)
        )
        self.calibrated_classifiers_ = state.get("calibrated_classifiers_", [])
        self._sklearn_version = state.get("_sklearn_version", "unknown")

    def predict_uncalibrated_proba(self, X) -> np.ndarray:
        """Return the mean uncalibrated Random Forest probability vector across CV folds."""
        if isinstance(X, pd.DataFrame):
            arr = X.to_numpy(dtype=np.float64)
        else:
            arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if not self.calibrated_classifiers_:
            raise ValueError("CalibratedClassifierCV contains no fitted fold classifiers.")
        fold_probs = []
        for cc in self.calibrated_classifiers_:
            cc_state = getattr(cc, "state", cc.__dict__)
            est = cc_state["estimator"]
            fold_probs.append(est.predict_proba(arr))
        mean_probs = np.mean(fold_probs, axis=0)
        row_sums = mean_probs.sum(axis=1, keepdims=True)
        return np.where(row_sums > 0, mean_probs / row_sums, mean_probs)

    def predict_proba(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            arr = X.to_numpy(dtype=np.float64)
        else:
            arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if not self.calibrated_classifiers_:
            raise ValueError("CalibratedClassifierCV contains no fitted fold classifiers.")

        classes_list = [int(c) for c in self.classes_]
        pos_idx = classes_list.index(1) if 1 in classes_list else 1

        cal_p1_folds = []
        for cc in self.calibrated_classifiers_:
            cc_state = getattr(cc, "state", cc.__dict__)
            est = cc_state["estimator"]
            calibrators = cc_state.get("calibrators", [])
            raw_p1 = est.predict_proba(arr)[:, pos_idx]
            if calibrators:
                cal0 = calibrators[0]
                cal_state = getattr(cal0, "state", cal0.__dict__)
                if "X_thresholds_" in cal_state and "y_thresholds_" in cal_state:
                    cal_p1 = np.interp(
                        raw_p1,
                        cal_state["X_thresholds_"],
                        cal_state["y_thresholds_"],
                    )
                else:
                    raise ValueError("Unsupported calibrator state in model artifact.")
            else:
                cal_p1 = raw_p1
            cal_p1_folds.append(cal_p1)

        mean_p1 = np.clip(np.mean(cal_p1_folds, axis=0), 0.0, 1.0)
        if pos_idx == 1:
            return np.column_stack([1.0 - mean_p1, mean_p1])
        return np.column_stack([mean_p1, 1.0 - mean_p1])

    def predict(self, X) -> np.ndarray:
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]


class _PureJoblibUnpickler(pickle._Unpickler):
    """
    Deterministic, version-agnostic loader for joblib-serialized scikit-learn
    RandomForestClassifier and CalibratedClassifierCV bundles.
    """

    def __init__(self, file_handle):
        super().__init__(file_handle)
        self.file_handle = file_handle

    def load_build(self):
        stack = self.stack
        inst = stack[-2]
        if type(inst).__name__ == "NumpyArrayWrapper":
            state = stack.pop()
            arr = inst.read_array(self.file_handle, state)
            stack[-1] = arr
            return
        pickle._Unpickler.load_build(self)

    dispatch = pickle._Unpickler.dispatch.copy()
    dispatch[pickle.BUILD[0]] = load_build

    def find_class(self, module: str, name: str):
        if module == "joblib.numpy_pickle" and name == "NumpyArrayWrapper":
            class NumpyArrayWrapper:
                def read_array(self, f, state):
                    shape = state["shape"]
                    order = state.get("order", "C")
                    dtype = state["dtype"]
                    align = state.get("numpy_array_alignment_bytes", None)
                    if dtype.hasobject:
                        return pickle._Unpickler(f).load()
                    if align is not None:
                        pad_b = f.read(1)
                        if not pad_b:
                            raise EOFError("Truncated NumpyArrayWrapper alignment byte.")
                        pad_len = pad_b[0]
                        f.read(pad_len)
                    count = int(np.prod(shape)) * dtype.itemsize
                    raw = f.read(count)
                    if len(raw) != count:
                        raise EOFError(
                            f"Truncated array stream: expected {count} bytes, got {len(raw)}."
                        )
                    return np.frombuffer(raw, dtype=dtype).reshape(shape, order=order)

            return NumpyArrayWrapper

        if module.startswith("sklearn") or module.startswith("scipy"):
            if name == "RandomForestClassifier":
                class _RFHolder:
                    def __init__(self, *args, **kwargs):
                        pass

                    def __setstate__(self, state):
                        self.state = state
                        self.__dict__.update(state)
                        self._wrapped = PureNumpyRandomForest(state)

                    def predict(self, X):
                        return self._wrapped.predict(X)

                    def predict_proba(self, X):
                        return self._wrapped.predict_proba(X)

                return _RFHolder

            if name == "CalibratedClassifierCV":
                class _CalibratedHolder:
                    def __init__(self, *args, **kwargs):
                        pass

                    def __setstate__(self, state):
                        self.state = state
                        self.__dict__.update(state)
                        self._wrapped = PureNumpyCalibratedClassifier(state)

                    def predict(self, X):
                        return self._wrapped.predict(X)

                    def predict_proba(self, X):
                        return self._wrapped.predict_proba(X)

                    def predict_uncalibrated_proba(self, X):
                        return self._wrapped.predict_uncalibrated_proba(X)

                return _CalibratedHolder

            class _SklearnObj:
                def __init__(self, *args, **kwargs):
                    self.args = args

                def __setstate__(self, state):
                    self.state = state
                    if isinstance(state, dict):
                        self.__dict__.update(state)

            _SklearnObj.__name__ = name
            return _SklearnObj

        return super().find_class(module, name)


def _validate_loaded_bundle(bundle: dict) -> None:
    """Validate bundle structure, feature schema, class mapping, and numerical probability output."""
    if not isinstance(bundle, dict):
        raise ValueError(
            f"Loaded bundle is not a dictionary. Got {type(bundle)} instead."
        )

    if "features" not in bundle and "feature_names" in bundle:
        bundle["features"] = bundle["feature_names"]

    required_keys = {"model", "features", "label_mapping"}
    missing_keys = required_keys - set(bundle.keys())
    if missing_keys:
        raise ValueError(f"Loaded bundle is missing required keys: {missing_keys}")

    if list(bundle["features"]) != list(feature_names):
        raise ValueError(
            f"Loaded bundle features {list(bundle['features'])} do not match expected {list(feature_names)}."
        )

    label_mapping = bundle["label_mapping"]
    if not isinstance(label_mapping, dict) or set(label_mapping.keys()) != {0, 1}:
        raise ValueError(
            f"Invalid binary label_mapping in model bundle: {label_mapping!r}. Expected keys {{0, 1}}."
        )
    if (
        str(label_mapping[0]).lower() != "benign"
        or str(label_mapping[1]).lower() != "phishing"
    ):
        raise ValueError(
            f"Unexpected class label names in label_mapping: {label_mapping!r}."
        )

    model = bundle["model"]
    if getattr(model, "n_features_in_", len(feature_names)) != len(feature_names):
        raise ValueError(
            "Loaded model feature count does not match expected number of features."
        )

    if hasattr(model, "feature_names_in_"):
        if list(model.feature_names_in_) != list(feature_names):
            raise ValueError(
                "Model feature_names_in_ order does not match expected feature_names."
            )

    classes = [int(c) for c in getattr(model, "classes_", [])]
    if classes != [0, 1]:
        raise ValueError(
            f"Model classes_ must be [0, 1] for binary phishing detection; got {classes}."
        )

    # Validate probability output on a canonical test vector
    probe_df = pd.DataFrame(
        [extract_url_features("https://example.com")], columns=feature_names
    )
    probs = np.asarray(model.predict_proba(probe_df), dtype=np.float64)
    if probs.shape != (1, 2):
        raise ValueError(f"Expected predict_proba shape (1, 2), got {probs.shape}.")
    if not np.all(np.isfinite(probs)):
        raise ValueError("Model returned non-finite probabilities on probe input.")
    if np.any(probs < -1e-6) or np.any(probs > 1.0 + 1e-6):
        raise ValueError(f"Model probabilities out of [0, 1] bounds: {probs}.")
    if not np.isclose(probs.sum(axis=1)[0], 1.0, atol=1e-5):
        raise ValueError(f"Model probabilities do not sum to 1.0: sum={probs.sum()}.")


def load_model_bundle(
    model_path: str | Path,
    allow_external_path: bool = False,
    force_pure_numpy: bool = False,
) -> dict:
    """
    Load and validate the trusted ARES model bundle from disk.

    Security & Version Compatibility Policy:
    - Refuses to load arbitrary user-supplied files outside the repository `models/` directory
      unless `allow_external_path=True` is explicitly set by a unit test.
    - If `joblib` / `scikit-learn` are installed and their version matches the artifact's
      serialized `_sklearn_version` without emitting `InconsistentVersionWarning`, uses `joblib.load()`.
    - If `scikit-learn` is not installed or its runtime version differs from the artifact's
      `_sklearn_version` (which would otherwise emit `InconsistentVersionWarning`), routes
      deterministically to `_PureJoblibUnpickler` (`PureNumpyCalibratedClassifier`) so tree
      and isotonic arrays are evaluated directly without Cython version skew.
    """
    resolved_path = Path(model_path).resolve()
    trusted_models_dir = (Path(__file__).resolve().parent / "models").resolve()

    if not allow_external_path and resolved_path.parent != trusted_models_dir:
        raise ValueError(
            f"Refusing to deserialize model file outside trusted models directory ({trusted_models_dir}): {resolved_path}"
        )

    if not resolved_path.exists():
        raise FileNotFoundError(f"Model file not found at {resolved_path}")

    bundle = None
    loader_used = "pure_numpy_calibrated_loader"
    version_note = ""

    if not force_pure_numpy:
        try:
            import joblib  # type: ignore
            import sklearn  # type: ignore

            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                candidate = joblib.load(resolved_path)

            artifact_ver = getattr(
                candidate.get("model"), "_sklearn_version", EXPECTED_ARTIFACT_SKLEARN_VERSION
            )
            runtime_ver = getattr(sklearn, "__version__", "unknown")
            has_version_warning = any(
                "InconsistentVersionWarning" in type(w.message).__name__
                or "Trying to unpickle estimator" in str(w.message)
                for w in caught_warnings
            )

            if has_version_warning or artifact_ver != runtime_ver:
                version_note = (
                    f"Artifact serialized with scikit-learn {artifact_ver}, runtime has {runtime_ver}; "
                    "using deterministic PureNumpyCalibratedClassifier to avoid cross-version Cython skew."
                )
            else:
                _validate_loaded_bundle(candidate)
                bundle = candidate
                loader_used = f"joblib_sklearn_{runtime_ver}"
        except Exception as exc:
            version_note = f"Standard joblib loader unavailable or incompatible ({type(exc).__name__}); using PureNumpyCalibratedClassifier."

    if bundle is None:
        with open(resolved_path, "rb") as file_handle:
            bundle = _PureJoblibUnpickler(file_handle).load()
        _validate_loaded_bundle(bundle)

    model_obj = bundle["model"]
    artifact_sklearn_version = getattr(
        model_obj, "_sklearn_version", EXPECTED_ARTIFACT_SKLEARN_VERSION
    )
    bundle["model_metadata"] = {
        "artifact_file": resolved_path.name,
        "model_type": type(model_obj).__name__,
        "artifact_sklearn_version": str(artifact_sklearn_version),
        "active_loader": loader_used,
        "loader_note": version_note or "Loaded natively with matching scikit-learn version.",
        "feature_count": len(feature_names),
        "feature_importance_source": (
            "separately_fitted_uncalibrated_random_forest_on_training_split"
            if "feature_importances" in bundle
            else "estimator_feature_importances"
        ),
    }
    return bundle


def compute_model_probabilities(bundle: dict, df: pd.DataFrame) -> dict:
    """
    Compute both the Isotonic-Calibrated probabilities and the Uncalibrated Base Random
    Forest probabilities from the 5 fitted fold estimators, verifying class ordering.
    """
    model = bundle["model"]
    label_mapping = bundle["label_mapping"]
    classes = [int(c) for c in model.classes_]
    if 0 not in classes or 1 not in classes:
        raise ValueError(f"Expected binary classes 0 and 1, got {classes}")

    pos_idx = classes.index(1)
    neg_idx = classes.index(0)

    cal_probs_arr = np.asarray(model.predict_proba(df), dtype=np.float64)[0]
    if not np.all(np.isfinite(cal_probs_arr)):
        raise ValueError("Non-finite calibrated probability output.")

    cal_phish = float(np.clip(cal_probs_arr[pos_idx], 0.0, 1.0))
    cal_benign = float(np.clip(cal_probs_arr[neg_idx], 0.0, 1.0))
    total_cal = cal_phish + cal_benign
    if total_cal > 0:
        cal_phish /= total_cal
        cal_benign /= total_cal

    # Calculate uncalibrated base-estimator probability across the 5 CV folds
    uncal_phish: float | None = None
    if hasattr(model, "predict_uncalibrated_proba"):
        uncal_arr = np.asarray(model.predict_uncalibrated_proba(df), dtype=np.float64)[0]
        uncal_phish = float(np.clip(uncal_arr[pos_idx], 0.0, 1.0))
    elif hasattr(model, "calibrated_classifiers_"):
        fold_p1 = []
        for cc in model.calibrated_classifiers_:
            est = getattr(cc, "estimator", None)
            if est is None and hasattr(cc, "state"):
                est = cc.state.get("estimator")
            if est is not None and hasattr(est, "predict_proba"):
                fold_p1.append(float(est.predict_proba(df)[0, pos_idx]))
        if fold_p1:
            uncal_phish = float(np.clip(np.mean(fold_p1), 0.0, 1.0))

    if uncal_phish is None:
        uncal_phish = cal_phish

    uncal_benign = 1.0 - uncal_phish
    predicted_class_int = 1 if cal_phish >= 0.50 else 0
    predicted_label = str(label_mapping.get(predicted_class_int, str(predicted_class_int)))

    return {
        "predicted_class_index": predicted_class_int,
        "predicted_label": predicted_label,
        "calibrated_probabilities": {
            str(label_mapping[0]): round(cal_benign, 6),
            str(label_mapping[1]): round(cal_phish, 6),
        },
        "uncalibrated_rf_probabilities": {
            str(label_mapping[0]): round(uncal_benign, 6),
            str(label_mapping[1]): round(uncal_phish, 6),
        },
        "calibrated_phishing_probability": cal_phish,
        "uncalibrated_rf_phishing_probability": uncal_phish,
    }


def inspect_url_anatomy(url: str) -> dict:
    """
    Break a URL into human-readable structural components using the centralized parser
    in `feature_extractor.py`.
    """
    host_info = parse_host_structure(url)
    parsed = host_info["parsed"]
    subs = host_info["subdomain_labels"]
    registrable = host_info["registrable_domain"]
    hostname = host_info["hostname"]
    tld = host_info["tld"]
    port = host_info["port"]

    scheme = (
        url.strip().split("://", 1)[0].lower()
        if "://" in url.strip()
        else "(none / bare domain)"
    )
    return {
        "scheme": scheme,
        "subdomain": ".".join(subs) if subs else "(none)",
        "registrable_domain": registrable or hostname or "(none)",
        "tld": f".{tld}" if tld else "(none)",
        "port": str(port) if port is not None else "(default)",
        "path": parsed.path or "/",
        "query": f"?{parsed.query}" if parsed.query else "(none)",
        "host_dot_count": host_info["host_dot_count"],
    }


def analyze_url_risk_factors(
    url: str,
    features: dict,
    calibrated_phish_prob: float,
    uncalibrated_rf_phish_prob: float | None = None,
) -> dict:
    """
    Generate consistent, human-readable explanations of the extracted 24 features.
    Does not mutate the model's calibrated probability and never confuses path/query dots
    with hostname subdomain depth.
    """
    risk_factors: list[str] = []
    protective_factors: list[str] = []

    lower_url = url.lower()
    matched_words = [w for w in suspicious_words if w in lower_url]

    if features.get("has_ip") == 1:
        risk_factors.append(
            "Hostname is a raw IP address rather than a registered domain name"
        )

    if features.get("has_userinfo") == 1:
        risk_factors.append(
            "URL authority contains '@' userinfo syntax, which can disguise the actual destination host"
        )

    if features.get("is_punycode") == 1:
        risk_factors.append(
            "Hostname uses IDN Punycode ('xn--') encoding, which can be used for look-alike homograph domains"
        )

    if features.get("is_shortener") == 1:
        risk_factors.append(
            "Domain matches a known URL-shortening service that obscures the final destination"
        )

    if features.get("embedded_url_in_query") == 1:
        risk_factors.append(
            "Query string or fragment contains an embedded redirect URL pattern"
        )

    if features.get("tld_risk") == 1:
        risk_factors.append(
            "Top-level domain appears in the static high-abuse TLD indicator list"
        )

    if features.get("susp_in_subdomain") == 1:
        risk_factors.append(
            "Sensitive keyword (e.g., login/secure/bank) appears inside the subdomain prefix left of the registrable domain"
        )

    if features.get("suspicious_words", 0) > 0 and matched_words:
        risk_factors.append(
            f"Contains {features['suspicious_words']} phishing-associated keyword match(es): {', '.join(matched_words)}"
        )

    if features.get("has_port") == 1:
        risk_factors.append("Specifies an explicit network port in the URL authority")

    if features.get("entropy", 0.0) >= 4.2:
        risk_factors.append(
            f"Elevated character entropy ({features['entropy']:.2f} bits), often caused by random tokens or long identifiers"
        )

    # Strictly distinguish hostname subdomain depth from path/query dots
    sub_count = int(features.get("subdomain_count", 0))
    total_dots = int(features.get("num_dots", 0))
    if sub_count >= 2:
        risk_factors.append(
            f"Deep hostname subdomain hierarchy ({sub_count} subdomain levels left of the registrable domain)"
        )
    elif total_dots >= 4:
        risk_factors.append(
            f"Multiple dot characters across the full URL ({total_dots} dots total, including path/query)"
        )

    if features.get("special_char_count", 0) >= 3:
        risk_factors.append(
            f"Contains {features['special_char_count']} special characters (-, _, %, @, =, ~)"
        )

    if features.get("url_length", 0) > 75:
        risk_factors.append(
            f"Long URL string ({features['url_length']} characters)"
        )

    # Contextual observations (explicitly qualified so HTTPS or shallow paths are never equated with guaranteed safety)
    if features.get("has_https") == 1:
        protective_factors.append(
            "Uses HTTPS transport encryption (note: HTTPS encrypts traffic in transit but does not prove domain legitimacy)"
        )

    if (
        features.get("has_ip") == 0
        and features.get("suspicious_words", 0) == 0
        and features.get("tld_risk", 0) == 0
        and features.get("is_shortener", 0) == 0
        and features.get("has_userinfo", 0) == 0
        and features.get("is_punycode", 0) == 0
    ):
        protective_factors.append(
            "Standard domain format with zero matched suspicious keywords, shorteners, or high-risk TLD flags"
        )

    if (
        features.get("has_ip") == 0
        and features.get("suspicious_words", 0) == 0
        and features.get("num_subdirs", 0) <= 1
        and features.get("num_params", 0) == 0
    ):
        protective_factors.append(
            "Simple path structure with no query parameters"
        )

    cal_prob = float(np.clip(calibrated_phish_prob, 0.0, 1.0))
    uncal_prob = float(
        np.clip(
            uncalibrated_rf_phish_prob
            if uncalibrated_rf_phish_prob is not None
            else cal_prob,
            0.0,
            1.0,
        )
    )

    return {
        "calibrated_phishing_probability": cal_prob,
        "uncalibrated_rf_phishing_probability": uncal_prob,
        "risk_factors": risk_factors,
        "protective_factors": protective_factors,
    }


class AdaptiveSecurityAgent:
    """
    Feedback-Driven Adaptive Threshold Policy for Endpoint Security Recommendations.

    Note on Terminology:
    This controller implements a heuristic, reward-guided threshold adaptation rule
    over two scalar decision boundaries (`review_threshold` and `block_threshold`).
    It does NOT retrain the underlying Random Forest classifier or learn a full
    contextual Markov Decision Process policy.
    """

    DEFAULT_REVIEW_THRESHOLD = 0.35
    DEFAULT_BLOCK_THRESHOLD = 0.70
    MIN_GAP = 0.15
    VALID_VERDICTS = {"correct", "false_positive", "false_negative"}
    VALID_LEVELS = {"Low", "Medium", "High"}
    VALID_ACTIONS = {"Allow", "Review", "Block"}

    def __init__(
        self,
        feedback_csv_path: str | Path = "results/feedback.csv",
        state_json_path: str | Path = "results/rl_policy_state.json",
    ):
        self.feedback_csv_path = Path(feedback_csv_path)
        self.state_json_path = Path(state_json_path)
        self.lock_path = self.state_json_path.with_suffix(".lock")
        self.review_threshold = self.DEFAULT_REVIEW_THRESHOLD
        self.block_threshold = self.DEFAULT_BLOCK_THRESHOLD
        self.total_episodes = 0
        self.cumulative_reward = 0.0
        self.diagnostics: list[str] = []
        self._load_or_initialize_state()

    def _sanitize_thresholds(self) -> None:
        """Enforce `0.15 <= review_threshold < block_threshold <= 0.95` with `MIN_GAP >= 0.15`."""
        r, b = self.review_threshold, self.block_threshold
        if not (np.isfinite(r) and np.isfinite(b)):
            self.diagnostics.append(
                "Repaired non-finite threshold values back to defaults (0.35, 0.70)."
            )
            r, b = self.DEFAULT_REVIEW_THRESHOLD, self.DEFAULT_BLOCK_THRESHOLD
        r = float(np.clip(r, 0.15, 0.60))
        b = float(np.clip(max(b, r + self.MIN_GAP), 0.45, 0.95))
        if b - r < self.MIN_GAP:
            r = float(np.clip(b - self.MIN_GAP, 0.15, 0.60))
        self.review_threshold, self.block_threshold = round(r, 4), round(b, 4)

    def _load_or_initialize_state(self) -> None:
        loaded_from_json = False
        if self.state_json_path.exists():
            try:
                data = json.loads(self.state_json_path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("State JSON root is not an object.")
                self.review_threshold = float(
                    data.get("review_threshold", self.DEFAULT_REVIEW_THRESHOLD)
                )
                self.block_threshold = float(
                    data.get("block_threshold", self.DEFAULT_BLOCK_THRESHOLD)
                )
                self.total_episodes = max(0, int(data.get("total_episodes", 0)))
                self.cumulative_reward = float(data.get("cumulative_reward", 0.0))
                if not np.isfinite(self.cumulative_reward):
                    self.cumulative_reward = 0.0
                self._sanitize_thresholds()
                loaded_from_json = True
            except Exception as exc:
                self.diagnostics.append(
                    f"Malformed state file {self.state_json_path.name} ({exc}); rebuilding from valid feedback.csv rows."
                )

        if not loaded_from_json and self.feedback_csv_path.exists():
            self.review_threshold = self.DEFAULT_REVIEW_THRESHOLD
            self.block_threshold = self.DEFAULT_BLOCK_THRESHOLD
            self.total_episodes = 0
            self.cumulative_reward = 0.0
            try:
                with open(self.feedback_csv_path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for line_num, row in enumerate(reader, start=2):
                        try:
                            prob = float(row.get("probability", 0.5))
                            verdict = (row.get("user_verdict") or "").strip().lower()
                            if (
                                not np.isfinite(prob)
                                or prob < 0.0
                                or prob > 1.0
                                or verdict not in self.VALID_VERDICTS
                            ):
                                self.diagnostics.append(
                                    f"Skipped malformed feedback row at line {line_num}."
                                )
                                continue
                            self._apply_reward_update(prob, verdict)
                        except Exception as row_exc:
                            self.diagnostics.append(
                                f"Skipped unreadable feedback row at line {line_num}: {row_exc}"
                            )
                self._save_state()
            except Exception as exc:
                self.diagnostics.append(
                    f"Could not replay {self.feedback_csv_path.name}: {exc}"
                )

    def _apply_reward_update(self, probability: float, verdict: str) -> float:
        """
        Probability-aware threshold update rule:
        - `correct` (+1.0 reward): gently anchors/stabilizes thresholds toward baseline defaults
          (0.35, 0.70) without crossing `probability` so confirmed decisions remain stable.
        - `false_positive` (-1.5 reward): raises thresholds proportionally to how far `probability`
          exceeded the review threshold, reducing false alarms.
        - `false_negative` (-3.0 reward): lowers thresholds proportionally to how far `probability`
          fell below the block threshold, tightening phishing sensitivity.
        """
        if not isinstance(probability, (int, float)) or not np.isfinite(probability):
            raise ValueError(f"Probability must be a finite number; got {probability!r}")
        prob = float(probability)
        if prob < 0.0 or prob > 1.0:
            raise ValueError(f"Probability must be within [0.0, 1.0]; got {prob}")

        if verdict not in self.VALID_VERDICTS:
            raise ValueError(
                f"Unknown verdict {verdict!r}; expected one of {sorted(self.VALID_VERDICTS)}"
            )

        self.total_episodes += 1
        lr = max(0.015, 0.05 / (1.0 + 0.05 * self.total_episodes))

        if verdict == "correct":
            reward = 1.0
            # Gentle stabilization toward default thresholds (20% of base step)
            drift_r = self.DEFAULT_REVIEW_THRESHOLD - self.review_threshold
            drift_b = self.DEFAULT_BLOCK_THRESHOLD - self.block_threshold
            self.review_threshold += float(np.clip(drift_r, -lr * 0.25, lr * 0.25))
            self.block_threshold += float(np.clip(drift_b, -lr * 0.25, lr * 0.25))
        elif verdict == "false_positive":
            reward = -1.5
            margin = max(0.0, prob - self.review_threshold)
            step = lr * (1.25 + margin)
            self.review_threshold += step
            self.block_threshold += step
        elif verdict == "false_negative":
            reward = -3.0
            margin = max(0.0, self.block_threshold - prob)
            step = lr * (1.75 + margin)
            self.review_threshold -= step
            self.block_threshold -= step

        self._sanitize_thresholds()
        self.cumulative_reward += reward
        return reward

    def _save_state(self) -> None:
        self._sanitize_thresholds()
        self.state_json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "policy_type": "feedback_driven_threshold_controller",
            "review_threshold": round(self.review_threshold, 4),
            "block_threshold": round(self.block_threshold, 4),
            "total_episodes": self.total_episodes,
            "cumulative_reward": round(self.cumulative_reward, 2),
            "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        tmp = self.state_json_path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.state_json_path)

    def reset_policy(self) -> dict:
        """Reset decision thresholds back to default values (`0.35`, `0.70`)."""
        with _exclusive_file_lock(self.lock_path):
            self.review_threshold = self.DEFAULT_REVIEW_THRESHOLD
            self.block_threshold = self.DEFAULT_BLOCK_THRESHOLD
            self._save_state()
        return self.evaluate_policy(0.0)

    def evaluate_policy(self, phishing_probability: float) -> dict:
        if not isinstance(phishing_probability, (int, float)) or not np.isfinite(
            phishing_probability
        ):
            raise ValueError(
                f"phishing_probability must be finite; got {phishing_probability!r}"
            )
        prob = float(np.clip(phishing_probability, 0.0, 1.0))
        self._sanitize_thresholds()

        if prob >= self.block_threshold:
            level = "High"
            action = "Block"
        elif prob >= self.review_threshold:
            level = "Medium"
            action = "Review"
        else:
            level = "Low"
            action = "Allow"

        return {
            "level": level,
            "action": action,
            "review_threshold": round(self.review_threshold, 4),
            "block_threshold": round(self.block_threshold, 4),
            "total_episodes": self.total_episodes,
            "cumulative_reward": round(self.cumulative_reward, 2),
        }

    def _inspect_existing_votes(self, normalized_url: str) -> set[str]:
        """Return the set of verdicts already recorded for `normalized_url`."""
        verdicts: set[str] = set()
        if not self.feedback_csv_path.exists():
            return verdicts
        try:
            with open(self.feedback_csv_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    row_url = redact_url_for_storage(row.get("url") or "")
                    row_verdict = (row.get("user_verdict") or "").strip().lower()
                    if row_url == normalized_url and row_verdict in self.VALID_VERDICTS:
                        verdicts.add(row_verdict)
        except Exception as exc:
            self.diagnostics.append(f"Error inspecting existing votes: {exc}")
        return verdicts

    def record_feedback(
        self,
        url: str,
        probability: float,
        level: str,
        action: str,
        user_verdict: str,
        allow_contradiction_override: bool = False,
    ) -> dict:
        """
        Validate, redact sensitive query tokens, check for duplicate/contradictory votes,
        append to `results/feedback.csv` under an exclusive file lock, and update thresholds.
        """
        if not isinstance(url, str) or not url.strip():
            raise ValueError("URL must be a non-empty string.")
        if not isinstance(probability, (int, float)) or not np.isfinite(probability):
            raise ValueError(f"Probability must be a finite float; got {probability!r}")
        prob = float(probability)
        if prob < 0.0 or prob > 1.0:
            raise ValueError(f"Probability must be within [0.0, 1.0]; got {prob}")
        if level not in self.VALID_LEVELS:
            raise ValueError(f"Invalid risk level {level!r}; expected one of {sorted(self.VALID_LEVELS)}")
        if action not in self.VALID_ACTIONS:
            raise ValueError(f"Invalid action {action!r}; expected one of {sorted(self.VALID_ACTIONS)}")
        if user_verdict not in self.VALID_VERDICTS:
            raise ValueError(f"Unknown verdict {user_verdict!r}")
        if user_verdict == "false_positive" and action == "Allow":
            raise ValueError("false_positive is invalid when recommended action is already Allow.")
        if user_verdict == "false_negative" and action == "Block":
            raise ValueError("false_negative is invalid when recommended action is already Block.")

        safe_url = redact_url_for_storage(url)

        with _exclusive_file_lock(self.lock_path):
            # Reload latest state under lock in case another session updated it
            self._load_or_initialize_state()
            prior_verdicts = self._inspect_existing_votes(safe_url)

            if user_verdict in prior_verdicts:
                return {
                    "logged_row": None,
                    "reward": 0.0,
                    "updated_policy": self.evaluate_policy(prob),
                    "duplicate": True,
                    "redacted_url": safe_url,
                }

            conflicting = prior_verdicts - {user_verdict}
            if conflicting and not allow_contradiction_override:
                raise ValueError(
                    f"Contradictory feedback rejected: URL {safe_url!r} already has recorded verdict(s) "
                    f"{sorted(conflicting)}, which conflicts with {user_verdict!r}."
                )

            self.feedback_csv_path.parent.mkdir(parents=True, exist_ok=True)
            file_exists = (
                self.feedback_csv_path.exists()
                and self.feedback_csv_path.stat().st_size > 0
            )

            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            row = {
                "time": timestamp,
                "url": safe_url,
                "probability": round(prob, 4),
                "level": level,
                "action": action,
                "user_verdict": user_verdict,
            }

            with open(self.feedback_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "time",
                        "url",
                        "probability",
                        "level",
                        "action",
                        "user_verdict",
                    ],
                )
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
                f.flush()

            reward = self._apply_reward_update(prob, user_verdict)
            self._save_state()

            return {
                "logged_row": row,
                "reward": reward,
                "updated_policy": self.evaluate_policy(prob),
                "duplicate": False,
                "redacted_url": safe_url,
            }
