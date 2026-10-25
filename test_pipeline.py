from __future__ import annotations

from pathlib import Path
import pandas as pd

from adaptive_agent import (
    AdaptiveSecurityAgent,
    analyze_url_risk_factors,
    compute_model_probabilities,
    inspect_url_anatomy,
    load_model_bundle,
)
from feature_extractor import (
    extract_url_features,
    feature_names,
    is_valid_url,
    redact_url_for_storage,
)
from qr_utils import decode_qr_code

project_directory = Path(__file__).resolve().parent
model_path = project_directory / "models" / "model.joblib"
feedback_path = project_directory / "results" / "feedback.csv"
rl_state_path = project_directory / "results" / "rl_policy_state.json"


def predict_url_direct(URL: str, require_scheme: bool = True) -> dict:
    """Validate and classify a URL string directly using the offline ARES pipeline."""
    bundle = load_model_bundle(model_path)
    model_features = bundle.get("features", bundle.get("feature_names", feature_names))

    URL = URL.strip()
    url_lower = URL.lower()

    if require_scheme and not url_lower.startswith(("http://", "https://")):
        raise ValueError(
            f"The URL does not start with 'http://' or 'https://': {URL}"
        )

    if not is_valid_url(URL, require_scheme=require_scheme):
        raise ValueError(f"The payload does not contain a valid URL: {URL}")

    url_features = extract_url_features(URL)

    if list(model_features) != list(feature_names):
        raise ValueError(
            f"The model's feature names do not match the expected feature names.\n"
            f"Model features: {model_features}\n"
            f"Expected features: {feature_names}"
        )

    X = pd.DataFrame(
        [[url_features[feature_name] for feature_name in feature_names]],
        columns=feature_names,
    )

    prob_info = compute_model_probabilities(bundle, X)
    calibrated_prob = prob_info["calibrated_phishing_probability"]
    uncalibrated_rf_prob = prob_info["uncalibrated_rf_phishing_probability"]
    predicted_label = prob_info["predicted_label"]

    risk_info = analyze_url_risk_factors(
        URL, url_features, calibrated_prob, uncalibrated_rf_prob
    )

    agent = AdaptiveSecurityAgent(
        feedback_csv_path=feedback_path,
        state_json_path=rl_state_path,
    )
    policy = agent.evaluate_policy(calibrated_prob)

    return {
        "decoded_url": URL,
        "redacted_url": redact_url_for_storage(URL),
        "prediction": predicted_label,
        "class_probabilities": prob_info["calibrated_probabilities"],
        "calibrated_probabilities": prob_info["calibrated_probabilities"],
        "uncalibrated_rf_probabilities": prob_info["uncalibrated_rf_probabilities"],
        "calibrated_phishing_probability": round(calibrated_prob, 4),
        "uncalibrated_rf_phishing_probability": round(uncalibrated_rf_prob, 4),
        "risk_level": policy["level"],
        "recommended_action": policy["action"],
        "url_anatomy": inspect_url_anatomy(URL),
        "risk_factors": risk_info["risk_factors"],
        "protective_factors": risk_info["protective_factors"],
        "features": url_features,
        "model_metadata": bundle.get("model_metadata", {}),
        "warning": (
            "Experimental offline lexical prediction: a low phishing probability does not "
            "guarantee website safety. Always verify untrusted QR destinations independently."
        ),
    }


def predict_qr_code(image_path: str | Path, require_scheme: bool = True) -> dict:
    """Decode a QR-code image and run the offline ARES URL phishing analysis pipeline."""
    decoded_url = decode_qr_code(image_path).strip()
    return predict_url_direct(decoded_url, require_scheme=require_scheme)


def _print_pretty(res: dict) -> None:
    action = res["recommended_action"]
    badge = {
        "Allow": "[RECOMMENDED: ALLOW - LOWER ESTIMATED RISK]",
        "Review": "[RECOMMENDED: REVIEW - MEDIUM ESTIMATED RISK]",
        "Block": "[RECOMMENDED: BLOCK - HIGH ESTIMATED RISK]",
    }.get(action, action)
    print("=" * 76)
    print(f"ARES ADVISORY VERDICT : {badge}")
    print(f"Decoded URL           : {res['decoded_url']}")
    print(
        f"Classification        : {res['prediction']} "
        f"(Calibrated Prob: {res['calibrated_phishing_probability']:.2%}, "
        f"Uncalibrated Base RF: {res['uncalibrated_rf_phishing_probability']:.2%})"
    )
    anat = res.get("url_anatomy", {})
    print(
        f"Registrable Domain    : {anat.get('registrable_domain')} | "
        f"Subdomain: {anat.get('subdomain')} | Path: {anat.get('path')}"
    )
    if res["risk_factors"]:
        print("Triggered Risk Signals:")
        for rf in res["risk_factors"]:
            print(f"  - [RISK] {rf}")
    if res["protective_factors"]:
        print("Contextual Notes      :")
        for pf in res["protective_factors"]:
            print(f"  - [INFO] {pf}")
    print(f"Disclaimer            : {res['warning']}")
    print("=" * 76)


if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="ARES CLI — Decode QR codes or analyze URLs offline."
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        type=str,
        help="Path to a QR code image file (optional if --url or --all-samples is used)",
    )
    parser.add_argument(
        "--url",
        type=str,
        help="Analyze a URL string directly without a QR image",
    )
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Run a batch summary across all sample QR images in tests/images/",
    )
    parser.add_argument(
        "--allow-bare-domain",
        action="store_true",
        help="Allow bare domain URLs that omit http:// or https://",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Print a human-readable scorecard instead of raw JSON",
    )
    arguments = parser.parse_args()

    if arguments.all_samples:
        images_dir = project_directory / "tests" / "images"
        print(
            f"{'Image File':<28} | {'Decoded Payload':<32} | {'Calib':>6} | {'UncalRF':>7} | {'Action':<8}"
        )
        print("-" * 92)
        for img in sorted(images_dir.glob("*.png")):
            try:
                r = predict_qr_code(img, require_scheme=not arguments.allow_bare_domain)
                print(
                    f"{img.name:<28} | {r['decoded_url'][:32]:<32} | "
                    f"{r['calibrated_phishing_probability']:>5.1%} | "
                    f"{r['uncalibrated_rf_phishing_probability']:>6.1%} | "
                    f"{r['recommended_action']:<8}"
                )
            except Exception as err:
                print(f"{img.name:<28} | REJECTED ({err})")
        sys.exit(0)

    try:
        if arguments.url:
            result = predict_url_direct(
                arguments.url, require_scheme=not arguments.allow_bare_domain
            )
        elif arguments.image_path:
            result = predict_qr_code(
                arguments.image_path, require_scheme=not arguments.allow_bare_domain
            )
        else:
            parser.print_help()
            sys.exit(1)

        if arguments.pretty:
            _print_pretty(result)
        else:
            print(json.dumps(result, indent=2))
    except Exception as error:
        print(f"ARES Pipeline Error: {error}", file=sys.stderr)
        sys.exit(1)
