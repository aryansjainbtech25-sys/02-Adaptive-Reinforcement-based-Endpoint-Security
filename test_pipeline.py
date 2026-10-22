import pandas as pd
import joblib
from pathlib import Path

from qr_utils import decode_qr_code
from feature_extractor import extract_url_features, feature_names

project_directory = Path(__file__).resolve().parent
model_path = project_directory / "models" / "model.joblib"

def predict_qr_code(image_path: str) -> dict:
    bundle = joblib.load(model_path)

    if isinstance(bundle, dict):
        model = bundle["model"]
        model_features = bundle.get("feature_names", feature_names)
        label_encoding = bundle.get("label_mapping", {0: "Benign", 1: "Phishing"})
    else:
        # Support legacy bundles that saved only the model object
        model = bundle
        model_features = feature_names
        label_encoding = {0: "Benign", 1: "Phishing"}

    URL = decode_qr_code(image_path)

    URL = URL.strip()

    url = URL.lower()

    # Enforce URL scheme to prevent non-URL QR content hitting the classifier
    if not url.startswith(("http://", "https://")):
        raise ValueError(
            f"The QR-Code does not start with 'http://' or 'https://': {URL}"
        )

    url_features = extract_url_features(URL)

    if list(model_features) != feature_names:
        raise ValueError(
            f"The model's feature names do not match the expected feature names.\n"
            f"Model features: {model_features}\n"
            f"Expected features: {feature_names}"
        )

    X = pd.DataFrame([[url_features[feature_name] for feature_name in feature_names]], columns=feature_names)

    prediction = int(model.predict(X)[0])

    result = {
        "decoded_url": URL,
        "prediction": label_encoding.get(prediction, str(prediction)),
        "features": url_features,
        "warning": "Experimental prediction: the current training dataset has a known URL-formatting shortcut."
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        result["class_probabilities"] = dict(zip(model.classes_.tolist(), probabilities.tolist()))

    return result

if __name__ == "__main__":
    import json
    import argparse
    import sys

    parser = argparse.ArgumentParser(description = "Decode a QR code and test the ARES phishing detector.")
    parser.add_argument("image_path", type=str, help="Path to the QR code image")
    arguments = parser.parse_args()

    try:
        result = predict_qr_code(arguments.image_path)
        print(json.dumps(result, indent = 2))
    except Exception as error:
        print(f"ARES Pipeline Error: {error}", file = sys.stderr)
        sys.exit(1)
