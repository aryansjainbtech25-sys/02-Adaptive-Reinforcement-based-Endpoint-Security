from __future__ import annotations

import tempfile
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from qr_utils import decode_qr_code
from feature_extractor import feature_names, extract_url_features

model_path = Path("models/model.joblib")

image_file_types = ["png", "jpg", "jpeg"]
max_image_size = 5 # in MB

st.set_page_config(page_title = "ARES - QR Code Phishing Detector", page_icon = "🔍", layout = "centered")

@st.cache_resource(show_spinner = "Loading ARES model...")
def load_bundle(model_path: Path) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    bundle = joblib.load(model_path)

    if not isinstance(bundle, dict):
        raise ValueError(f"Loaded bundle is not a dictionary. Got {type(bundle)} instead.")

    required_features = {"model", "features", "label_mapping"}
    missing_features = required_features - set(bundle.keys())

    if missing_features:
        raise ValueError(f"Loaded bundle is missing required features: {missing_features}")

    # Validate bundle was trained with the same feature schema we use at inference
    if list(bundle["features"]) != list(feature_names):
        raise ValueError(f"Loaded bundle features do not match expected feature names.")

    if bundle["model"].n_features_in_ != len(feature_names):
        raise ValueError(f"Loaded model features do not match expected number of features.")

    return bundle

def predict_url(bundle: dict, URL: str) -> tuple[str, dict[str, float]]:
    url_features = extract_url_features(URL)
    df = pd.DataFrame([url_features], columns = feature_names)

    model = bundle["model"]
    label_encoding = bundle["label_mapping"]

    predicted_index = int(model.predict(df)[0])
    predicted_label = label_encoding.get(predicted_index, str(predicted_index))

    probabilities: dict[str, float] = {}

    # Only some classifiers expose calibrated probabilities
    if hasattr(model, "predict_proba"):
        predicted_probabilities = model.predict_proba(df)[0]

        for class_index, class_probability in zip(model.classes_, predicted_probabilities):
            class_name = label_encoding.get(int(class_index), str(class_index))
            probabilities[class_name] = float(class_probability)

    return predicted_label, probabilities

st.title("ARES - QR Code Phishing Detector")

st.markdown(
    """
**Quishing** (QR-code phishing) hides a malicious URL inside a QR code
that victims scan with their phone. Because the destination URL is not
visible until *after* scanning, a QR code on a poster, menu, or email
can silently redirect to a credential-stealing site.

Upload a QR-code image below. ARES decodes the embedded URL and runs it
through a machine-learning classifier trained to recognise phishing
patterns in URLs.
    """
)

st.divider()

uploaded_image = st.file_uploader(
    "Upload a QR-Code image",
    type = image_file_types,
    accept_multiple_files = False,
    help = f"PNG, JPG, or JPEG - up to {max_image_size} MB"
)

if uploaded_image is not None:
    if uploaded_image.size > max_image_size * (1024 ** 2):
        st.error(f"Uploaded image size is larger than {max_image_size} MB.")
        st.stop()

    st.image(uploaded_image, caption = uploaded_image.name, use_container_width = False)

analyze_btn = st.button(
    "🔍 Analyze QR code", type = "primary", disabled = (uploaded_image is None), use_container_width = True
)

if analyze_btn and uploaded_image is not None:
    try:
        bundle = load_bundle(model_path)
    except (FileNotFoundError, ValueError) as error:
        st.error(f"❌ Model error: {error}")
        st.stop()

    with st.spinner("Decoding QR-Code..."):
        try:
            suffix = Path(uploaded_image.name).suffix or ".png"

            # Use temp file because decode_qr_code expects a filesystem path
            with tempfile.NamedTemporaryFile(suffix = suffix, delete = False) as file:
                file.write(uploaded_image.getbuffer())
                temporary_file_path = Path(file.name)

            url = decode_qr_code(temporary_file_path)

        except ValueError as error:
            st.error(f"❌ Could not decode a QR code from this image: {error}")
            st.stop()

        finally:
            try:
                temporary_file_path.unlink(missing_ok = True)
            except Exception:
                pass

    st.success("✅ QR code decoded")
    st.markdown("**URL:**")

    if url.lower().startswith(("http://", "https://")):
        st.caption(
            "⚠ Do not open this link in a browser from here. "
            "It is displayed as plain text on purpose."
        )
    else:
        st.caption("The decoded URL is a bare domain.")

    with st.spinner("Classifying decoded URL..."):
        try:
            predicted_label, predicted_probabilities = predict_url(bundle, url)
        except Exception as error:
            st.error(f"❌ Prediction failed: {error}")
            st.stop()

    st.divider()
    st.markdown("### Prediction")

    if predicted_label.lower().startswith("phish"):
        prediction_status = "error"
        status_message = f"🚨 {predicted_label}"
    elif predicted_label.lower().startswith("benign"):
        prediction_status = "complete"
        status_message = f"✅ {predicted_label}"
    else:
        prediction_status = "error"
        status_message = f"Predicted class: {predicted_label}"

    with st.status("✅ Prediction complete", state = prediction_status, expanded = True):
        st.write(status_message)
        st.markdown("**Decoded URL:**")
        st.code(url, language = None)

    if predicted_probabilities:
        st.markdown("**Class Probabilities:**")

        sorted_probabilities = dict(sorted(predicted_probabilities.items(), key = lambda kv: kv[1], reverse = True))

        for name, probability in sorted_probabilities.items():
            st.progress(
                min(max(probability, 0.0), 1.0),
                text = f"{name}: {probability:.2%}"
            ) 
    else:
        st.caption("This model does not show per-class probabilities.")

    if st.button("🔄 Analyze Another Image", use_container_width = True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        st.rerun()
