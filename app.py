# Importing the annotations module from the __future__ package to enable the use of postponed evaluation of type annotations.
# This allows for the use of forward references in type hints, which can be useful for defining types that are not yet defined at the time of annotation.
# Example: def foo(x: 'MyClass') -> 'MyClass': ...
from __future__ import annotations

# Importing the tempfile module to create temporary files and directories. This is useful for creating temporary storage that can be automatically cleaned up after use.
import tempfile

# Importing the streamlit module, which is a framework for building interactive web applications in Python. It allows for the creation of data-driven applications with minimal code.
import streamlit as st

import numpy as np
import pandas as pd

import joblib

from pathlib import Path

# Importing the decode_qr_code from the qr_utils module. This function is likely used to decode QR codes from images or other sources.
from qr_utils import decode_qr_code

# Importing the feature_names and extract_url_features from the feature_extractor module. These are likely used to extract features from URLs for further analysis or processing.
from feature_extractor import feature_names, extract_url_features

model_path = Path("models/model.joblib")

# Defining a list of common image file types that may be used in the application. This list can be used to filter or validate image files based on their extensions.
image_file_types = ["png", "jpg", "jpeg"]

# Defining the maximum image size to be uploaded in megabytes. This is used to limit the size of images that can be processed by the application.
max_image_size = 5

# Configuring the Streamlit application with a title and an icon. This sets the title of the web application and the favicon that appears in the browser tab.
st.set_page_config(page_title = "ARES - QR Code Phishing Detector", page_icon = "🔍", layout = "centered")
# page_title is the browser-tab title
# page_icon is the favicon that appears in the browser tab
# layout can be set to "centered" or "wide" to control the layout of the application. "centered" centers the content, while "wide" allows the content to span the full width of the page.

# Defining a decorator that tells the Streamlit application to cache the results of the function it decorates. This can improve performance by avoiding redundant computations for the same inputs.
@st.cache_resource(show_spinner = "Loading ARES model...")

# Defining a function to load the machine learning model from a specified path. This function uses joblib to load the model and returns it for use in the application.
def load_bundle(model_path: Path) -> dict:
    # Checking if the model file exists at the specified path. If it does not exist, a FileNotFoundError is raised.
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Loading the model using joblib
    bundle = joblib.load(model_path)

    # Checking if the loaded bundle is a dictionary. If it is not, a ValueError is raised.
    if not isinstance(bundle, dict):
        raise ValueError(f"Loaded bundle is not a dictionary. Got {type(bundle)} instead.")

    # Required keys that must exist inside the saved model bundle.
    required_features = {"model", "features", "label_mapping"}

    # Find required keys that are missing from the loaded bundle.
    missing_features = required_features - set(bundle.keys())

    # If there are any missing features, raise a ValueError with a message indicating which features are missing.
    if missing_features:
        raise ValueError(f"Loaded bundle is missing required features: {missing_features}")

    # Checking if the features in the loaded bundle match the expected feature names. If they do not match, a ValueError is raised.
    if list(bundle["features"]) != list(feature_names):
        raise ValueError(f"Loaded bundle features do not match expected feature names.")

    # Checking if the number of features in the loaded bundle matches the expected number of features. If they do not match, a ValueError is raised.
    if bundle["model"].n_features_in_ != len(feature_names):
        raise ValueError(f"Loaded model features do not match expected number of features.")

    return bundle

# Defining a function to predict whether a URL is phishing or benign based on the extracted features. This function takes a URL and a model bundle as input and returns the prediction and the associated probability.
def predict_url(bundle: dict, URL: str) -> tuple[str, dict[str, float]]:
    # Calling the extract_url_features function to extract features from the provided URL. The extracted features are stored in a dictionary.
    url_features = extract_url_features(URL)

    # Defining a dataframe with the extracted features. This dataframe is used as input to the machine learning model for prediction.
    df = pd.DataFrame([url_features], columns = feature_names)

    model = bundle["model"]

    label_encoding = bundle["label_mapping"]

    # Making a prediction using the loaded model and the extracted features. The predicted label is obtained by calling the predict method of the model.
    predicted_index = int(model.predict(df)[0])

    # Getting the predicted label from the label encoding dictionary using the predicted index.
    predicted_label = label_encoding.get(predicted_index, str(predicted_index))

    # Creating an empty dictionary to store the probabilities of each class (phishing or benign) for the given URL.
    probabilities: dict[str, float] = {}

    # Checking if the model has the predict_proba method, which is used to obtain the probabilities of each class. If the method exists, the probabilities are calculated and stored in the probabilities dictionary.
    if hasattr(model, "predict_proba"):
        # Getting the predicted probabilities for each class using the predict_proba method of the model.
        predicted_probabilities = model.predict_proba(df)[0]

        # Pairing each class label with its corresponding probability and storing them in the probabilities dictionary.
        for class_index, class_probability in zip(model.classes_, predicted_probabilities):
            # Converting the class label into a readable class name
            class_name = label_encoding.get(int(class_index), str(class_index))

            probabilities[class_name] = float(class_probability)

    return predicted_label, probabilities

# Displaying the title of the application in the Streamlit interface. This is the main heading that users will see when they access the application.
st.title("ARES - QR Code Phishing Detector")

# Explaining QR-Code Phishing to the user
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

# Displaying a horizontal line to seperate sections
st.divider()

# Creating an image uploader
uploaded_image = st.file_uploader(
    "Upload a QR-Code image",
    type = image_file_types,
    accept_multiple_files = False,
    help = f"PNG, JPG, or JPEG - up to {max_image_size} MB"
)

# If the user has uploaded an image
if uploaded_image is not None:
    # Checking the maximum uploaded size of the image
    if uploaded_image.size > max_image_size * (1024 ** 2):
        # Showing a red error image
        st.error(f"Uploaded image size is larger than {max_image_size} MB.")

        # Stopping the rest of the streamlit application from running
        st.stop()

    # Showing the uploaded image on the page
    st.image(uploaded_image, caption = uploaded_image.name, use_container_width = False)

# Generating the Analyze QR-Code button, that is disabled until the user uploads the image.
analyze_btn = st.button(
    "🔍 Analyze QR code", type = "primary", disabled = (uploaded_image is None), use_container_width = True
)

# Running the analysis when the analyze button is clicked and an image is uploaded
if analyze_btn and uploaded_image is not None:
    # Trying to load the saved model
    try:
        bundle = load_bundle(model_path)

    # Checking if the model file is missing, show a FileNotFound error
    except (FileNotFoundError, ValueError) as error:
        st.error(f"❌ Model error: {error}")

        st.stop()

    # Showing a temporary loading message while the QR-Code is being decoded
    with st.spinner("Decoding QR-Code..."):
        try:
            # Getting the uploaded file extension or assuming it is a PNG file
            suffix = Path(uploaded_image.name).suffix or ".png"

            # Creating a file with the correct extension and delete it later
            with tempfile.NamedTemporaryFile(suffix = suffix, delete = False) as file:
                # Writing the uploaded image bytes into the temporary file
                file.write(uploaded_image.getbuffer())

                # Saving the temporary file path
                temporary_file_path = Path(file.name)

            url = decode_qr_code(temporary_file_path)

        # If no readable QR-Code is found, show a ValueError
        except ValueError as error:
            st.error(f"❌ Could not decode a QR code from this image: {error}")

            st.stop()

        # This block of code always runs
        finally:
            try:
                # Deleting the temporary file path
                temporary_file_path.unlink(missing_ok = True)

            except Exception:
                # Preventing the page from crashing if the deletion fails
                pass

    # Displaying the decoded URL
    st.success("✅ QR code decoded")

    # Adding a markdown header for the decoded URL
    st.markdown("**URL:**")

    # Checking if the decoded url starts with http:// or https://, and warning the user to not open it directly
    if url.lower().startswith(("http://", "https://")):
        st.caption(
            "⚠️ Do not open this link in a browser from here. "
            "It is displayed as plain text on purpose."
        )

    # If it does not, then warn the user that it may be a domain.
    else:
        st.caption("The decoded URL is a bare domain.")

    # Showing a loading message as the URL is classified
    with st.spinner("Classifying decoded URL..."):
        try:
            predicted_label, predicted_probabilities = predict_url(bundle, url)

        # Checking if the prediction fails, show an error
        except Exception as error:
            st.error(f"❌ Prediction failed: {error}")

            st.stop()

    # Seperating the decoded URL and the prediction
    st.divider()

    # Adding a markdown header for Prediciton
    st.markdown("### Prediction")

    # Checking if the predicted label is phishing
    if predicted_label.lower().startswith("phish"):
        prediction_status = "error"

        status_message = f"🚨 {predicted_label}"

    elif predicted_label.lower().startswith("benign"):
        prediction_status = "complete"
                
        status_message = f"✅ {predicted_label}"

    # For other labels, show neutral information
    else:
        prediction_status = "error"
                
        status_message = f"Predicted class: {predicted_label}"

    # Displaying the overall predicition status
    with st.status("✅ Prediction complete", state = prediction_status, expanded = True):
        st.write(status_message)

        # Showing the decoded URL in a readable, copyable code block
        st.markdown("**Decoded URL:**")
        
        st.code(url, language = None)

    # Showing class probabilities if supported by the model
    if predicted_probabilities:
        st.markdown("**Class Probabilities:**")

        # Sorting probabilities in descending order
        sorted_probabilities = dict(sorted(predicted_probabilities.items(), key = lambda kv: kv[1], reverse = True))
        # key=lambda kv: kv[1] — sorts by the probability, not the class name

        # Displaying one progress bar for each class probability
        for name, probability in sorted_probabilities.items():
            st.progress(
                min(max(probability, 0.0), 1.0), # Keeping the probability between 0 & 1
                text = f"{name}: {probability:.2%}"
            ) 

    # If probabilities are not available
    else:
        st.caption("This model does not show per-class probabilities.")

    # Creating a full-width button to analyze another image
    if st.button("🔄 Analyze Another Image", use_container_width = True):
        # Removing the uploaded image from Streamlit's temporary session rate
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        # Rerunning the application
        st.rerun()