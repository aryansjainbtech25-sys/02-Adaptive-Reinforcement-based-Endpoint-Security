import pandas as pd

import joblib

# Importing the decode_qr_code function from the qr_utils module to decode QR codes and extract URLs
from qr_utils import decode_qr_code

# Importing the extract_url_features function and feature_names list from the feature_extractor module to extract features from URLs
from feature_extractor import extract_url_features, feature_names

# Sample path to the pre-trained machine learning model used for phishing detection
model_path = "models/model.joblib"

# Defining a function to predict the QR-Code
def predict_qr_code(image_path: str) -> dict:
    # Loading the ML model from the specified path using joblib as model, artifact, or bundle
    bundle = joblib.load(model_path)

    # Checking if the loaded bundle is a dictionary containing the model and feature names
    if isinstance(bundle, dict):
        # Extracting the model and feature names from the bundle
        model = bundle["model"]

        # Extracting the feature names from the bundle, or using the default feature_names if not present
        model_features = bundle.get("feature_names", feature_names)

        # Extracting the label encoding from the bundle, or using a default mapping if not present
        label_encoding = bundle.get("label_mapping", {0: "Benign", 1: "Phishing"})

    # If the loaded bundle is not a dictionary, assume it is a model and use default feature names and label encoding
    else:
        # Using the loaded bundle as the model
        model = bundle

        # Using the default feature names for the model
        model_features = feature_names

        # Using a default label encoding mapping for the model
        label_encoding = {0: "Benign", 1: "Phishing"}

    # Decoding the QR code from the provided image path to extract the URL
    URL = decode_qr_code(image_path)

    # Do not send arbitrary URLs to the URL classifier
    URL = URL.strip()  # Remove leading/trailing whitespace from the URL

    # Lowercase the URL for consistent feature extraction
    url = URL.lower()

    # If the URL does not start with "http://" or "https://", raise a ValueError
    if not url.startswith(("http://", "https://")):
        raise ValueError(
            f"The QR-Code does not start with 'http://' or 'https://': {URL}"
        )

    # Extracting features from the URL using the extract_url_features function
    url_features = extract_url_features(URL)

    # Checking if the list of model features does not match the feature_names, and raising a ValueError if they do not match
    if list(model_features) != feature_names:
        raise ValueError(
            f"The model's feature names do not match the expected feature names.\n"
            f"Model features: {model_features}\n"
            f"Expected features: {feature_names}"
        )

    # Saving X as the input DataFrame for the model, containing the extracted URL features
    X = pd.DataFrame([[url_features[feature_name] for feature_name in feature_names]], columns=feature_names)

    # Making a prediction using the loaded model on the input DataFrame X
    prediction = int(model.predict(X)[0])

    # Saving the resulting dictionary
    result = {
        "decoded_url": URL,  # The decoded URL from the QR code
        "prediction": label_encoding.get(prediction, str(prediction)),  # The prediction result
        "features": url_features,  # The extracted features from the URL
        "warning": "Experimental prediction: the current training dataset has a known URL-formatting shortcut."
    }

    # Checking if the model has a predict_proba method, and if so, adding the prediction probabilities to the result dictionary
    if hasattr(model, "predict_proba"):
        # Saving the prediction probabilities for each class (Benign and Phishing) in the result dictionary
        probabilities = model.predict_proba(X)[0]

        # Adding the class probabilities to the result dictionary, mapping class labels to their corresponding probabilities
        result["class_probabilities"] = dict(zip(model.classes_.tolist(), probabilities.tolist()))

    return result  # Returning the result dictionary containing the decoded URL, prediction, features, and any warnings

if __name__ == "__main__":
    import json
    
    # Importing the argparse module to handle command-line arguments for the script
    import argparse

    # Importing the sys module to handle system-specific parameters and functions
    import sys

    # Creating an argument parser to handle command-line arguments for the script
    parser = argparse.ArgumentParser(description = "Decode a QR code and test the ARES phishing detector.")

    # Adding a positional argument for the image path of the QR code to be decoded and tested
    parser.add_argument("image_path", type=str, help="Path to the QR code image")

    # Parsing the command-line arguments
    arguments = parser.parse_args()

    # Try to predict the QR code using the provided image path and print the result as a JSON string
    try:
        # Calling the predict_qr_code function with the provided image path to get the prediction result
        result = predict_qr_code(arguments.image_path)

        # Printing the prediction result as a formatted JSON string
        print(json.dumps(result, indent = 2))

    # Catching any exceptions that occur during the prediction process and printing an error message to stderr
    except Exception as error:
        # Printing the error message to stderr with a prefix indicating that it is an ARES Pipeline Error
        print(f"ARES Pipeline Error: {error}", file = sys.stderr)

        sys.exit(1)  # Exiting the script with a non-zero exit code to indicate an error occurred