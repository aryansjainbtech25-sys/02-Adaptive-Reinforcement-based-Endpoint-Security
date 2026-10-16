# ARES — Adaptive Reinforcement-based Endpoint Security

**A machine-learning project for QR-code phishing detection (quishing).**

ARES aims to identify potentially malicious URLs embedded in QR codes by decoding their contents, extracting URL-based features, and using a trained machine-learning model to classify the URL.

> **Project status:** Prototype in development. Predictions are experimental and must not be treated as a guarantee that a URL is safe or malicious.

## Overview

QR-code phishing, commonly known as *quishing*, uses QR codes to direct people to potentially malicious websites. Because the destination may not be visible before scanning, users can be exposed to phishing pages without recognising the risk.

ARES explores a machine-learning-based approach to this problem by connecting QR-code decoding with URL analysis and classification.

## Key Features

- Decode QR codes from image files.
- Extract URL-based features for machine-learning inference.
- Classify supported URLs using a trained model.
- Display the prediction and available class probabilities.
- Reject QR codes that do not contain supported HTTP or HTTPS URLs.
- Test the pipeline using synthetic QR-code images.

## How It Works

The current prototype follows this workflow:

1. **QR input:** The user provides a QR-code image.
2. **QR decoding:** The application extracts the encoded text.
3. **URL validation:** The prototype checks whether the decoded content begins with `http://` or `https://`.
4. **Feature extraction:** URL characteristics are calculated, including URL length, dot count, HTTPS presence, IP-address usage, path separators, query parameters, suspicious words, special characters, digit count, and Shannon entropy.
5. **Model inference:** The extracted features are passed to the trained classifier.
6. **Result:** The application reports the predicted class and, when supported, class probabilities.

The current classifier uses ten URL-based features. QR-image classification is a separate task and is not performed by this URL-feature model.

## Repository Structure

```text
ARES/
├── app.py
├── feature_extractor.py
├── qr_utils.py
├── test_pipeline.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   └── model.joblib
├── notebooks/
│   ├── ares_model_development.ipynb
│   └── ares_model_verification.ipynb
├── tests/
│   ├── test_feature_extractor.py
│   └── images/
├── results/
└── data/
```

Additional utility scripts and test files may be included as the project develops.

## Dataset

The project uses a URL dataset containing phishing and benign examples, with URL-derived features used for model training and evaluation.

The dataset includes labels and features such as URL length, HTTPS presence, IP-address usage, path separators, query parameters, suspicious words, special characters, digit count, and entropy.

The project also explores a separate QR-code dataset containing QR images represented as NumPy arrays and associated labels.

**Dataset handling:** Raw datasets are not necessarily included in this repository. Please obtain them from their original sources, follow the applicable licences and terms, and place the required files in the appropriate local data directory.

## Technology Stack

- Python
- pandas and NumPy
- scikit-learn
- OpenCV
- joblib
- Streamlit (for the application interface)

See `requirements.txt` for the project's Python dependencies.

## Getting Started

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ARES
```

Replace the placeholder with your actual repository URL.

### 2. Create and activate a virtual environment

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Check the trained model

Ensure that the trained model is available at:

```text
models/model.joblib
```

The model file must match the feature extraction logic and feature order expected by the application. Only load model files from trusted sources because joblib files can execute code when loaded.

### 5. Run the QR pipeline test

For example, after placing the supplied test image in `tests/images/`:

```bash
python test_pipeline.py tests/images/https_url.png
```

The pipeline should decode the QR code, extract URL features, and display a prediction or a clear error message.

### 6. Launch the application

If `app.py` is implemented as a Streamlit application, run:

```bash
streamlit run app.py
```

Follow the local URL printed in the terminal to open the interface in your browser.

## Testing

The initial synthetic test cases cover:

- QR codes containing HTTPS URLs.
- QR codes containing HTTP URLs.
- QR codes containing plain text.
- URLs containing an IP address.
- URLs containing query parameters.

These tests check basic decoding, feature extraction, prediction, and error handling. They do not establish real-world detection accuracy.

## Current Limitations

- The training dataset has a severe class imbalance and known URL-formatting and data-collection biases.
- Initial evaluation produced suspiciously perfect results, which may indicate that the model learned dataset-specific shortcuts rather than general phishing patterns.
- Predictions on synthetic examples are functional checks, not proof of classification correctness.
- Model probabilities are not necessarily calibrated estimates of real-world risk.
- HTTPS alone does not establish that a website is trustworthy.
- The current URL-feature classifier does not directly classify QR-code pixels.
- Independent evaluation on representative, unseen data is still required.

ARES should be treated as an educational research prototype, not as a production security product. Do not rely on its output alone when deciding whether to visit a website.

## Future Improvements

- Evaluate the model using independent, representative data.
- Improve class balancing and investigate data leakage and source bias.
- Compare additional classification models and evaluation metrics.
- Integrate QR decoding, URL analysis, and a user-friendly interface.
- Add appropriate feedback and monitoring mechanisms.
- Investigate a separate QR-image classification model if sufficient time and data are available.

## Academic Project

ARES is developed as part of an Artificial Intelligence and Machine Learning semester project at RV University.

## Disclaimer

This project is intended for educational and research purposes. It does not guarantee the detection of phishing websites and must not be used as the sole basis for security decisions.
