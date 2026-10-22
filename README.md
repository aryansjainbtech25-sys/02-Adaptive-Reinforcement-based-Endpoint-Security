# ARES — Adaptive Reinforcement-based Endpoint Security

**QR-Code Phishing Detection using Machine Learning**

ARES is a machine-learning-based prototype designed to identify potentially malicious URLs embedded in QR codes. It decodes a QR-code image, extracts URL-based features, and uses a trained classification model to predict whether the URL is **Benign** or **Phishing**.

> **Disclaimer:** ARES is an academic prototype developed for educational and research purposes. Its predictions are not a guarantee of website safety. Do not use it as the sole method for deciding whether a URL is trustworthy.

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [System Workflow](#system-workflow)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Features Extracted](#features-extracted)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Testing](#testing)
- [Model Evaluation](#model-evaluation)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Contributors](#contributors)

## Project Overview

QR codes are widely used to share website links, payment links, and other information. However, attackers can exploit QR codes to direct users to phishing websites that attempt to steal credentials or sensitive information. This technique is commonly known as **quishing (QR-code phishing)**.

ARES provides a prototype workflow for analyzing QR codes and classifying the URLs they contain. The project combines QR-code decoding, feature extraction, and machine learning in a single pipeline.

## Objectives

- Decode QR codes from uploaded image files.
- Extract useful URL characteristics for classification.
- Detect potentially suspicious URLs using a trained machine-learning model.
- Display the prediction and extracted features.
- Provide a simple interface for testing QR-code images.
- Evaluate the pipeline using synthetic QR-code test cases.

## System Workflow

1. **QR-code input:** The user supplies an image containing a QR code.
2. **QR decoding:** The system extracts the encoded text or URL.
3. **Input validation:** The pipeline checks whether the decoded content begins with `http://` or `https://`.
4. **Feature extraction:** The URL is converted into numerical and categorical features.
5. **Model inference:** The saved machine-learning model predicts a class.
6. **Result display:** The application presents the prediction and, when available, class probabilities and extracted features.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | QR-code decoding and image processing |
| Pandas | Feature data preparation |
| NumPy | Numerical operations |
| Scikit-learn | Machine-learning model and prediction |
| Joblib | Saving and loading the trained model |
| Streamlit | Web-based user interface |
| Jupyter Notebook | Model development and verification |
| Git and GitHub | Version control and project hosting |

## Project Structure

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
│       ├── https_url.png
│       ├── http_url.png
│       ├── ip_url.png
│       ├── plain_text.png
│       └── query_url.png
├── results/
│   └── day4_app_testing.md
└── data/
```

The `data/` directory is intended for datasets used during development. Do not commit large datasets or externally sourced data unless their licensing and redistribution terms permit it.

## Dataset

The model-development dataset contains URL records labeled as benign or phishing.

The dataset used during development contains the following classes:

| Class | Label | Number of records |
|---|---:|---:|
| Benign | 0 | 820 |
| Phishing | 1 | 159,244 |
| **Total** | | **160,064** |

The dataset is highly imbalanced, with substantially more phishing samples than benign samples. This imbalance must be considered when interpreting model performance.

Dataset sources used during development included phishing URL feeds and lists of popular websites. Verify the applicable source terms and include the relevant dataset references in the project report before publication.

## Features Extracted

ARES extracts URL characteristics that may help the model distinguish suspicious URLs from benign ones.

| Feature | Description |
|---|---|
| `url_length` | Length of the URL |
| `num_dots` | Number of dot characters |
| `has_https` | Indicates whether the URL uses HTTPS |
| `has_ip` | Indicates whether the hostname contains an IP address |
| `num_subdirs` | Number of URL path or slash-related components, according to the implemented extraction rule |
| `num_params` | Number of URL query parameters |
| `suspicious_words` | Count of configured suspicious terms |
| `tld` | Extracted top-level domain |
| `special_char_count` | Count of selected special characters |
| `digits_count` | Number of numeric digits |
| `entropy` | Character-distribution entropy of the URL |

**Important:** Feature definitions used during model training must match the definitions used during prediction. In particular, confirm how `num_subdirs`, missing TLD values, and suspicious-word counts are handled before reporting final evaluation results.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/aryansjainbtech25-sys/02-Adaptive-Reinforcement-based-Endpoint-Security.git
cd ARES
```

### 2. Create a virtual environment

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ensure that the dependencies in `requirements.txt` are compatible with the environment used to train and save `models/model.joblib`.

## Running the Application

Start the Streamlit application from the project root:

```bash
streamlit run app.py
```

Open the local URL printed in the terminal by Streamlit. Upload a QR-code image and run the analysis.

If the Streamlit interface is not yet implemented in `app.py`, the command will not provide the intended application until that file is completed.

### Running a QR-code test from the command line

The pipeline can be tested with an image file:

```bash
python test_pipeline.py tests/images/https_url.png
```

Other sample images can be tested similarly:

```bash
python test_pipeline.py tests/images/http_url.png
python test_pipeline.py tests/images/ip_url.png
python test_pipeline.py tests/images/plain_text.png
python test_pipeline.py tests/images/query_url.png
```

The expected behavior depends on the contents of each QR code. A QR code containing plain text rather than an HTTP or HTTPS URL should be rejected by the URL-validation step.

## Testing

Run the feature-extraction tests from the project root:

```bash
python -m pytest tests/test_feature_extractor.py -v
```

If `pytest` is not installed, add it to the development dependencies or install it in the active virtual environment.

Test cases should cover:

- Valid HTTPS URLs.
- HTTP URLs.
- URLs containing IP addresses.
- URLs containing query parameters.
- QR codes containing plain text.
- Invalid or unreadable QR-code images.
- Missing model files and invalid input paths.
- Consistency between training and inference feature names and order.

Record actual outcomes in `results/day4_app_testing.md`. Do not mark a test as passed unless it has been run successfully.

## Model Evaluation

The project includes notebooks for model development and verification:

- `notebooks/ares_model_development.ipynb`
- `notebooks/ares_model_verification.ipynb`

Evaluation should include the confusion matrix, precision, recall, F1-score, and balanced accuracy. Precision-recall metrics are especially important because the dataset contains far fewer benign samples than phishing samples.

A perfect score on a random train/test split does not necessarily indicate real-world effectiveness. The dataset may contain source-related patterns or differences in feature construction that allow a model to distinguish the classes without learning robust phishing indicators.

Before presenting final performance claims:

- Verify the feature-extraction formulas.
- Check for data leakage and source-specific patterns.
- Evaluate on a separate, representative dataset.
- Use predicted probabilities—not hard class predictions—to calculate probability-based metrics such as average precision.
- Document the limitations of the test data.

## Limitations

- The model analyzes URL characteristics rather than the full content or behavior of a website.
- Legitimate websites can contain suspicious-looking URL patterns.
- Phishing URLs can be short or appear legitimate.
- The training dataset is highly imbalanced.
- Dataset collection methods can introduce bias.
- A model trained on historical examples may not recognize newly emerging attacks.
- HTTPS indicates an encrypted connection, not that a website is trustworthy.
- Class probabilities are model outputs and should not automatically be interpreted as calibrated real-world risk probabilities.

## Future Enhancements

- Improve dataset balance and diversity.
- Validate the model on independent and up-to-date datasets.
- Add URL parsing and more robust hostname and domain features.
- Investigate feature importance and model explainability.
- Add confidence thresholds and a cautious inconclusive-result category.
- Improve handling of malformed URLs and unsupported QR-code content.
- Add logging and reproducible evaluation reports.
- Explore adaptive learning and reinforcement-learning techniques if supported by a clearly defined environment, feedback signal, and evaluation protocol.

## Contributors

Developed as part of an AIML Semester 3 academic project at RV University.

Add the names of the project members and their contributions here.

## Disclaimer

ARES is an educational research prototype. It may produce false positives and false negatives. Always verify URLs independently and follow appropriate cybersecurity practices. A benign classification does not guarantee that a website is safe.
