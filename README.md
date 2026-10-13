# 02-Adaptive-Reinforcement-based-Endpoint-Security

## Phishing URL Detection

This project uses machine learning to classify URLs as **phishing** or **benign/legitimate** based on structural and statistical features extracted from the URL.

The dataset used in this project is obtained from Kaggle.

## Project Overview

Phishing websites often use deceptive URLs to trick users into revealing sensitive information such as passwords, banking details, or personal data. This project analyzes URL-based features and trains a machine learning model to predict whether a URL is phishing or benign.

## Dataset

The dataset contains URLs along with engineered features that describe their structure and behavior.

### Main Columns

| Column | Description |
|---|---|
| `url` | The original URL |
| `label` | Target variable: phishing or benign |
| `url_length` | Total length of the URL |
| `num_dots` | Number of dots in the URL |
| `has_https` | Whether the URL uses HTTPS |
| `has_ip` | Whether the URL contains an IP address |
| `num_subdirs` | Number of subdirectories |
| `num_params` | Number of URL parameters |
| `suspicious_words` | Count of suspicious keywords |
| `tld` | Top-level domain |
| `special_char_count` | Number of special characters |
| `digits_count` | Number of digits |
| `entropy` | Character-level entropy of the URL |

### Target Variable

- `label = 1`: Phishing URL  
- `label = 0`: Benign/legitimate URL  

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Kaggle Notebooks / Jupyter Notebook

## Workflow

1. Load the Kaggle dataset.
2. Inspect column names, data types, and sample records.
3. Check for missing values.
4. Analyze the phishing/benign class distribution.
5. Select relevant URL features.
6. Split the dataset into training and testing sets.
7. Train a machine learning classification model.
8. Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.

## Data Splitting

The dataset is split using an 80/20 ratio:

- 80% for training
- 20% for testing

Stratified sampling is used to preserve the phishing/benign class distribution in both sets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

## How to Run

1. Clone this repository:

```bash
git clone [https://github.com/your-username/phishing-url-detection.git](https://github.com/your-username/phishing-url-detection.git)
cd phishing-url-detection
```

2. Install the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib
```

3. Open the notebook:

```bash
jupyter notebook phishing_url_detection.ipynb
```

4. Place the Kaggle dataset in the correct input folder or update the file path in the notebook.

5. Run all notebook cells.

## Results

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

These metrics are important because phishing detection is a classification problem where both false positives and false negatives matter.

## Future Improvements

- Use natural language processing on the raw URL text.
- Add WHOIS, DNS, and webpage-content features.
- Compare multiple machine learning models.
- Perform hyperparameter tuning.
- Deploy the model as a web application or API.

## License

This project is for educational purposes.
