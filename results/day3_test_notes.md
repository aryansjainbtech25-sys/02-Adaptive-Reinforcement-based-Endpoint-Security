# ARES — Day 3 Pipeline Testing

## Objective
Verify the QR decoding, URL feature extraction, and phishing prediction pipeline.

## Test results

| Test | Expected behavior | Observed result |
|---|---|---|
| HTTPS URL | Decode and predict | Passed |
| HTTP URL | Decode and predict | Passed |
| Plain-text QR | Reject unsupported content | Passed |
| IP-address URL | Decode and predict | Passed |
| URL with query parameters | Decode and predict | Passed |

## Observations
- QR codes were decoded successfully.
- URL features were extracted and passed to the trained model.
- Unsupported plain-text content was rejected with an explanatory error.
- Predictions and class probabilities were returned for supported URLs.
- Some predictions were borderline, indicating that output should be treated cautiously.

## Limitations
The model's training dataset has known URL-formatting and collection biases. Successful execution does not establish reliable real-world phishing detection performance.

## Conclusion
The initial QR-to-URL-to-feature-extraction-to-prediction pipeline works for the tested cases. Further evaluation and testing on independent, representative data are required.
