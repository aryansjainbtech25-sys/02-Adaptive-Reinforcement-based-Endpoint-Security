# ARES — Day 4 Application Testing

## Objective

Integrate the QR-code decoding and URL classification pipeline into a Streamlit web application.

## Implemented Features

- QR-code image upload.
- QR content decoding.
- HTTP/HTTPS URL validation.
- URL feature extraction.
- Model prediction and class probabilities.
- Display of extracted features.
- User-facing error handling.
- Experimental prediction warning.

## Test Results

| Test case | Result |
|---|---|
| HTTPS QR code | Pass |
| HTTP QR code | Pass |
| Plain-text QR code | Pass |
| IP-address URL | Pass |
| URL with query parameters | Pass |
| Unreadable or invalid image | Pass |
| Unsupported file type | Pass |

## Limitations

The model's training data has known URL-formatting and collection biases. Successful application execution does not establish reliable real-world phishing detection performance. Model predictions are experimental.
