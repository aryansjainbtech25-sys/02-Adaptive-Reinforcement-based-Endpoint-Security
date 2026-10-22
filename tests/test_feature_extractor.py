import pytest

from feature_extractor import (
    calculate_shannon_entropy,
    extract_url_features,
    dict_to_list,
    feature_names
)

def test_entropy_is_zero_for_repeated_characters():
    assert calculate_shannon_entropy("aaaa") == 0.0

def test_entropy_is_positive_for_varied_characters():
    entropy = calculate_shannon_entropy("abcdef")
    assert entropy > 0

def test_entropy_is_zero_for_empty_string():
    assert calculate_shannon_entropy("") == 0.0

def test_extract_features_for_https_url():
    features = extract_url_features("https://www.google.com/search")

    assert features["url_length"] == len("https://www.google.com/search")
    assert features["has_https"] == 1
    assert features["has_ip"] == 0
    assert features["num_dots"] == 3
    assert features["num_params"] == 0

def test_ip_url_is_detected():
    features = extract_url_features("http://192.168.1.1/login")

    assert features["has_ip"] == 1
    assert features["has_https"] == 0

def test_suspicious_words_are_counted():
    features = extract_url_features(
        "http://secure-login.example.com/verify-account"
    )
    assert features["suspicious_words"] == 4

def test_query_parameters_are_counted():
    features = extract_url_features(
        "https://example.com/page?user=1&token=abc"
    )
    assert features["num_params"] == 2

def test_dict_to_list_returns_ordered_features():
    url = "https://example.com/login?user=1"

    features = dict_to_list(url)

    assert len(features) == len(feature_names)

    expected = extract_url_features(url)
    expected_list = [expected[name] for name in feature_names]

    assert features == expected_list

def test_empty_url_raises_error():
    with pytest.raises(ValueError):
        extract_url_features("")

def test_non_string_url_raises_error():
    with pytest.raises(ValueError):
        extract_url_features(123)
