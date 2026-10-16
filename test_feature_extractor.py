# test_feature_extractor.py

# Import pytest to write and run test functions
import pytest

# Import the functions you want to test from feature_extractor.py
from feature_extractor import (
    calculate_shannon_entropy,
    extract_url_features,
    dict_to_list,
    feature_names
)


# Test that entropy is zero for a URL containing only one repeated character
def test_entropy_is_zero_for_repeated_characters():
    assert calculate_shannon_entropy("aaaa") == 0.0


# Test that entropy is positive for a URL with varied characters
def test_entropy_is_positive_for_varied_characters():
    entropy = calculate_shannon_entropy("abcdef")

    # A string with six equally likely characters should have positive entropy
    assert entropy > 0


# Test that entropy is zero for an empty string
def test_entropy_is_zero_for_empty_string():
    assert calculate_shannon_entropy("") == 0.0


# Test feature extraction for a normal HTTPS URL
def test_extract_features_for_https_url():
    features = extract_url_features("https://www.google.com/search")

    # Check individual expected feature values
    assert features["url_length"] == len("https://www.google.com/search")
    assert features["has_https"] == 1
    assert features["has_ip"] == 0
    assert features["num_dots"] == 3
    assert features["num_params"] == 0


# Test that an IP-address URL is detected correctly
def test_ip_url_is_detected():
    features = extract_url_features("http://192.168.1.1/login")

    assert features["has_ip"] == 1
    assert features["has_https"] == 0


# Test that suspicious words are counted
def test_suspicious_words_are_counted():
    features = extract_url_features(
        "http://secure-login.example.com/verify-account"
    )

    # "secure", "login", "verify", and "account" should each be counted
    assert features["suspicious_words"] == 4


# Test that query parameters are counted
def test_query_parameters_are_counted():
    features = extract_url_features(
        "https://example.com/page?user=1&token=abc"
    )

    # One '?' and one '&' should produce a count of 2
    assert features["num_params"] == 2


# Test that the feature list has the expected length and order
def test_dict_to_list_returns_ordered_features():
    url = "https://example.com/login?user=1"

    features = dict_to_list(url)

    # There should be one value for each feature name
    assert len(features) == len(feature_names)

    # The list order should match the dictionary values
    expected = extract_url_features(url)
    expected_list = [expected[name] for name in feature_names]

    assert features == expected_list


# Test that an empty URL raises a ValueError
def test_empty_url_raises_error():
    with pytest.raises(ValueError):
        extract_url_features("")


# Test that a non-string URL raises a ValueError
def test_non_string_url_raises_error():
    with pytest.raises(ValueError):
        extract_url_features(123)