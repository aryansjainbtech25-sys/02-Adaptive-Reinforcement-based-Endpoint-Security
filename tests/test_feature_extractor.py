import unittest

from feature_extractor import (
    calculate_shannon_entropy,
    dict_to_list,
    extract_registrable_domain,
    extract_url_features,
    feature_names,
    is_valid_url,
    parse_host_structure,
    redact_url_for_storage,
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
    assert features["num_dots"] == 2
    assert features["num_subdirs"] == 1
    assert features["num_params"] == 0
    assert features["subdomain_count"] == 1


def test_bare_domain_does_not_inflate_length_or_entropy():
    features = extract_url_features("google.com")

    assert features["url_length"] == 10
    assert features["has_https"] == 0
    assert features["num_subdirs"] == 0
    assert abs(features["entropy"] - calculate_shannon_entropy("google.com")) < 1e-12


def test_ip_url_is_detected():
    features = extract_url_features("http://192.168.1.1/login")
    assert features["has_ip"] == 1
    assert features["has_https"] == 0

    ipv6_features = extract_url_features("http://[2001:db8::1]:8080/admin")
    assert ipv6_features["has_ip"] == 1
    assert ipv6_features["has_port"] == 1


def test_suspicious_words_are_counted_in_paths_and_compound_domains():
    features = extract_url_features("http://secure-login.example.com/verify-account")
    assert features["suspicious_words"] == 4
    assert features["susp_token_hits"] == 4
    assert features["susp_in_subdomain"] == 1

    compound = extract_url_features(
        "http://forum.uk.securebankinggroup.com/107519/8af662"
    )
    assert compound["suspicious_words"] == 2
    assert compound["susp_compound_hits"] == 2

    obfuscated = extract_url_features("https://mmettamaskilogin.webflow.io/")
    assert obfuscated["suspicious_words"] == 1


def test_query_parameters_and_encoded_redirects():
    features = extract_url_features("https://example.com/page?user=1&token=abc")
    assert features["num_params"] == 2
    assert features["embedded_url_in_query"] == 0

    single_enc = extract_url_features(
        "https://example.com/out?redirect=https%3A%2F%2Fevil.xyz%2Flogin"
    )
    assert single_enc["embedded_url_in_query"] == 1
    assert single_enc["num_subdirs"] == 1

    double_enc = extract_url_features(
        "https://example.com/out?url=http%253a%252f%252f192.0.2.1"
    )
    assert double_enc["embedded_url_in_query"] == 1


def test_centralized_registrable_domain_and_multipart_suffixes():
    assert extract_registrable_domain("https://login.PayPal.co.uk:8443/path") == "paypal.co.uk"
    assert extract_registrable_domain("http://a.b.portal.ac.in/test") == "portal.ac.in"
    assert extract_registrable_domain("http://user:pass@sub.example.com.au./p") == "example.com.au"
    assert extract_registrable_domain("http://[::1]:8000/test") == "::1"

    host_info = parse_host_structure("https://a.b.example.co.uk/file.v1.v2.tar.gz")
    assert host_info["subdomain_labels"] == ["a", "b"]
    assert host_info["registrable_domain"] == "example.co.uk"
    assert host_info["host_dot_count"] == 4


def test_query_redaction_preserves_structure_and_hides_secrets():
    redacted = redact_url_for_storage(
        "https://admin:SecretPass@Example.com:443/login?sessionid=abcdef123456&lang=en#frag123"
    )
    assert "SecretPass" not in redacted
    assert "abcdef123456" not in redacted
    assert "frag123" not in redacted
    assert "sessionid=[REDACTED]" in redacted
    assert "lang=en" in redacted


def test_dict_to_list_returns_ordered_features():
    url = "https://example.com/login?user=1"
    features = dict_to_list(url)

    assert len(features) == len(feature_names) == 24
    expected = extract_url_features(url)
    expected_list = [expected[name] for name in feature_names]
    assert features == expected_list


def test_url_validation_helper():
    assert is_valid_url("https://example.com", require_scheme=True) is True
    assert is_valid_url("http://192.0.2.1/login", require_scheme=True) is True
    assert is_valid_url("www.ecslimited.com/environ.cfm", require_scheme=False) is True
    assert is_valid_url("www.ecslimited.com/environ.cfm", require_scheme=True) is False
    assert is_valid_url("Hello ARES", require_scheme=False) is False
    assert is_valid_url("javascript:alert(1)", require_scheme=False) is False
    assert is_valid_url("", require_scheme=False) is False


def test_empty_and_non_string_urls_raise_error():
    for bad in ("", "   ", 123, None):
        try:
            extract_url_features(bad)  # type: ignore[arg-type]
            assert False, f"Expected ValueError for {bad!r}"
        except ValueError:
            pass


class TestFeatureExtractorSuite(unittest.TestCase):
    def test_all_feature_extractor_checks(self):
        test_entropy_is_zero_for_repeated_characters()
        test_entropy_is_positive_for_varied_characters()
        test_entropy_is_zero_for_empty_string()
        test_extract_features_for_https_url()
        test_bare_domain_does_not_inflate_length_or_entropy()
        test_ip_url_is_detected()
        test_suspicious_words_are_counted_in_paths_and_compound_domains()
        test_query_parameters_and_encoded_redirects()
        test_centralized_registrable_domain_and_multipart_suffixes()
        test_query_redaction_preserves_structure_and_hides_secrets()
        test_dict_to_list_returns_ordered_features()
        test_url_validation_helper()
        test_empty_and_non_string_urls_raise_error()


if __name__ == "__main__":
    unittest.main()
