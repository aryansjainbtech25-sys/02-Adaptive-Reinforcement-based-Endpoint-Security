import math
import re
from collections import Counter
from urllib.parse import urlsplit
from ipaddress import ip_address

# Keep order fixed - must match the order used during model training
feature_names = [
    "url_length",
    "num_dots",
    "has_https",
    "has_ip",
    "num_subdirs",
    "num_params",
    "suspicious_words",
    "special_char_count",
    "digits_count",
    "entropy"
]

# Keywords commonly abused in phishing URLs
suspicious_words = [
    "login", "verify", "account", "secure",
    "update", "password", "signin", "bank",
    "confirm", "wallet", "payment"
]

suspicious_pattern = re.compile(
    r"\b(?:" + "|". join(re.escape(suspicious_word) for suspicious_word in suspicious_words) + r")\b"
)

def calculate_shannon_entropy(url: str) -> float:
    # High entropy often indicates obfuscation / random strings in phishing URLs
    if not url:
        return 0.0

    count_characters = Counter(url)
    url_length = len(url)

    shannon_entropy = 0.0
    for character_count in count_characters.values():
        character_probability = character_count / url_length
        shannon_entropy -= character_probability * math.log2(character_probability)

    return shannon_entropy

def extract_url_features(URL: str) -> dict:
    if not isinstance(URL, str) or not URL:
        raise ValueError("URL must be a non-empty string.")

    URL = URL.strip()

    # urlsplit needs // to parse bare domains correctly
    if not "://" in URL:
        URL = "//" + URL

    parsed_url = urlsplit(URL)
    hostname = parsed_url.hostname or ""

    # Phishing often uses direct IP instead of domain to avoid reputation checks
    try:
        ip_address(hostname)
        has_ip = 1
    except ValueError:
        has_ip = 0

    url_length = len(URL)
    num_dots = URL.count('.')
    
    # urlsplit counts // as two slashes, so subtract 2 to get actual path depth
    num_subdirs = URL.count('/') - 2
    num_params = URL.count('?') + URL.count('&')
    special_char_count = sum(URL.count(special_character) for special_character in "-_%@=~")
    digits_count = sum(character.isdigit() for character in URL)
    shannon_entropy = calculate_shannon_entropy(URL)

    url = URL.lower()

    has_https = 1 if url.startswith("https://") else 0
    suspicious_word_count = len(suspicious_pattern.findall(url))

    url_features = {
        "url_length": url_length,
        "num_dots": num_dots,
        "has_https": has_https,
        "has_ip": has_ip,
        "num_subdirs": num_subdirs,
        "num_params": num_params,
        "suspicious_words": suspicious_word_count,
        "special_char_count": special_char_count,
        "digits_count": digits_count,
        "entropy": shannon_entropy
    }

    return url_features

def dict_to_list(URL: str) -> list:
    url_features = extract_url_features(URL)
    return [url_features[url_feature] for url_feature in feature_names]
