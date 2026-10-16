# Importing the math module to perform mathematical operations
import math

# Importing the regular-expressions module for pattern-based text searching
import re

# Importing the Counter class from the collections module to count the number of occurrences of each character in a URL
from collections import Counter

# Importing the urlsplit function from the urllib.parse module to split a URL into its components
from urllib.parse import urlsplit

# Importing the ip_address module from the ipaddress module to check if a URL is an IP address
from ipaddress import ip_address

# Defining an ordered list of all feature names to be extracted from a URL
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

# Defining a list of suspicious words that may indicate a phishing URL
suspicious_words = [
    "login", "verify", "account", "secure",
    "update", "password", "signin", "bank",
    "confirm", "wallet", "payment"
]

# Defining a function to calculate the Shannon entropy (randomness of a piece of text) in a URL
def calculate_shannon_entropy(url: str) -> float:
    # Checking if url is empty and returning 0.0 if it is
    if not url:
        return 0.0

    # Counting the occurrences of each character in the URL
    count_characters = Counter(url)

    # Storing the length of the URL
    url_length = len(url)

    shannon_entropy = 0.0
    # Shannon entropy formula: H(X) = -summation(p(x) * log2(p(x))) for each unique character x in the URL
    for character_count in count_characters.values():
        # Calculating the probability of each character in the URL
        character_probability = character_count / url_length

        # Calculating the contribution of each character to the Shannon entropy and subtracting it from the total entropy
        shannon_entropy -= character_probability * math.log2(character_probability)

    return shannon_entropy

# Defining a function to extract numerical features from a given URL
def extract_url_features(URL: str) -> dict:
    # Checking if the URL is an empty string and raising a ValueError if it is
    if not isinstance(URL, str) or not URL:
        raise ValueError("URL must be a non-empty string.")

    URL = URL.strip()  # Removing leading and trailing whitespace from the URL

    # To parse hostnames even when the URL is not complete, we can add a scheme (like 'http://') to the URL before parsing it.
    if not "://" in URL:
        URL = "//" + URL  # Adding a scheme to the URL for proper parsing

    # Splitting the URL into its components using urlsplit
    parsed_url = urlsplit(URL)

    # Extracting the hostname from the parsed URL
    hostname = parsed_url.hostname or ""

    # Trying to interpret the hostname as an IP address to check if it is a valid IP address
    try:
        ip_address(hostname)

        has_ip = 1  # If the hostname is a valid IP address, set has_ip to 1

    # If the hostname is not a valid IP address, set has_ip to 0
    except ValueError:
        has_ip = 0

    # Saving the length of the URL for feature extraction
    url_length = len(URL)

    # Saving the number of dots in the URL for feature extraction
    num_dots = URL.count('.')

    # Saving the number of subdirectories in the URL for feature extraction
    num_subdirs = URL.count('/') - 2  # Subtracting 2 to account for the '//' in the URL

    # Saving the number of query parameters in the URL for feature extraction
    num_params = URL.count('?') + URL.count('&')  # Counting both '?' and '&' to get the total number of parameters

    # Saving the number of special characters in the URL for feature extraction
    special_char_count = sum(URL.count(special_character) for special_character in "-_%@=~")

    # Saving the number of digits in the URL for feature extraction
    digits_count = sum(character.isdigit() for character in URL)

    # Calculating the Shannon entropy of the URL for feature extraction
    shannon_entropy = calculate_shannon_entropy(URL)

    # Saving the URL in lowercase for consistent feature extraction
    url = URL.lower()

    # Checking if the URL uses HTTPS and saving the result for feature extraction
    has_https = 1 if url.startswith("https://") else 0

    # Counting the occurrences of suspicious words in the URL for feature extraction
    suspicious_word_count = sum(
        # Using re.findall to find all occurrences of each suspicious word in the URL and summing them up
        len(re.findall(rf"\b{suspicious_word}\b", url)) for suspicious_word in suspicious_words
        # r signifies a raw string, which treats backslashes as literal characters, and \b is a word boundary anchor that ensures we only match whole words
        # re.findall() returns a list of all non-overlapping matches of the pattern in the string
    )

    # Saving all the features in a dictionary for easy access and manipulation
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

# Defining a function to convert the extracted features into a list format for easier processing
def dict_to_list(URL: str) -> list:
    # Calling the extract_url_features function to get the features of the URL in dictionary format
    url_features = extract_url_features(URL)

    # Returning the features as a list in the order specified by the feature_names list
    return [url_features[url_feature] for url_feature in feature_names]