from __future__ import annotations

import math
import re
from collections import Counter
from ipaddress import ip_address
from urllib.parse import SplitResult, parse_qsl, urlencode, urlsplit, urlunsplit

# Keep order fixed - must match the exact 24-feature order in models/model.joblib
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
    "entropy",
    "host_length",
    "path_length",
    "query_length",
    "subdomain_count",
    "digit_ratio",
    "susp_token_hits",
    "susp_compound_hits",
    "susp_in_subdomain",
    "is_shortener",
    "is_punycode",
    "has_port",
    "has_userinfo",
    "embedded_url_in_query",
    "tld_risk",
]

# Keywords commonly abused in phishing URLs (static lexical heuristic list)
suspicious_words = [
    "login",
    "verify",
    "account",
    "secure",
    "update",
    "password",
    "signin",
    "bank",
    "confirm",
    "wallet",
    "payment",
]

suspicious_pattern = re.compile(
    "(?:" + "|".join(re.escape(word) for word in suspicious_words) + ")",
    re.IGNORECASE,
)

# Common English words containing a suspicious keyword substring that are benign
BENIGN_SUPERWORDS = (
    "accountant",
    "accounting",
    "accountability",
    "databank",
    "foodbank",
    "blood-bank",
    "bankruptcy",
    "bankroll",
    "riverbank",
    "snowbank",
    "updated-news",
    "securely",
    "securitas",
    "insecure",
    "walletcard",
)

# Known URL shortening services (static heuristic list; does not cover all custom shorteners)
URL_SHORTENERS = {
    "bit.ly",
    "t.co",
    "tinyurl.com",
    "goo.gl",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "rebrand.ly",
    "cutt.ly",
    "shorturl.at",
    "t.ly",
    "rb.gy",
    "tiny.cc",
    "lnkd.in",
    "qr.io",
    "qrco.de",
}

# Coarse static prior of frequently abused TLDs (replaceable with empirical feed)
HIGH_RISK_TLDS = {
    "zip",
    "mov",
    "top",
    "xyz",
    "click",
    "link",
    "gq",
    "tk",
    "ml",
    "cf",
    "ga",
    "work",
    "support",
    "country",
    "kim",
    "loan",
    "men",
    "rest",
    "cfd",
    "sbs",
}

# Expanded table of ICANN multi-label public suffixes for offline registrable-domain extraction.
# Note: Does not include private hosting suffixes (e.g., github.io, webflow.io) without an external PSL file.
_MULTI_PART_SUFFIXES = {
    "co.uk",
    "org.uk",
    "ac.uk",
    "gov.uk",
    "me.uk",
    "net.uk",
    "com.au",
    "net.au",
    "org.au",
    "edu.au",
    "gov.au",
    "co.in",
    "net.in",
    "org.in",
    "ac.in",
    "gov.in",
    "res.in",
    "gen.in",
    "co.jp",
    "ne.jp",
    "or.jp",
    "ac.jp",
    "go.jp",
    "com.br",
    "net.br",
    "org.br",
    "gov.br",
    "com.cn",
    "net.cn",
    "org.cn",
    "edu.cn",
    "gov.cn",
    "com.hk",
    "org.hk",
    "edu.hk",
    "gov.hk",
    "com.sg",
    "org.sg",
    "edu.sg",
    "gov.sg",
    "com.my",
    "org.my",
    "edu.my",
    "gov.my",
    "co.nz",
    "net.nz",
    "org.nz",
    "govt.nz",
    "co.za",
    "org.za",
    "gov.za",
    "ac.za",
    "com.mx",
    "org.mx",
    "gob.mx",
    "edu.mx",
    "com.tr",
    "org.tr",
    "edu.tr",
    "gov.tr",
    "co.kr",
    "or.kr",
    "ac.kr",
    "go.kr",
    "co.il",
    "org.il",
    "ac.il",
    "gov.il",
    "com.tw",
    "org.tw",
    "edu.tw",
    "gov.tw",
    "com.ar",
    "org.ar",
    "gob.ar",
    "edu.ar",
    "com.co",
    "org.co",
    "edu.co",
    "gov.co",
    "com.pe",
    "org.pe",
    "gob.pe",
    "edu.pe",
    "com.ng",
    "org.ng",
    "edu.ng",
    "gov.ng",
    "com.pk",
    "org.pk",
    "edu.pk",
    "gov.pk",
    "com.bd",
    "org.bd",
    "edu.bd",
    "gov.bd",
    "com.vn",
    "org.vn",
    "edu.vn",
    "gov.vn",
    "co.th",
    "or.th",
    "ac.th",
    "go.th",
    "co.id",
    "or.id",
    "ac.id",
    "go.id",
    "com.ph",
    "org.ph",
    "edu.ph",
    "gov.ph",
}

_DOMAIN_LABEL_RE = re.compile(r"^[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$")

_SENSITIVE_PARAM_RE = re.compile(
    r"(token|key|pass|pwd|secret|auth|session|sid|code|jwt|id|user|mail|acct|cred|hash|sig)",
    re.IGNORECASE,
)

# Matches embedded http(s):// or // redirects in raw, single-encoded (%3A%2F%2F), or double-encoded (%253A%252F%252F) query/fragment
_EMBEDDED_URL_RE = re.compile(
    r"(?:https?(?::|%3a|%253a)(?:/|%2f|%252f){2}|(?:^|[?&=#])(?:redirect|url|next|dest|target|return|continue|goto)=(?:/|%2f|%252f){2}[a-z0-9])",
    re.IGNORECASE,
)


def canonical_split(raw: str) -> SplitResult:
    """
    Parse `raw` using consistent browser-inspired pre-normalization:
    - Strips ASCII whitespace CR/LF/TAB characters that browsers elide.
    - Converts backslashes '\\' in authority/path (before '?' or '#') into '/' so
      `http://192.168.1.1\\@google.com` resolves the authority the same way a browser does.
    - Adds a temporary '//' prefix for bare domains so `urlsplit` populates `netloc` and `hostname`.
    """
    cleaned = raw.strip().replace("\t", "").replace("\r", "").replace("\n", "")
    if re.search(r"[?#]", cleaned):
        head, sep, tail = re.split(r"([?#])", cleaned, maxsplit=1)
    else:
        head, sep, tail = cleaned, "", ""
    head = head.replace("\\", "/")
    cleaned = head + sep + tail
    target = cleaned if "://" in head else f"//{cleaned}"
    return urlsplit(target)


def _host_parts(hostname: str) -> tuple[list[str], str]:
    """
    Return `(subdomain_labels, registrable_domain)` using the offline multi-part suffix table.
    Handles IPv4/IPv6 addresses, IDN/punycode normalization, and trailing root dots.
    """
    clean_host = (hostname or "").strip().lower().rstrip(".")
    if not clean_host:
        return [], ""

    try:
        ip_address(clean_host)
        return [], clean_host
    except ValueError:
        pass

    try:
        ascii_host = clean_host.encode("idna").decode("ascii")
    except UnicodeError:
        ascii_host = clean_host

    labels = [lbl for lbl in ascii_host.split(".") if lbl]
    if len(labels) <= 2:
        return [], ".".join(labels)
    n = 3 if ".".join(labels[-2:]) in _MULTI_PART_SUFFIXES else 2
    if len(labels) < n:
        return [], ".".join(labels)
    return labels[:-n], ".".join(labels[-n:])


def extract_registrable_domain(url_or_host: str) -> str:
    """
    Centralized helper to extract the canonical registrable domain (or IP literal)
    from any URL or hostname string. Used across feature extraction, UI anatomy,
    and group-split model training to guarantee zero parser discrepancy.
    """
    if not isinstance(url_or_host, str) or not url_or_host.strip():
        return ""
    try:
        parsed = canonical_split(url_or_host)
        hostname = (parsed.hostname or "").strip().lower().rstrip(".")
    except ValueError:
        hostname = ""
    if not hostname:
        # Fallback only if urlsplit failed on malformed bracket/port syntax
        raw_head = (
            url_or_host.strip()
            .lower()
            .removeprefix("https://")
            .removeprefix("http://")
        )
        hostname = re.split(r"[/?#]", raw_head, maxsplit=1)[0].rsplit("@", 1)[-1].split(":")[0].rstrip(".")
    _, registrable = _host_parts(hostname)
    return registrable or hostname


def parse_host_structure(url: str) -> dict:
    """
    Parse a URL into validated host and structural components using the single
    canonical parser.
    """
    parsed = canonical_split(url)
    raw_hostname = (parsed.hostname or "").rstrip(".")
    try:
        ip_address(raw_hostname)
        is_ip = True
    except ValueError:
        is_ip = False

    try:
        ascii_host = (
            raw_hostname.encode("idna").decode("ascii")
            if raw_hostname and not is_ip
            else raw_hostname.lower()
        )
    except UnicodeError:
        ascii_host = raw_hostname.lower()

    subs, registrable = (
        _host_parts(ascii_host) if not is_ip else ([], raw_hostname.lower())
    )
    tld = registrable.rsplit(".", 1)[-1] if ("." in registrable and not is_ip) else ""
    try:
        port = parsed.port
    except ValueError:
        port = None

    return {
        "parsed": parsed,
        "hostname": raw_hostname,
        "ascii_host": ascii_host,
        "is_ip": is_ip,
        "subdomain_labels": subs,
        "registrable_domain": registrable,
        "tld": tld,
        "host_dot_count": ascii_host.count("."),
        "port": port,
    }


def redact_url_for_storage(url: str) -> str:
    """
    Normalize a URL for feedback storage/deduplication while redacting potentially
    sensitive query parameter values (session tokens, IDs, credentials, or long values)
    and stripping URL userinfo passwords.
    """
    if not isinstance(url, str) or not url.strip():
        return ""
    cleaned = url.strip()
    has_scheme = re.match(r"^https?://", cleaned, re.IGNORECASE) is not None
    try:
        parsed = canonical_split(cleaned)
        scheme = parsed.scheme.lower() if has_scheme else ""
        hostname = (parsed.hostname or "").lower().rstrip(".")
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        try:
            port = parsed.port
        except ValueError:
            port = None
        if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
            port = None
        netloc = f"{hostname}:{port}" if port else hostname
        if "@" in (parsed.netloc or ""):
            netloc = f"[REDACTED_USER]@{netloc}"

        path = parsed.path or ""
        redacted_query = ""
        if parsed.query:
            pairs = parse_qsl(parsed.query, keep_blank_values=True)
            safe_pairs = []
            for k, v in pairs:
                if v and (_SENSITIVE_PARAM_RE.search(k) or len(v) >= 6 or any(c.isdigit() for c in v)):
                    safe_pairs.append((k, "[REDACTED]"))
                else:
                    safe_pairs.append((k, v))
            redacted_query = urlencode(safe_pairs, safe="[]")

        fragment = "[REDACTED_FRAGMENT]" if parsed.fragment else ""
        if scheme:
            return urlunsplit((scheme, netloc, path, redacted_query, fragment))
        rebuilt = f"{netloc}{path}"
        if redacted_query:
            rebuilt += f"?{redacted_query}"
        if fragment:
            rebuilt += f"#{fragment}"
        return rebuilt
    except Exception:
        return re.sub(r"=[^&#]+", "=[REDACTED]", cleaned)


def is_valid_url(candidate: str, require_scheme: bool = False) -> bool:
    """
    Validate decoded QR text as an http(s) URL or (when allowed) a bare domain.
    Accepts ports, IPv6 literals, IDN (U-label or xn--), a trailing root dot, and query
    strings containing ':' or '://'. Rejects whitespace/control chars and other schemes.
    """
    if not isinstance(candidate, str):
        return False
    cleaned = candidate.strip()
    if not cleaned or any(ch.isspace() or ord(ch) < 32 for ch in cleaned):
        return False

    m = re.match(r"^([a-zA-Z][a-zA-Z0-9+.-]*)://", cleaned)
    if m and m.group(1).lower() not in ("http", "https"):
        return False
    has_http_scheme = m is not None
    if require_scheme and not has_http_scheme:
        return False
    if not has_http_scheme:
        authority = re.split(r"[/?#\\]", cleaned, maxsplit=1)[0]
        if ":" in authority and not re.fullmatch(
            r"[^:\[\]]+:\d{1,5}|\[[0-9a-fA-F:.]+\](:\d{1,5})?", authority
        ):
            return False

    try:
        parsed = canonical_split(cleaned)
        hostname = parsed.hostname or ""
        _ = parsed.port
    except ValueError:
        return False
    if not hostname:
        return False

    try:
        ip_address(hostname)
        return True
    except ValueError:
        pass

    hostname = hostname.rstrip(".")
    if not hostname or ".." in hostname:
        return False
    try:
        ascii_host = hostname.encode("idna").decode("ascii")
    except UnicodeError:
        return False
    labels = ascii_host.split(".")
    if len(labels) < 2 or not all(_DOMAIN_LABEL_RE.match(lbl) for lbl in labels):
        return False
    return len(labels[-1]) >= 2 and not labels[-1].isdigit()


def _keyword_hits(text: str) -> tuple[int, int]:
    """Return `(whole_token_hits, compound_substring_hits)` after masking benign superwords."""
    text = text.lower()
    for w in BENIGN_SUPERWORDS:
        text = text.replace(w, " ")
    tokens = [t for t in re.split(r"[^a-z0-9]+", text) if t]
    token_hits = sum(t in suspicious_words for t in tokens)
    compound = sum(
        1
        for t in tokens
        if t not in suspicious_words
        for w in suspicious_words
        if w in t
    )
    return token_hits, compound


def calculate_shannon_entropy(url: str) -> float:
    """Calculate the Shannon character entropy (in bits) of a URL string."""
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
    """Extract all 24 numerical model features from a URL string (in `feature_names` order)."""
    if not isinstance(URL, str) or not URL.strip():
        raise ValueError("URL must be a non-empty string.")

    raw_url = URL.strip()
    lower_url = raw_url.lower()
    host_info = parse_host_structure(raw_url)
    parsed = host_info["parsed"]
    hostname = host_info["hostname"]
    ascii_host = host_info["ascii_host"]
    is_ip = host_info["is_ip"]
    subs = host_info["subdomain_labels"]
    registrable = host_info["registrable_domain"]
    tld = host_info["tld"]

    path = parsed.path or ""
    query = parsed.query or ""
    fragment = parsed.fragment or ""

    # Count directory slashes strictly on the path component (never query/fragment/scheme)
    num_subdirs = path.count("/")
    num_params = len([p for p in query.split("&") if p]) if query else 0
    special_char_count = sum(raw_url.count(ch) for ch in "-_%@=~")
    digits_count = sum(ch.isdigit() for ch in raw_url)

    tok_hits, compound_hits = _keyword_hits(raw_url)
    sub_tok, sub_comp = _keyword_hits(".".join(subs)) if subs else (0, 0)
    suspicious_total = tok_hits + compound_hits

    query_and_frag = f"{query}#{fragment}".lower()
    has_embedded_redirect = 1 if _EMBEDDED_URL_RE.search(query_and_frag) else 0

    return {
        "url_length": len(raw_url),
        "num_dots": raw_url.count("."),
        "has_https": 1 if lower_url.startswith("https://") else 0,
        "has_ip": 1 if is_ip else 0,
        "num_subdirs": num_subdirs,
        "num_params": num_params,
        "suspicious_words": suspicious_total,
        "special_char_count": special_char_count,
        "digits_count": digits_count,
        "entropy": calculate_shannon_entropy(raw_url),
        "host_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query) + len(fragment),
        "subdomain_count": len(subs),
        "digit_ratio": digits_count / max(len(raw_url), 1),
        "susp_token_hits": tok_hits,
        "susp_compound_hits": compound_hits,
        "susp_in_subdomain": 1 if (sub_tok + sub_comp) else 0,
        "is_shortener": 1 if registrable in URL_SHORTENERS else 0,
        "is_punycode": 1
        if any(lbl.startswith("xn--") for lbl in ascii_host.split("."))
        else 0,
        "has_port": 1 if host_info["port"] is not None else 0,
        "has_userinfo": 1 if ("@" in (parsed.netloc or "")) else 0,
        "embedded_url_in_query": has_embedded_redirect,
        "tld_risk": 1 if tld in HIGH_RISK_TLDS else 0,
    }


def dict_to_list(URL: str) -> list:
    """Return the extracted URL features in the exact order defined by `feature_names`."""
    url_features = extract_url_features(URL)
    return [url_features[url_feature] for url_feature in feature_names]
