from __future__ import annotations

import json
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from adaptive_agent import (
    AdaptiveSecurityAgent,
    analyze_url_risk_factors,
    compute_model_probabilities,
    inspect_url_anatomy,
    load_model_bundle,
)
from feature_extractor import (
    extract_url_features,
    feature_names,
    is_valid_url,
    redact_url_for_storage,
)
from qr_utils import QRDecodeError, decode_qr_codes, generate_qr_image

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "model.joblib"
FEEDBACK_PATH = PROJECT_DIR / "results" / "feedback.csv"
RL_STATE_PATH = PROJECT_DIR / "results" / "rl_policy_state.json"
BASELINE_PATH = PROJECT_DIR / "results" / "baseline.json"
SAMPLE_IMAGES_DIR = PROJECT_DIR / "tests" / "images"

IMAGE_FILE_TYPES = ["png", "jpg", "jpeg"]
MAX_IMAGE_SIZE_MB = 5

SAMPLE_CATALOG = {
    "Low-Risk HTTPS — Google (test_qr.png)": {
        "file": "test_qr.png",
        "hint": "Standard HTTPS root domain (https://google.com)",
        "needs_bare": False,
    },
    "Low-Risk HTTPS — Example (https_url.png)": {
        "file": "https_url.png",
        "hint": "Standard HTTPS domain (https://example.com)",
        "needs_bare": False,
    },
    "Low-Risk HTTP — Example (http_url.png)": {
        "file": "http_url.png",
        "hint": "HTTP domain without suspicious tokens (http://example.com)",
        "needs_bare": False,
    },
    "Suspicious Login Query (query_url.png)": {
        "file": "query_url.png",
        "hint": "HTTPS login path with query parameter (https://example.com/login?id=123)",
        "needs_bare": False,
    },
    "High-Risk IP Phishing (ip_url.png)": {
        "file": "ip_url.png",
        "hint": "Direct IPv4 address hosting a /login path (http://192.0.2.1/login)",
        "needs_bare": False,
    },
    "Bare Domain Dataset Sample (sample_benign_qr_code02.png)": {
        "file": "sample_benign_qr_code02.png",
        "hint": "Bare domain without http/https scheme (www.ecslimited.com/environ.cfm)",
        "needs_bare": True,
    },
    "Non-URL Plain Text (plain_text.png)": {
        "file": "plain_text.png",
        "hint": "Plain text message ('Hello ARES') — demonstrates non-URL rejection",
        "needs_bare": False,
    },
}

FEATURE_METADATA = {
    "url_length": ("Lexical", "Total character length of the URL"),
    "num_dots": ("Structure", "Number of dot ('.') separators across the full URL"),
    "has_https": ("Protocol", "Uses HTTPS transport scheme (1 = Yes, 0 = No; does not prove domain trust)"),
    "has_ip": ("Host Risk", "Hostname is a raw IP address instead of a domain (1 = Yes, 0 = No)"),
    "num_subdirs": ("Structure", "Number of path directory slashes ('/')"),
    "num_params": ("Structure", "Number of query parameters in the URL"),
    "suspicious_words": ("Keywords", "Total phishing keyword matches (whole tokens + compound words)"),
    "special_char_count": ("Lexical", "Count of special characters (-, _, %, @, =, ~)"),
    "digits_count": ("Lexical", "Count of numeric digits (0-9) in the URL"),
    "entropy": ("Lexical", "Shannon character-distribution entropy in bits"),
    "host_length": ("Structure", "Character length of the hostname portion"),
    "path_length": ("Structure", "Character length of the URL path component"),
    "query_length": ("Structure", "Character length of query string and fragment"),
    "subdomain_count": ("Structure", "Number of subdomain levels left of the registrable domain"),
    "digit_ratio": ("Lexical", "Ratio of numeric digits to total URL length"),
    "susp_token_hits": ("Keywords", "Phishing keywords appearing as standalone tokens"),
    "susp_compound_hits": ("Keywords", "Phishing keywords embedded inside compound words"),
    "susp_in_subdomain": ("Keywords", "Phishing keyword appears inside the subdomain prefix (1 = Yes, 0 = No)"),
    "is_shortener": ("Host Risk", "Domain matches the static URL-shortener list (1 = Yes, 0 = No)"),
    "is_punycode": ("Host Risk", "Hostname uses Punycode/IDN 'xn--' encoding (1 = Yes, 0 = No)"),
    "has_port": ("Protocol", "URL specifies an explicit network port (1 = Yes, 0 = No)"),
    "has_userinfo": ("Host Risk", "Authority section contains '@' credential syntax (1 = Yes, 0 = No)"),
    "embedded_url_in_query": ("Host Risk", "Query/fragment contains an embedded redirect URL pattern"),
    "tld_risk": ("Host Risk", "Top-level domain is in the static high-abuse TLD list (1 = Yes, 0 = No)"),
}

st.set_page_config(
    page_title="ARES — QR Code Phishing Detector",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading ARES calibrated model...")
def get_cached_bundle(model_path: Path) -> dict:
    return load_model_bundle(model_path)


def run_url_analysis(bundle: dict, url: str, agent: AdaptiveSecurityAgent) -> dict:
    url_features = extract_url_features(url)
    df = pd.DataFrame([url_features], columns=feature_names)

    prob_info = compute_model_probabilities(bundle, df)
    calibrated_phish_prob = prob_info["calibrated_phishing_probability"]
    uncalibrated_rf_phish_prob = prob_info["uncalibrated_rf_phishing_probability"]
    predicted_label = prob_info["predicted_label"]

    risk_analysis = analyze_url_risk_factors(
        url, url_features, calibrated_phish_prob, uncalibrated_rf_phish_prob
    )
    policy = agent.evaluate_policy(calibrated_phish_prob)

    importances = bundle.get("feature_importances")
    if importances is None or len(importances) != len(feature_names):
        importances = getattr(
            bundle["model"], "feature_importances_", np.zeros(len(feature_names))
        )

    return {
        "url": url,
        "anatomy": inspect_url_anatomy(url),
        "features": url_features,
        "feature_importances": [float(x) for x in importances],
        "model_metadata": bundle.get("model_metadata", {}),
        "predicted_label": predicted_label,
        "calibrated_probabilities": prob_info["calibrated_probabilities"],
        "uncalibrated_rf_probabilities": prob_info["uncalibrated_rf_probabilities"],
        "calibrated_phishing_probability": calibrated_phish_prob,
        "uncalibrated_rf_phishing_probability": uncalibrated_rf_phish_prob,
        "risk_factors": risk_analysis["risk_factors"],
        "protective_factors": risk_analysis["protective_factors"],
        "policy": policy,
    }


def decode_and_store(
    image_bytes: bytes,
    source_id: str,
    strict_scheme: bool,
    bundle: dict,
    agent: AdaptiveSecurityAgent,
) -> None:
    """Decode an image buffer and update session_state with the result or an actionable error."""
    temporary_file_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(image_bytes)
            temporary_file_path = Path(tmp.name)
        decoded_payloads = decode_qr_codes(temporary_file_path)
    except (QRDecodeError, ValueError, FileNotFoundError) as error:
        st.session_state.pop("last_analysis", None)
        st.error(f"Could not decode a QR code from this image: {error}")
        return
    finally:
        if temporary_file_path is not None:
            try:
                temporary_file_path.unlink(missing_ok=True)
            except OSError as cleanup_err:
                st.caption(f"Note: temporary file cleanup warning ({cleanup_err}).")

    if len(decoded_payloads) > 1:
        st.warning(
            f"{len(decoded_payloads)} distinct QR codes were detected in this image. "
            "Because a smartphone camera could scan any of them, select which decoded payload to inspect below:"
        )
        selected_index = st.radio(
            "Select decoded QR payload index to inspect:",
            options=list(range(len(decoded_payloads))),
            format_func=lambda idx: f"QR #{idx + 1}: {decoded_payloads[idx][:80]}",
            key=f"multi_qr_select_{source_id}",
        )
        decoded_text = decoded_payloads[int(selected_index)]
    else:
        decoded_text = decoded_payloads[0]

    if not is_valid_url(decoded_text, require_scheme=strict_scheme):
        st.session_state.pop("last_analysis", None)
        scheme_req_str = "HTTP/HTTPS " if strict_scheme else ""
        st.error(
            f"Unsupported QR content: The decoded payload is not a valid {scheme_req_str}URL."
        )
        st.markdown("**Decoded Raw Payload (rendered as inert plain text):**")
        st.code(decoded_text[:500], language=None)
        if not decoded_text.lower().startswith(("http://", "https://")) and is_valid_url(
            decoded_text, require_scheme=False
        ):
            st.info(
                "Tip: This QR code contains a bare domain without 'http://' or 'https://'. "
                "Uncheck 'Require explicit http:// or https:// scheme' in the left sidebar to analyze bare domains."
            )
        return

    analysis = run_url_analysis(bundle, decoded_text, agent)
    st.session_state["last_analysis"] = analysis
    st.session_state["analysis_key"] = (source_id, decoded_text, strict_scheme)
    st.session_state["feedback_submitted"] = False


# ---------------------------------------------------------------------------
# Sidebar: Policy Settings, Feedback-Driven Threshold Controls & Model Metrics
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🛡️ ARES Control Panel")
    strict_http_scheme = st.checkbox(
        "Require explicit http:// or https:// scheme",
        value=True,
        help="When enabled, bare domains without http:// or https:// are rejected unless toggled off.",
    )
    enable_demo_feedback = st.checkbox(
        "Enable local demo threshold updates",
        value=True,
        help=(
            "Allows user verdicts to adjust the local Review/Block thresholds in results/rl_policy_state.json. "
            "In a shared multi-user deployment, threshold updates should be restricted to authenticated analysts."
        ),
    )

    agent = AdaptiveSecurityAgent(
        feedback_csv_path=FEEDBACK_PATH, state_json_path=RL_STATE_PATH
    )
    if agent.diagnostics:
        with st.expander("⚠️ Policy State Diagnostics", expanded=False):
            for diag in agent.diagnostics:
                st.caption(f"- {diag}")

    st.divider()
    st.subheader("🎛️ Adaptive Threshold Policy")
    st.caption(
        "Feedback-driven threshold controller (adjusts decision boundaries; does not retrain the classifier)."
    )
    mcol1, mcol2 = st.columns(2)
    mcol1.metric(
        "Review At", f"{agent.review_threshold:.0%}", help="Medium-risk recommendation threshold"
    )
    mcol2.metric(
        "Block At", f"{agent.block_threshold:.0%}", help="High-risk recommendation threshold"
    )
    st.caption(
        f"Feedback episodes logged: **{agent.total_episodes}** | "
        f"Cumulative score: **{agent.cumulative_reward:+.1f}**"
    )

    if st.button("↺ Reset Thresholds to Default", use_container_width=True):
        agent.reset_policy()
        if "last_analysis" in st.session_state:
            st.session_state["last_analysis"]["policy"] = agent.evaluate_policy(
                st.session_state["last_analysis"]["calibrated_phishing_probability"]
            )
        st.toast("Thresholds reset to 35% (Review) and 70% (Block).")
        st.rerun()

    if BASELINE_PATH.exists():
        with st.expander("📈 Held-Out Validation Metrics"):
            try:
                base_data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))[
                    "baseline"
                ]["test"]
                st.caption(
                    "Evaluated on a domain-grouped split of the semi-synthetic dataset "
                    "(not an independent real-world deployment benchmark):"
                )
                st.text(
                    f"Split      : {base_data.get('split', 'GroupShuffleSplit by domain')}\n"
                    f"Test URLs  : {base_data.get('n_test', 0)} ({base_data.get('n_test_domains', 0)} domains)\n"
                    f"ROC-AUC    : {base_data.get('roc_auc', 0):.4f}\n"
                    f"PR-AUC     : {base_data.get('pr_auc', 0):.4f}\n"
                    f"Brier Loss : {base_data.get('brier', 0):.4f}\n"
                    f"Recall/TPR : {base_data.get('tpr', 0):.2%}\n"
                    f"False Pos  : {base_data.get('fpr', 0):.2%}"
                )
            except Exception as err:
                st.caption(f"Could not load baseline metrics: {err}")

# ---------------------------------------------------------------------------
# Main Header & 4-Mode Input Tabs
# ---------------------------------------------------------------------------
st.title("🛡️ ARES — QR Code Phishing (Quishing) Detector")
st.markdown(
    "Offline lexical and structural URL risk analyzer for QR codes. "
    "ARES decodes QR images without visiting or fetching the destination link, extracts **24 URL features**, "
    "evaluates an **isotonic-calibrated Random Forest classifier**, and recommends an endpoint policy action "
    "(`Allow`, `Review`, or `Block`)."
)
st.caption(
    "Disclaimer: ARES is an educational prototype that analyzes URL strings only; it does not inspect live webpage content, "
    "check external threat-intelligence feeds, or guarantee that a low-scoring website is safe."
)

try:
    bundle = get_cached_bundle(MODEL_PATH)
except (FileNotFoundError, ValueError) as error:
    st.error(f"Model bundle error: {error}")
    st.stop()

tab_samples, tab_upload, tab_camera, tab_url = st.tabs(
    [
        "🧪 1-Click Sample QR Gallery",
        "📁 Upload QR Image",
        "📸 Camera Snapshot",
        "🔗 Direct URL Analyzer & QR Encoder",
    ]
)

with tab_samples:
    st.markdown("##### Select a built-in test QR code to run an immediate offline analysis:")
    scol1, scol2 = st.columns([2, 1])
    with scol1:
        selected_sample_label = st.selectbox(
            "Choose a sample QR code:",
            options=list(SAMPLE_CATALOG.keys()),
        )
        sample_meta = SAMPLE_CATALOG[selected_sample_label]
        st.caption(f"Scenario: {sample_meta['hint']}")
        if sample_meta["needs_bare"] and strict_http_scheme:
            st.caption(
                "Note: This sample encodes a bare domain without http(s)://. "
                "Keep the checkbox below enabled or uncheck the sidebar setting to classify it."
            )
        override_bare = st.checkbox(
            "Allow bare domain for this sample",
            value=sample_meta["needs_bare"],
            key="sample_bare_override",
        )
        run_sample_btn = st.button(
            "🔍 Analyze Selected Sample QR",
            type="primary",
            use_container_width=True,
        )
    with scol2:
        sample_path = SAMPLE_IMAGES_DIR / sample_meta["file"]
        if sample_path.exists():
            st.image(str(sample_path), caption=sample_meta["file"], width=160)
        else:
            st.warning(f"Sample image not found at {sample_path.name}")

    if run_sample_btn and sample_path.exists():
        effective_strict = strict_http_scheme and not override_bare
        decode_and_store(
            sample_path.read_bytes(),
            source_id=f"sample:{sample_meta['file']}:{effective_strict}",
            strict_scheme=effective_strict,
            bundle=bundle,
            agent=agent,
        )

with tab_upload:
    uploaded_image = st.file_uploader(
        "Upload a QR-Code image (PNG, JPG, JPEG)",
        type=IMAGE_FILE_TYPES,
        accept_multiple_files=False,
        help=f"Maximum upload size: {MAX_IMAGE_SIZE_MB} MB",
    )
    if uploaded_image is not None:
        if uploaded_image.size > MAX_IMAGE_SIZE_MB * (1024**2):
            st.error(
                f"Uploaded image exceeds the {MAX_IMAGE_SIZE_MB} MB limit ({uploaded_image.size / (1024**2):.2f} MB)."
            )
        else:
            ucol1, ucol2 = st.columns([1, 2])
            with ucol1:
                st.image(uploaded_image, width=200)
            with ucol2:
                st.text(
                    f"Filename : {uploaded_image.name}\n"
                    f"Size     : {uploaded_image.size / 1024:.1f} KB"
                )
                if st.button(
                    "🔍 Analyze Uploaded QR Code",
                    type="primary",
                    use_container_width=True,
                ):
                    decode_and_store(
                        uploaded_image.getvalue(),
                        source_id=f"upload:{uploaded_image.file_id}:{strict_http_scheme}",
                        strict_scheme=strict_http_scheme,
                        bundle=bundle,
                        agent=agent,
                    )

with tab_camera:
    st.markdown(
        "##### Capture a single camera snapshot of a QR code for offline decoding:"
    )
    st.caption(
        "Note: Uses Streamlit's single-frame snapshot input (`st.camera_input`), not continuous video streaming."
    )
    camera_photo = st.camera_input("Take a snapshot of a QR code")
    if camera_photo is not None:
        decode_and_store(
            camera_photo.getvalue(),
            source_id=f"camera:{camera_photo.file_id}:{strict_http_scheme}",
            strict_scheme=strict_http_scheme,
            bundle=bundle,
            agent=agent,
        )

with tab_url:
    st.markdown(
        "##### Analyze a URL string directly and optionally encode it into a QR image:"
    )
    st.caption(
        "Warning: The QR encoder simply converts the text you enter into a QR matrix; "
        "it does not verify or make the destination URL safe."
    )
    ucol_input, ucol_qr = st.columns([2, 1])
    with ucol_input:
        custom_url = st.text_input(
            "Enter a URL to analyze:",
            value="https://secure-login.paypal.com.account-verify.xyz/signin?session=8942",
            placeholder="https://example.com/login",
        )
        analyze_url_btn = st.button(
            "🔍 Analyze URL",
            type="primary",
            use_container_width=True,
        )
    with ucol_qr:
        if custom_url.strip():
            try:
                qr_arr = generate_qr_image(custom_url.strip(), scale=6)
                ok, png_buf = cv2.imencode(".png", qr_arr)
                if ok:
                    png_bytes = png_buf.tobytes()
                    st.image(
                        png_bytes,
                        caption="Encoded QR Matrix (Unverified Destination)",
                        width=160,
                    )
                    st.download_button(
                        "⬇️ Download QR PNG",
                        data=png_bytes,
                        file_name="encoded_qr.png",
                        mime="image/png",
                        use_container_width=True,
                    )
                else:
                    st.warning("OpenCV failed to encode the QR matrix as PNG.")
            except (QRDecodeError, ValueError) as qr_err:
                st.warning(f"QR generation error: {qr_err}")

    if analyze_url_btn:
        candidate_url = custom_url.strip()
        if not is_valid_url(candidate_url, require_scheme=strict_http_scheme):
            st.session_state.pop("last_analysis", None)
            scheme_req_str = "HTTP/HTTPS " if strict_http_scheme else ""
            st.error(
                f"The entered string is not a valid {scheme_req_str}URL. Raw input:"
            )
            st.code(candidate_url[:500], language=None)
        else:
            analysis = run_url_analysis(bundle, candidate_url, agent)
            st.session_state["last_analysis"] = analysis
            st.session_state["analysis_key"] = (
                "direct_url",
                candidate_url,
                strict_http_scheme,
            )
            st.session_state["feedback_submitted"] = False

# ---------------------------------------------------------------------------
# Analysis Results Dashboard
# ---------------------------------------------------------------------------
if "last_analysis" in st.session_state:
    result = st.session_state["last_analysis"]
    url = result["url"]
    anatomy = result["anatomy"]
    policy = agent.evaluate_policy(result["calibrated_phishing_probability"])
    result["policy"] = policy
    predicted_label = result["predicted_label"]
    cal_prob = result["calibrated_phishing_probability"]
    uncal_prob = result["uncalibrated_rf_phishing_probability"]
    model_meta = result.get("model_metadata", {})

    st.divider()
    st.subheader("🛡️ Risk Assessment & Recommended Policy Action")

    if policy["action"] == "Block":
        st.error(
            f"🚨 HIGH ESTIMATED RISK — BLOCK RECOMMENDED | "
            f"Isotonic-Calibrated Phishing Probability: {cal_prob:.1%} "
            f"(Uncalibrated Base RF Vote: {uncal_prob:.1%}) exceeds the {policy['block_threshold']:.0%} Block threshold. "
            f"Do not visit this link."
        )
    elif policy["action"] == "Review":
        st.warning(
            f"⚠️ MEDIUM ESTIMATED RISK — MANUAL REVIEW RECOMMENDED | "
            f"Isotonic-Calibrated Phishing Probability: {cal_prob:.1%} "
            f"(Uncalibrated Base RF Vote: {uncal_prob:.1%}) crosses the {policy['review_threshold']:.0%} Review threshold. "
            f"Verify the domain owner independently before proceeding."
        )
    else:
        st.success(
            f"✅ LOWER ESTIMATED RISK — ALLOW RECOMMENDED | "
            f"Isotonic-Calibrated Phishing Probability: {cal_prob:.1%} "
            f"(Uncalibrated Base RF Vote: {uncal_prob:.1%}) is below the {policy['review_threshold']:.0%} Review threshold. "
            f"Note: A low lexical risk score does not guarantee that a website is safe."
        )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Model Classification (0.50 cutoff)",
        predicted_label,
        help="Binary class at a fixed 50% calibrated probability threshold",
    )
    k2.metric(
        "Calibrated Phishing Prob.",
        f"{cal_prob:.1%}",
        help=f"5-fold Isotonic-Calibrated probability (Uncalibrated base RF mean: {uncal_prob:.1%})",
    )
    k3.metric("Policy Risk Tier", policy["level"])
    k4.metric(
        "Recommended Action",
        f"{policy['action']} (Advisory)",
        help="Advisory policy recommendation based on adaptive thresholds",
    )

    st.markdown("**Decoded Destination URL (rendered as inert plain text for safety):**")
    st.code(url, language=None)

    # URL Anatomy Breakdown rendered safely via st.code / dataframe without raw markdown interpolation
    st.markdown("#### 🧬 URL Structure & Host Anatomy")
    acol1, acol2, acol3, acol4, acol5 = st.columns(5)
    with acol1:
        st.caption("Scheme")
        st.code(anatomy["scheme"], language=None)
    with acol2:
        st.caption("Subdomain Prefix")
        st.code(anatomy["subdomain"], language=None)
    with acol3:
        st.caption("Registrable Domain")
        st.code(anatomy["registrable_domain"], language=None)
    with acol4:
        st.caption("Path Component")
        st.code(anatomy["path"], language=None)
    with acol5:
        st.caption("Query Component")
        st.code(anatomy["query"], language=None)

    # Risk & Contextual Signals
    rcol, pcol = st.columns(2)
    with rcol:
        st.markdown("#### 🚨 Triggered Risk Signals")
        if result["risk_factors"]:
            for factor in result["risk_factors"]:
                st.text(f"• [RISK] {factor}")
        else:
            st.caption("No lexical or host-level risk heuristics triggered.")
    with pcol:
        st.markdown("#### ℹ️ Contextual & Transport Observations")
        if result["protective_factors"]:
            for factor in result["protective_factors"]:
                st.text(f"• [INFO] {factor}")
        else:
            st.caption("No additional structural observations.")

    # Calibrated Class Probability Bars
    st.markdown("#### 📊 Isotonic-Calibrated Class Probabilities")
    for name, probability in sorted(
        result["calibrated_probabilities"].items(),
        key=lambda kv: kv[1],
        reverse=True,
    ):
        st.progress(
            min(max(probability, 0.0), 1.0),
            text=(
                f"{name} (Calibrated): {probability:.2%} | "
                f"Uncalibrated Base RF: {result['uncalibrated_rf_probabilities'].get(name, 0.0):.2%}"
            ),
        )

    # 24-Feature Inspector Table
    with st.expander(
        "🔬 View All 24 Extracted URL Features & Reference RF Gini Importance",
        expanded=False,
    ):
        st.caption(
            "Feature importance values (%) come from the reference uncalibrated RandomForestClassifier "
            "fitted on the training split alongside the 5-fold CalibratedClassifierCV ensemble."
        )
        feature_rows = []
        for fname, imp in zip(feature_names, result["feature_importances"]):
            val = result["features"][fname]
            cat, desc = FEATURE_METADATA.get(fname, ("Other", ""))
            feature_rows.append(
                {
                    "Feature": fname,
                    "Category": cat,
                    "Extracted Value": round(val, 4) if isinstance(val, float) else val,
                    "Reference RF Importance (%)": round(imp * 100, 2),
                    "What It Measures": desc,
                }
            )
        feat_df = pd.DataFrame(feature_rows).sort_values(
            by="Reference RF Importance (%)", ascending=False
        )
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # Feedback & JSON Export Section
    st.divider()
    f_left, f_right = st.columns([2, 1])

    with f_left:
        st.markdown("#### 🔄 Feedback-Driven Threshold Adaptation")
        st.caption(
            "Submitting a verdict logs a query-redacted record to `results/feedback.csv` and adjusts "
            "the `Review` and `Block` probability thresholds. It does not retrain the classifier."
        )
        b1, b2, b3 = st.columns(3)
        already_done = (
            st.session_state.get("feedback_submitted", False)
            or not enable_demo_feedback
        )
        fp_disabled = already_done or (policy["action"] == "Allow")
        fn_disabled = already_done or (policy["action"] == "Block")

        verdict_choice = None
        if b1.button(
            "✅ Verdict Correct",
            use_container_width=True,
            disabled=already_done,
            help="Confirm this recommendation (+1.0 score, gently stabilizes thresholds toward defaults)",
        ):
            verdict_choice = "correct"
        if b2.button(
            "⚠️ False Positive (Benign)",
            use_container_width=True,
            disabled=fp_disabled,
            help=(
                "Disabled when recommended action is already Allow"
                if policy["action"] == "Allow"
                else "Report a false alarm (-1.5 score -> raises Review/Block thresholds)"
            ),
        ):
            verdict_choice = "false_positive"
        if b3.button(
            "🚨 False Negative (Phishing)",
            use_container_width=True,
            disabled=fn_disabled,
            help=(
                "Disabled when recommended action is already Block"
                if policy["action"] == "Block"
                else "Report a missed phishing URL (-3.0 score -> lowers Review/Block thresholds)"
            ),
        ):
            verdict_choice = "false_negative"

        if verdict_choice is not None:
            try:
                update_info = agent.record_feedback(
                    url=url,
                    probability=cal_prob,
                    level=policy["level"],
                    action=policy["action"],
                    user_verdict=verdict_choice,
                )
                if update_info.get("duplicate"):
                    st.info(
                        "You already submitted this verdict for this destination; thresholds unchanged."
                    )
                else:
                    st.session_state["feedback_submitted"] = True
                    st.session_state["last_analysis"]["policy"] = update_info[
                        "updated_policy"
                    ]
                    st.success(
                        f"Recorded '{verdict_choice}' (score delta {update_info['reward']:+.1f}). "
                        f"Updated thresholds — Review: {update_info['updated_policy']['review_threshold']:.1%}, "
                        f"Block: {update_info['updated_policy']['block_threshold']:.1%}."
                    )
            except ValueError as err:
                st.warning(f"Feedback rejected: {err}")

    with f_right:
        st.markdown("#### 📥 Export Analysis")
        st.caption("Download a structured JSON report of this offline lexical analysis.")
        report_payload = {
            "report_schema_version": "2.1",
            "analysis_mode": "offline_lexical_and_structural_url_analysis",
            "url": url,
            "redacted_url": redact_url_for_storage(url),
            "model_classification_at_0_50": predicted_label,
            "calibrated_phishing_probability": round(cal_prob, 6),
            "uncalibrated_rf_phishing_probability": round(uncal_prob, 6),
            "calibrated_class_probabilities": result["calibrated_probabilities"],
            "uncalibrated_rf_class_probabilities": result[
                "uncalibrated_rf_probabilities"
            ],
            "recommended_policy_decision": policy,
            "url_anatomy": anatomy,
            "risk_factors": result["risk_factors"],
            "contextual_observations": result["protective_factors"],
            "extracted_features": result["features"],
            "model_metadata": model_meta,
            "disclaimer": (
                "Offline lexical URL estimate only. Does not inspect live page content or external "
                "reputation feeds and does not guarantee destination safety."
            ),
        }
        st.download_button(
            "📥 Download JSON Security Report",
            data=json.dumps(report_payload, indent=2),
            file_name="ares_scan_report.json",
            mime="application/json",
            use_container_width=True,
        )
        if st.button("🧹 Clear Current Result", use_container_width=True):
            st.session_state.pop("last_analysis", None)
            st.session_state.pop("feedback_submitted", None)
            st.rerun()

    if FEEDBACK_PATH.exists():
        with st.expander("🗂️ View Logged Feedback History (Query-Redacted)"):
            try:
                feedback_df = pd.read_csv(FEEDBACK_PATH)
                if "url" in feedback_df.columns:
                    feedback_df["url"] = (
                        feedback_df["url"]
                        .astype(str)
                        .apply(lambda u: redact_url_for_storage(u)[:120])
                    )
                st.dataframe(
                    feedback_df.tail(20), use_container_width=True, hide_index=True
                )
            except Exception as err:
                st.caption(f"Could not read feedback log: {err}")
