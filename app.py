from __future__ import annotations

import csv
import html
import re
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from feature_extractor import extract_url_features, feature_names, suspicious_words
from qr_utils import decode_qr_code

# --------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
FEEDBACK_PATH = BASE_DIR / "results" / "feedback.csv"

DEFAULT_LABELS = {0: "Benign", 1: "Phishing"}

URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc",
}

TONES = {
    "Low": "#1f9d6b",
    "Medium": "#c97a00",
    "High": "#c8323b",
}

ACTIONS = {
    "Allow": "Nothing suspicious found. Safe to open, but stay alert.",
    "Warn": "The model is not worried, but the link has warning signs. Check it before you open it.",
    "Review": "The model is unsure. Have a person look at this link before anyone opens it.",
    "Block": "Strong phishing signals. Do not open this link.",
}

ACTION_TONE = {"Allow": "Low", "Warn": "Medium", "Review": "Medium", "Block": "High"}

st.set_page_config(
    page_title="ARES | QR phishing check",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --ink: #14213d;
  --muted: #5b6b82;
  --line: #d6dee8;
  --mist: #f3f6f9;
  --paper: #ffffff;
  --low: #1f9d6b;
  --med: #c97a00;
  --high: #c8323b;
}

html, body, [class*="css"], .stMarkdown, .stTextInput, .stButton, .stTabs {
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
}
.stApp { background: var(--mist); color: var(--ink); }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1250px; }
h1, h2, h3, .ares-title { font-family: 'Bricolage Grotesque', 'IBM Plex Sans', sans-serif; color: var(--ink); }

/* ---- header ---- */
.ares-title { font-size: 3.1rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1; margin: 0; }
.ares-title span { color: var(--muted); font-weight: 500; font-size: 1.15rem; letter-spacing: 0; margin-left: .8rem; }
.ares-lede { font-size: 1.12rem; color: var(--muted); margin: .7rem 0 1.6rem 0; max-width: 46rem; }

/* ---- verdict ---- */
.verdict {
  background: var(--paper);
  border: 1px solid var(--line);
  border-top: 6px solid var(--tone);
  border-radius: 6px;
  padding: 1.5rem 1.7rem 1.3rem 1.7rem;
  margin-bottom: 1rem;
}
.verdict-top { display: flex; justify-content: space-between; align-items: center; gap: 1rem; }
.v-prob { font-family: 'Bricolage Grotesque', sans-serif; font-size: 4rem; font-weight: 800; line-height: 1; color: var(--tone); letter-spacing: -0.03em; }
.v-sub { color: var(--muted); font-size: .98rem; margin-top: .35rem; }
.stamp {
  font-family: 'Bricolage Grotesque', sans-serif;
  font-weight: 800; font-size: 2rem; letter-spacing: .06em;
  color: var(--tone);
  border: 4px double var(--tone);
  border-radius: 8px;
  padding: .35rem 1.1rem;
  transform: rotate(-5deg);
  opacity: .92;
  white-space: nowrap;
}
.v-msg { margin-top: 1.1rem; font-size: 1.02rem; line-height: 1.5; }

/* ---- risk ruler ---- */
.ruler { position: relative; margin-top: 1.5rem; }
.ruler-track { display: flex; height: 14px; border-radius: 7px; overflow: hidden; }
.ruler-track div { height: 100%; }
.zone-low  { background: color-mix(in srgb, var(--low) 35%, white); }
.zone-mid  { background: color-mix(in srgb, var(--med) 40%, white); }
.zone-high { background: color-mix(in srgb, var(--high) 40%, white); }
.ruler-pin {
  position: absolute; top: -6px; width: 4px; height: 26px;
  background: var(--ink); border-radius: 2px; transform: translateX(-2px);
}
.ruler-labels { display: flex; font-size: .8rem; color: var(--muted); margin-top: .4rem; }
.ruler-labels div { text-align: center; overflow: hidden; white-space: nowrap; }

/* ---- chips ---- */
.chip {
  display: inline-block; margin: 0 .5rem .55rem 0; padding: .3rem .75rem;
  border-radius: 999px; font-size: .92rem; border: 1px solid;
}
.chip.high { color: #8f1d25; background: #fbe9ea; border-color: #efb9bd; }
.chip.medium { color: #8a5200; background: #fdf1de; border-color: #f0d09a; }
.chip.low { color: #29507a; background: #e8f0f9; border-color: #bcd2ea; }
.chip.ok { color: #14694a; background: #e4f5ed; border-color: #a9dcc5; }

/* ---- empty state ---- */
.empty {
  border: 2px dashed var(--line); border-radius: 6px; padding: 3rem 2rem;
  text-align: left; color: var(--muted); background: rgba(255,255,255,.5);
}
.empty b { color: var(--ink); font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.35rem; display: block; margin-bottom: .4rem; }

/* ---- misc widgets ---- */
section[data-testid="stSidebar"] { background: var(--paper); border-right: 1px solid var(--line); }
div[data-testid="stMetric"] { background: var(--paper); border: 1px solid var(--line); border-radius: 6px; padding: .8rem 1rem; }
.stButton > button { border-radius: 6px; font-weight: 600; }
.stTabs [data-baseweb="tab"] { font-weight: 600; }
code { font-family: 'IBM Plex Mono', monospace !important; }
.small-note { color: var(--muted); font-size: .88rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def html_block(markup: str) -> str:
    """Strip indentation so Markdown does not treat the HTML as a code block."""
    return "\n".join(line.strip() for line in markup.strip().splitlines() if line.strip())


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Load the saved bundle. Only load model files you trust: joblib can run code."""
    if not MODEL_PATH.exists():
        return None, f"Model file not found at `{MODEL_PATH.relative_to(BASE_DIR)}`."

    bundle = joblib.load(MODEL_PATH)

    if isinstance(bundle, dict):
        model = bundle["model"]
        saved_features = bundle.get("features") or bundle.get("feature_names") or feature_names
        labels = bundle.get("label_mapping", DEFAULT_LABELS)
    else:
        model = bundle
        saved_features = feature_names
        labels = DEFAULT_LABELS

    if list(saved_features) != list(feature_names):
        return None, (
            "The model's feature list does not match `feature_extractor.py`.\n\n"
            f"Model: {list(saved_features)}\n\nCode: {list(feature_names)}"
        )

    return {"model": model, "labels": labels}, None


# --------------------------------------------------------------------------
# QR decoding
# --------------------------------------------------------------------------
def decode_image_bytes(data: bytes) -> str: 
    """
    Decode a QR code from uploaded image bytes.

    1. Try OpenCV directly on the image (works for normal photos/screenshots).
    2. Fall back to qr_utils.decode_qr_code (padding + upscaling), which is
       what makes the tiny dataset QR images readable.
    """
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("That file could not be read as an image. Use PNG, JPG or BMP.")

    text, _, _ = cv2.QRCodeDetector().detectAndDecode(image)
    if text.strip():
        return text.strip()

    # qr_utils upscales 4-10x, so shrink very large images first to keep it fast.
    longest_side = max(image.shape)
    if longest_side > 400:
        factor = 400 / longest_side
        image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "upload.png"
        cv2.imwrite(str(path), image)
        return decode_qr_code(str(path))


# --------------------------------------------------------------------------
# Scoring, signals and policy
# --------------------------------------------------------------------------
def risk_level(probability: float, low: float, high: float) -> str:
    if probability >= high:
        return "High"
    if probability >= low:
        return "Medium"
    return "Low"


def find_signals(url: str, features: dict) -> list[tuple[str, str]]:
    """Plain-language warning signs. These explain the link; they are not the model's inputs."""
    signals: list[tuple[str, str]] = []
    lowered = url.lower()
    host = (urlsplit(url).hostname or "").lower()

    if features["has_ip"]:
        signals.append(("high", "The address is a raw IP number, not a domain name"))
    if "@" in url:
        signals.append(("high", "Contains an @ sign, which can hide the real destination"))
    if not lowered.startswith("https://"):
        signals.append(("medium", "Not encrypted: the link uses HTTP, not HTTPS"))
    if host in URL_SHORTENERS:
        signals.append(("medium", f"{host} is a link shortener, so the real destination is hidden"))
    if host.startswith("xn--") or ".xn--" in host:
        signals.append(("medium", "Uses punycode, which can imitate look-alike domain names"))

    found_words = [w for w in suspicious_words if re.search(rf"\b{w}\b", lowered)]
    if found_words:
        signals.append(("medium", "Contains words common in phishing: " + ", ".join(found_words)))

    if host.count(".") >= 4:
        signals.append(("medium", f"Many subdomain levels ({host.count('.')} dots in the host)"))
    if len(url) > 100:
        signals.append(("low", f"Unusually long link ({len(url)} characters)"))
    if features["entropy"] > 4.8:
        signals.append(("low", "The link looks random (high character entropy)"))

    return signals


def decide_action(probability: float, n_signals: int, low: float, high: float) -> str:
    """
    Fixed-rule response policy. This is the baseline the RL agent will be compared against.

    State the agent will see: probability, n_signals.   Actions: Allow, Warn, Review, Block.
    """
    if probability >= high:
        return "Block"
    if probability >= low:
        return "Review"
    if n_signals >= 1:
        return "Warn"
    return "Allow"


def analyse(url: str, source: str, low: float, high: float) -> dict:
    """Run the full pipeline on one URL and return everything the UI needs."""
    bundle = st.session_state["bundle"]
    model = bundle["model"]

    features = extract_url_features(url)
    row = pd.DataFrame([[features[name] for name in feature_names]], columns=feature_names)

    classes = list(model.classes_)
    phishing_index = classes.index(1) if 1 in classes else len(classes) - 1
    probability = float(model.predict_proba(row)[0][phishing_index])
    verdict = bundle["labels"].get(int(model.predict(row)[0]), "Unknown")

    signals = find_signals(url, features)
    action = decide_action(probability, len(signals), low, high)

    return {
        "time": datetime.now().strftime("%H:%M:%S"),
        "source": source,
        "url": url,
        "features": features,
        "probability": probability,
        "verdict": verdict,
        "level": risk_level(probability, low, high),
        "action": action,
        "signals": signals,
        "low": low,
        "high": high,
        "feedback": None,
    }


def run_scan(url: str, source: str, low: float, high: float) -> None:
    result = analyse(url, source, low, high)
    st.session_state["current"] = result
    st.session_state["history"].append(result)


def handle_decoded_text(text: str, source: str, low: float, high: float) -> None:
    """Accept only http(s) links; anything else is shown but never scored."""
    if text.lower().startswith(("http://", "https://")):
        st.session_state["notice"] = None
        run_scan(text, source, low, high)
    else:
        st.session_state["current"] = None
        st.session_state["notice"] = text


def save_feedback(verdict: str) -> None:
    """Log the person's judgement. This file becomes training data for the RL agent."""
    current = st.session_state.get("current")
    if not current or current["feedback"]:
        return
    try:
        FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        is_new = not FEEDBACK_PATH.exists()
        with open(FEEDBACK_PATH, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if is_new:
                writer.writerow(["time", "url", "probability", "level", "action", "user_verdict"])
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"), current["url"],
                round(current["probability"], 4), current["level"], current["action"], verdict,
            ])
        current["feedback"] = verdict
    except OSError as error:
        st.session_state["feedback_error"] = str(error)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def render_verdict(result: dict) -> None:
    tone = TONES[ACTION_TONE[result["action"]]]
    p, low, high = result["probability"], result["low"], result["high"]

    st.markdown(
        html_block(f"""
        <div class="verdict" style="--tone:{tone}">
          <div class="verdict-top">
            <div>
              <div class="v-prob">{p:.0%}</div>
              <div class="v-sub">chance this link is phishing</div>
            </div>
            <div class="stamp">{result["action"].upper()}</div>
          </div>
          <div class="ruler">
            <div class="ruler-track">
              <div class="zone-low" style="width:{low * 100:.1f}%"></div>
              <div class="zone-mid" style="width:{(high - low) * 100:.1f}%"></div>
              <div class="zone-high" style="width:{(1 - high) * 100:.1f}%"></div>
            </div>
            <div class="ruler-pin" style="left:{p * 100:.1f}%"></div>
            <div class="ruler-labels">
              <div style="width:{low * 100:.1f}%">Low risk</div>
              <div style="width:{(high - low) * 100:.1f}%">Medium</div>
              <div style="width:{(1 - high) * 100:.1f}%">High risk</div>
            </div>
          </div>
          <div class="v-msg">{html.escape(ACTIONS[result["action"]])}</div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    st.caption("Link found in the QR code (shown as plain text on purpose, do not open it):")
    st.code(result["url"], language=None)


def render_signals(result: dict) -> None:
    signals = result["signals"]
    if not signals:
        st.markdown('<span class="chip ok">No warning signs found in the link itself</span>', unsafe_allow_html=True)
    else:
        chips = "".join(f'<span class="chip {level}">{html.escape(text)}</span>' for level, text in signals)
        st.markdown(chips, unsafe_allow_html=True)
    st.markdown(
        '<p class="small-note">These are readable warning signs for people. '
        "The model works from the ten numbers on the next tab.</p>",
        unsafe_allow_html=True,
    )


def render_features(result: dict) -> None:
    frame = pd.DataFrame({
        "Feature": list(result["features"].keys()),
        "Value": [round(v, 3) if isinstance(v, float) else v for v in result["features"].values()],
    })
    st.dataframe(frame, hide_index=True)


def render_anatomy(result: dict) -> None:
    parts = urlsplit(result["url"])
    frame = pd.DataFrame({
        "Part": ["Scheme", "Host", "Path", "Query", "Length"],
        "Value": [
            parts.scheme or "-",
            parts.hostname or "-",
            parts.path or "-",
            parts.query or "-",
            f"{len(result['url'])} characters",
        ],
    })
    st.dataframe(frame, hide_index=True)


def render_feedback(result: dict) -> None:
    st.markdown("**Was this verdict right?**")
    if result["feedback"]:
        st.success(f"Thanks. You marked it \"{result['feedback']}\". Saved to results/feedback.csv.")
        return
    left, right, _ = st.columns([1, 1, 2])
    left.button("Yes, correct", key=f"ok_{result['time']}_{len(st.session_state['history'])}",
                on_click=save_feedback, args=("correct",))
    right.button("No, wrong", key=f"bad_{result['time']}_{len(st.session_state['history'])}",
                 on_click=save_feedback, args=("wrong",))
    if st.session_state.get("feedback_error"):
        st.warning(f"Could not save feedback: {st.session_state['feedback_error']}")


def render_empty() -> None:
    st.markdown(
        html_block("""
        <div class="empty">
          <b>No link checked yet</b>
          Upload a QR code image or paste a link on the left. The verdict, the risk
          level and the reasons behind it will show up here.
        </div>
        """),
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
st.session_state.setdefault("history", [])
st.session_state.setdefault("current", None)
st.session_state.setdefault("notice", None)

bundle, load_error = load_model()
if load_error:
    st.error(load_error)
    st.info("Place the trained model at `models/model.joblib` and reload this page.")
    st.stop()
st.session_state["bundle"] = bundle

# ---- sidebar ----
with st.sidebar:
    st.markdown("### Risk thresholds")
    low, high = st.slider(
        "Where each risk level starts",
        min_value=0.05, max_value=0.95, value=(0.30, 0.70), step=0.05,
        help="Below the left handle is Low risk, between the handles is Medium, above the right handle is High.",
    )
    st.caption(
        "Block above the right handle. Review between the handles. "
        "Below the left handle: Warn if the link has warning signs, otherwise Allow."
    )
    st.divider()
    st.markdown("### Model")
    st.markdown(f"**{type(bundle['model']).__name__}** on {len(feature_names)} URL features")
    st.divider()
    st.markdown("### Read this first")
    st.caption(
        "ARES is a student prototype. The training data has a known formatting "
        "shortcut, so scores can look better than they are. Do not use it as the "
        "only check before opening a link."
    )

# ---- header ----
st.markdown(
    '<h1 class="ares-title">ARES<span>Adaptive Reinforcement-based Endpoint Security</span></h1>'
    '<p class="ares-lede">Check where a QR code really leads before you open it. '
    "ARES reads the code, scores the link for phishing risk and recommends what to do.</p>",
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([5, 6], gap="large")

# ---- input column ----
with left_col:
    scan_tab, url_tab = st.tabs(["Scan a QR image", "Paste a link"])

    with scan_tab:
        upload = st.file_uploader("QR code image", type=["png", "jpg", "jpeg", "bmp"])
        image_bytes, source = None, None
        if upload is not None:
            image_bytes, source = upload.getvalue(), f"upload: {upload.name}"

        if image_bytes:
            st.image(image_bytes, width=220, caption=source)
            if st.button("Check this QR code", type="primary"):
                try:
                    handle_decoded_text(decode_image_bytes(image_bytes), source, low, high)
                except (ValueError, FileNotFoundError) as error:
                    st.session_state["current"] = None
                    st.error(f"Could not read a QR code from this image. {error}")

    with url_tab:
        typed = st.text_input("Link", placeholder="https://example.com/login")
        if st.button("Check this link", type="primary"):
            typed = typed.strip()
            if typed.lower().startswith(("http://", "https://")):
                st.session_state["notice"] = None
                run_scan(typed, "typed link", low, high)
            else:
                st.error("Start the link with http:// or https://")

# ---- result column ----
with right_col:
    current = st.session_state["current"]
    notice = st.session_state["notice"]

    if notice:
        st.warning("This QR code does not contain a web link, so ARES did not score it.")
        st.caption("What the QR code contains:")
        st.code(notice, language=None)
        st.caption("QR codes can also carry Wi-Fi logins, phone numbers or payment requests. Treat unknown ones with care.")
    elif current:
        render_verdict(current)
        m1, m2, m3 = st.columns(3)
        m1.metric("Risk level", current["level"])
        m2.metric("Model verdict", current["verdict"])
        m3.metric("Warning signs", len(current["signals"]))

        signals_tab, features_tab, anatomy_tab = st.tabs(["Warning signs", "Model inputs", "Link anatomy"])
        with signals_tab:
            render_signals(current)
        with features_tab:
            render_features(current)
        with anatomy_tab:
            render_anatomy(current)

        st.divider()
        render_feedback(current)
    else:
        render_empty()

# ---- footer sections ----
st.divider()
log_col, importance_col = st.columns(2, gap="large")

with log_col:
    with st.expander(f"Scans this session ({len(st.session_state['history'])})"):
        if st.session_state["history"]:
            log = pd.DataFrame([
                {
                    "Time": item["time"],
                    "Source": item["source"],
                    "Link": item["url"][:60] + ("..." if len(item["url"]) > 60 else ""),
                    "Phishing %": round(item["probability"] * 100, 1),
                    "Risk": item["level"],
                    "Action": item["action"],
                    "Your verdict": item["feedback"] or "-",
                }
                for item in reversed(st.session_state["history"])
            ])
            st.dataframe(log, hide_index=True)
        else:
            st.caption("Nothing scanned yet.")

with importance_col:
    with st.expander("What the model relies on"):
        importances = getattr(bundle["model"], "feature_importances_", None)
        if importances is not None:
            chart = pd.Series(importances, index=feature_names).sort_values()
            st.bar_chart(chart, horizontal=True)
            st.caption(
                "If one feature carries most of the weight, the model may be using a shortcut "
                "in the training data instead of real phishing patterns."
            )
        else:
            st.caption("This model type does not report feature importances.")