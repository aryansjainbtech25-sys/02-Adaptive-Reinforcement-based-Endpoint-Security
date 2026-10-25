#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] python3 is not installed or not on PATH." >&2
    exit 1
fi

if [ ! -f "models/model.joblib" ]; then
    echo "[ERROR] Required model artifact models/model.joblib is missing." >&2
    exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
    echo "[ARES] Creating Python virtual environment (.venv)..."
    python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

if ! python -c "import streamlit, cv2, sklearn, pandas, PIL, joblib" >/dev/null 2>&1; then
    echo "[ARES] Installing dependencies from requirements.txt..."
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
fi

echo "[ARES] Verifying Python source compilation..."
python -m py_compile app.py adaptive_agent.py feature_extractor.py qr_utils.py test_pipeline.py train_model.py

echo "[ARES] Starting ARES Web Application..."
exec python -m streamlit run app.py
