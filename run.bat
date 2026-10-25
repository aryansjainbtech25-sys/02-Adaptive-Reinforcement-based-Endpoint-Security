@echo off
setlocal EnableExtensions
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH. Install Python 3.10+ from https://www.python.org/ and ensure "Add Python to PATH" is checked.
    exit /b 1
)

if not exist "models\model.joblib" (
    echo [ERROR] Required model artifact "models\model.joblib" is missing.
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [ARES] Creating Python virtual environment in .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    exit /b 1
)

python -c "import streamlit, cv2, sklearn, pandas, PIL, joblib" >nul 2>&1
if errorlevel 1 (
    echo [ARES] Installing required dependencies from requirements.txt ...
    python -m pip install --upgrade pip
    if errorlevel 1 (
        echo [ERROR] Failed to upgrade pip.
        exit /b 1
    )
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed. If scikit-learn==1.9.1 wheel is unavailable for your Python version, install scikit-learn>=1.5.0 and run: python train_model.py
        exit /b 1
    )
)

echo [ARES] Verifying source compilation and model artifact ...
python -m py_compile app.py adaptive_agent.py feature_extractor.py qr_utils.py test_pipeline.py train_model.py
if errorlevel 1 (
    echo [ERROR] Python syntax verification failed.
    exit /b 1
)

echo [ARES] Launching Streamlit application ...
python -m streamlit run app.py
exit /b %ERRORLEVEL%
