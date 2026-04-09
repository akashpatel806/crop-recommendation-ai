@echo off
title Crop Recommendation AI - Launcher
color 0A

echo.
echo  =====================================================
echo   CROP RECOMMENDATION AI - ONE-CLICK LAUNCHER
echo  =====================================================
echo.

:: ── Check Python ──────────────────────────────────────
py --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found! Please install Python 3.10+
    pause
    exit /b 1
)

:: ── Go to script folder ───────────────────────────────
cd /d "%~dp0"

:: ── Check model exists, train if not ──────────────────
if not exist "crop_model.pkl" (
    echo  [INFO]  crop_model.pkl not found. Training model first...
    echo.
    if not exist "Crop_recommendation.csv" (
        echo  [INFO]  Downloading dataset...
        py download_dataset.py
        if errorlevel 1 (
            echo  [ERROR] Dataset download failed!
            pause
            exit /b 1
        )
    )
    py train_model.py
    if errorlevel 1 (
        echo  [ERROR] Model training failed!
        pause
        exit /b 1
    )
    echo.
    echo  [OK]    Model trained successfully!
) else (
    echo  [OK]    Model found: crop_model.pkl
)

:: ── Check .env exists ─────────────────────────────────
if not exist ".env" (
    echo  [WARN]  .env file not found! MongoDB connection may fail.
)

:: ── Start Flask API in new window ─────────────────────
echo.
echo  [START] Launching Flask API on http://localhost:5000 ...
start "Crop AI - Flask API" cmd /k "cd /d "%~dp0" && color 0B && echo  Flask API starting... && py app.py"

:: ── Wait 3 seconds for Flask to initialize ────────────
echo  [WAIT]  Waiting for Flask API to start...
ping -n 4 127.0.0.1 >nul

:: ── Start Streamlit Dashboard in new window ───────────
echo  [START] Launching Streamlit Dashboard on http://localhost:8501 ...
start "Crop AI - Dashboard" cmd /k "cd /d "%~dp0" && color 0D && echo  Streamlit Dashboard starting... && py -m streamlit run dashboard.py --server.headless false"

:: ── Wait 4 seconds for Streamlit to boot ──────────────
echo  [WAIT]  Waiting for Dashboard to start...
ping -n 5 127.0.0.1 >nul

:: ── Open browser ──────────────────────────────────────
echo  [OPEN]  Opening Dashboard in browser...
start "" "http://localhost:8501"

:: ── Done ──────────────────────────────────────────────
echo.
echo  =====================================================
echo   ALL SERVICES STARTED!
echo  =====================================================
echo.
echo   Dashboard  :  http://localhost:8501
echo   Flask API  :  http://localhost:5000
echo   Recommend  :  http://localhost:5000/recommend
echo   History    :  http://localhost:5000/history?n=20
echo.
echo   To STOP everything: close the two opened windows.
echo.
pause
