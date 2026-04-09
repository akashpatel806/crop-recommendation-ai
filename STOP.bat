@echo off
title Stop Crop AI Services
color 0C

echo.
echo  Stopping all Crop AI services...
echo.

:: Kill Flask (python app.py on port 5000)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5000"') do (
    taskkill /PID %%a /F >nul 2>&1
)

:: Kill Streamlit (port 8501)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8501"') do (
    taskkill /PID %%a /F >nul 2>&1
)

echo  [OK]  Flask API stopped  (port 5000)
echo  [OK]  Streamlit stopped  (port 8501)
echo.
echo  All services stopped. Press any key to close.
pause >nul
