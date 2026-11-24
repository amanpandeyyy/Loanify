@echo off
title CreditWise AI - Streamlit Dashboard
cd /d "%~dp0"
echo =======================================================
echo   Starting CreditWise AI Streamlit UI...
echo   Opening in your browser at: http://localhost:8501
echo =======================================================
echo.
"C:\Users\ap640\AppData\Local\Python\bin\python.exe" -m streamlit run streamlit_app.py
if %errorlevel% neq 0 (
    echo.
    echo Streamlit stopped with error code %errorlevel%.
    pause
)
