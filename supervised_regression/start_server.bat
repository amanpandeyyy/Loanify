@echo off
title CreditWise AI - Loan Approval Server
cd /d "%~dp0app.py"
echo =======================================================
echo   Starting CreditWise AI FastAPI Server and Dashboard...
echo   Web UI will be available at: http://127.0.0.1:8000
echo   Swagger Docs at:             http://127.0.0.1:8000/docs
echo =======================================================
echo.
"C:\Users\ap640\AppData\Local\Python\bin\python.exe" app.py
if %errorlevel% neq 0 (
    echo.
    echo Server stopped with error code %errorlevel%.
    pause
)
