@echo off
cd /d "%~dp0"
title Prosper AI Dashboard
echo Starting the Prosper dashboard server...
echo.
echo When it is up, open http://127.0.0.1:8000 in your browser. Do not close this window!
echo (Press CTRL+C to stop the server)
echo.

rem Prefer the in-project virtualenv; fall back to Poetry, then to PATH.
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m uvicorn prosper.api.server:app --port 8000
) else (
    where poetry >nul 2>nul && (
        poetry run uvicorn prosper.api.server:app --port 8000
    ) || (
        python -m uvicorn prosper.api.server:app --port 8000
    )
)
pause
