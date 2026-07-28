@echo off
cd /d "%~dp0"
title Prosper AI Dashboard
echo Uruchamianie silnika sztucznej inteligencji Prosper...
echo.
echo Za chwile otworzy sie strona w przegladarce. Nie zamykaj tego okna!
echo (Aby wylaczyc serwer, wcisnij CTRL+C)
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
