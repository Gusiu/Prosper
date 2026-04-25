@echo off
cd /d "%~dp0"
title Prosper AI Dashboard
echo Uruchamianie silnika sztucznej inteligencji Prosper...
echo.
echo Za chwile otworzy sie strona w przegladarce. Nie zamykaj tego okna!
echo (Aby wylaczyc serwer, wcisnij CTRL+C)
echo.
poetry run uvicorn prosper.api.server:app --port 8000
pause
