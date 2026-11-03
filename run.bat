@echo off
title EchoScan AI Pro - Launcher
echo ========================================================
echo               EchoScan AI Pro Launcher
echo ========================================================
echo Starting FastAPI Backend (Port 8000)...
echo Starting Vite Frontend...
echo ========================================================

:: Change directory to script root
cd /d "%~dp0"

:: Start FastAPI Backend in a separate terminal window
start "EchoScan Backend (FastAPI)" cmd /k "cd /d "%~dp0backend" && venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"

:: Start Vite Frontend in a separate terminal window
start "EchoScan Frontend (Vite)" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo.
echo Both services are running in separate windows!
echo  - Backend API: http://localhost:8000
echo  - Frontend:    http://localhost:5173 (or assigned port)
echo.
pause
