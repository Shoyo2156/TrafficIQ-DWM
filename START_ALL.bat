@echo off
cd /d "%~dp0"
start "TrafficIQ Backend" cmd /k "call START_BACKEND.bat"
start "TrafficIQ Frontend" cmd /k "call START_FRONTEND.bat"
echo Backend http://127.0.0.1:8000/docs
 echo Frontend http://127.0.0.1:5173
pause
