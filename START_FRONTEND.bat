@echo off
cd /d "%~dp0frontend"
if not exist node_modules (call npm install)
if errorlevel 1 (echo Install Node.js LTS and retry & pause & exit /b 1)
call npm run dev
pause
