@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  py -3 -m venv .venv
  if errorlevel 1 (echo Python 3 required! & pause & exit /b 1)
)
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
if errorlevel 1 (echo Dependency installation failed & pause & exit /b 1)
python -c "from backend.pipeline import bootstrap; bootstrap()"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
pause
