@echo off
setlocal
cd /d "%~dp0"
title ArthroKNEE
echo ================================================
echo   ArthroKNEE - starting (first run installs packages)
echo ================================================
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install it from python.org and tick "Add Python to PATH".
  pause
  exit /b 1
)
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
  mkdir "%USERPROFILE%\.streamlit" 2>nul
  (echo [general]& echo email = "") > "%USERPROFILE%\.streamlit\credentials.toml"
)
python -m pip install -r requirements.txt
python -m streamlit run arthroknee_app.py
pause
