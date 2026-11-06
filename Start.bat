@echo off
setlocal
cd /d "%~dp0"
title Plates

if not "%~1"=="" goto source

if exist "%~dp0portable\Plates\Plates.exe" (
  start "" "%~dp0portable\Plates\Plates.exe"
  exit /b 0
)

:source
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo First launch: creating a local Python environment...
  python -m venv "%~dp0.venv"
  if errorlevel 1 goto nopython
  "%~dp0.venv\Scripts\python.exe" -m pip install --upgrade pip
  "%~dp0.venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"
  if errorlevel 1 goto pipfail
)

if "%~1"=="" (
  "%~dp0.venv\Scripts\pythonw.exe" "%~dp0app.py"
  if errorlevel 1 (
    "%~dp0.venv\Scripts\python.exe" "%~dp0app.py"
    echo.
    pause
  )
) else (
  "%~dp0.venv\Scripts\python.exe" "%~dp0app.py" %*
)
exit /b 0

:nopython
echo Python 3.11 or newer was not found.
echo Install it from https://www.python.org/downloads/ and run this again.
pause
exit /b 1

:pipfail
echo Could not install the required libraries.
pause
exit /b 1
