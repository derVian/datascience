@echo off
REM Auto-create and run datascience.py using the project's .venv
setlocal enabledelayedexpansion
set REPO_DIR=%~dp0
set VENV_DIR=%REPO_DIR%.venv
set VENV_PY=%VENV_DIR%\Scripts\python.exe

if not exist "%VENV_DIR%" (
  echo Virtual environment not found. Creating at "%VENV_DIR%"...
  python -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo Failed to create virtual environment. Ensure Python is on PATH.
    exit /b 1
  )
  echo Installing requirements...
  "%VENV_PY%" -m pip install --upgrade pip
  if exist "%REPO_DIR%requirements.txt" (
    "%VENV_PY%" -m pip install -r "%REPO_DIR%requirements.txt"
  )
)

if exist "%VENV_PY%" (
  "%VENV_PY%" "%REPO_DIR%datascience.py" %*
) else (
  echo Using system Python to run the script.
  python "%REPO_DIR%datascience.py" %*
)

endlocal
