@echo off
REM Installation script for Windows

REM Check Python version
python --version | findstr /r "3\.1[0-9]\." > nul
if errorlevel 1 (
    echo Error: Python 3.10 or higher is required
    echo Current Python version:
    python --version
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Run the Python installation script
echo Running installation script...
python install.py --with-samples

echo Installation complete. Activate the virtual environment with:
echo venv\Scripts\activate
