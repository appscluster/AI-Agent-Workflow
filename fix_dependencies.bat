@echo off
echo Fixing dependencies for AI Agent Workflow...
echo.

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing specific versions of dependencies...
pip uninstall -y pydantic chromadb
pip install pydantic==1.10.8
pip install chromadb==0.4.13
pip install -r requirements.txt

echo.
echo Dependencies fixed! You can now run the application.
echo python main.py --document "path/to/document.pdf" --interactive
