@echo off
echo Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo Failed to create venv. Make sure Python is installed.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Install failed.
    pause
    exit /b 1
)

echo.
echo Setup complete!
echo Copy .env.example to .env and fill in your credentials.
echo.
echo To start the server:
echo   venv\Scripts\activate
echo   python main.py --api
pause
