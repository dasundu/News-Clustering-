#!/bin/bash
echo "Creating virtual environment..."
python3 -m venv venv || { echo "Failed to create venv. Make sure Python 3 is installed."; exit 1; }

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt || { echo "Install failed."; exit 1; }

echo ""
echo "Setup complete!"
echo "Copy .env.example to .env and fill in your credentials."
echo ""
echo "To start the server:"
echo "  source venv/bin/activate"
echo "  python main.py --api"
