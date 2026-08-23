#!/usr/bin/env bash
echo "========================================================"
echo "         HILLOCK NEURO-SYMBOLIC MEMORY ENGINE          "
echo "========================================================"
echo "[1/3] Checking Python Environment..."

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "[2/3] Installing / Updating Dependencies..."
pip install -r requirements.txt --quiet

echo "[3/3] Checking SpaCy English Language Model..."
python3 -c "import spacy; spacy.load('en_core_web_sm')" >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Downloading SpaCy 'en_core_web_sm' model..."
    python3 -m spacy download en_core_web_sm
fi

echo "========================================================"
echo "Starting Hillock Engine..."
echo "========================================================"
python3 main.py