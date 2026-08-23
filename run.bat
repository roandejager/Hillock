@echo off
echo ========================================================
echo          HILLOCK NEURO-SYMBOLIC MEMORY ENGINE          
echo ========================================================
echo [1/3] Checking Python Environment...

IF NOT EXIST .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate

echo [2/3] Installing / Updating Dependencies...
pip install -r requirements.txt --quiet

echo [3/3] Checking SpaCy English Language Model...
python -c "import spacy; spacy.load('en_core_web_sm')" >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Downloading SpaCy 'en_core_web_sm' model...
    python -m spacy download en_core_web_sm
)

echo ========================================================
echo Starting Hillock Engine...
echo ========================================================
python main.py
pause