# F1 Season Analytics (Streamlit)

Interactive analytics app for Formula 1 seasons (2024–2025 ready) using the Jolpi mirror of the Ergast API.

## Features
- Results & official standings (drivers/constructors)
- Cumulative points (with top-10 annotations)
- Qualifying vs race analysis (per-driver averages, colored by team)
- Constructors pie (points share)
- Stacked bars: constructor points broken down by driver (team color + darker shade)

## Quickstart

```bash
# from the project root
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

# run
python -m streamlit run src/app.py
```

If your network rate-limits (HTTP 429) or blocks the API, the app caches all responses under `data/raw/jolpi`. You can prefill these JSONs manually if needed and the app will use the cache.
