# Environmental Knowledge Assistant — Biodiversity & Soil Health

A knowledge-driven conversational system that gives evidence-backed, multi-metric
recommendations to improve biodiversity and soil health, combining a RAG layer
(ChromaDB + sentence embeddings) with a hardcoded multi-metric reasoning engine.

## Quickstart (run it on localhost)

Requires Python 3.10+. From this folder:

```bash
# 1. Install dependencies (rasterio and spacy are optional — see notes below)
pip install chromadb pandas pyarrow scikit-learn fastapi uvicorn rasterio spacy
pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.7.1/en_core_web_md-3.7.1-py3-none-any.whl"

# 2. Build the vector store (once, or whenever you edit data/knowledge_base.csv)
python3 ingest_data.py --reset

# 3. Start the app
python3 main.py --serve
```
Then open **http://localhost:8000** in your browser — that's the chat UI.
Leave the terminal running; press `Ctrl+C` there to stop the server.

**Optional dependencies, safe to skip:**
- `rasterio` — only needed for the India geo-coordinate lookup
  (`bhi_lookup.py`). Without it, that feature just no-ops silently.
- `spacy` + the model wheel — gives real pretrained semantic embeddings.
  Without it, retrieval automatically falls back to TF-IDF (still works,
  just matches on shared words rather than meaning).

**If port 8000 is already in use:** open `main.py`, find `uvicorn.run(app,
host="0.0.0.0", port=8000)` near the bottom, and change `8000` to any free
port (e.g. `8001`), then reload the matching URL.

**Prefer a terminal chat instead of the browser UI?** Run `python3 main.py`
(no `--serve`) for the old REPL.


## Setup

```bash
pip install chromadb pandas pyarrow scikit-learn fastapi uvicorn rasterio spacy
pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.7.1/en_core_web_md-3.7.1-py3-none-any.whl"
python3 ingest_data.py --reset        # builds ./env_knowledge_db from data/knowledge_base.csv
python3 main.py --serve               # UI + API on http://localhost:8000
```
Open **http://localhost:8000** in a browser for the chat UI (`webapp/index.html`,
served by the same FastAPI process — no separate frontend server needed).
`python3 main.py` (no `--serve`) runs the old terminal-only REPL instead, if
you prefer that.




