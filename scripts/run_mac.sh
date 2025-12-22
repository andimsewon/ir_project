#!/usr/bin/env bash

# Simple macOS runner for the SAP web UI
# - Creates/uses .venv
# - Installs deps (fallbacks if some optional packages fail)
# - Builds index if missing
# - Launches Streamlit

set -u

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "[info] Working directory: $ROOT_DIR"

# Pick python3 if available
PY=python3
if ! command -v python3 >/dev/null 2>&1; then
  PY=python
fi

# Create venv if missing
if [ ! -d .venv ]; then
  echo "[info] Creating virtualenv (.venv)"
  "$PY" -m venv .venv || {
    echo "[error] Failed to create venv. Ensure Python 3 is installed."; exit 1; }
fi

# Activate venv
source .venv/bin/activate

echo "[info] Python: $(python -V)"
python -m pip install --upgrade pip setuptools wheel >/dev/null

echo "[info] Installing dependencies"
if ! pip install -r requirements.txt; then
  echo "[warn] 'pip install -r requirements.txt' failed. Installing minimal web deps..."
  # Minimal set for the web UI to run
  pip install streamlit ir_datasets torch transformers sentence-transformers tqdm || {
    echo "[error] Failed to install minimal web dependencies."; exit 1; }
  # Optional extras (best-effort)
  pip install pytrec_eval -q || true
  pip install faiss-cpu -q || true
fi

# Ensure data/index exists
if [ ! -f data/index.pkl ]; then
  echo "[info] data/index.pkl not found; preparing data and index"
  if ! python download_data.py; then
    echo "[warn] download_data.py failed. If network is restricted, copy the 'data/' folder from another machine."
  fi
  if [ -f data/documents.tsv ]; then
    if ! python build_index.py; then
      echo "[error] Failed to build index. See logs above."; exit 1;
    fi
  fi
fi

echo "[info] Launching Streamlit (http://localhost:8501)"
exec streamlit run app.py

