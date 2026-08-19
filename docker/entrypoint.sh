#!/bin/bash
set -euo pipefail

if [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "ERROR: GEMINI_API_KEY is required (pass -e GEMINI_API_KEY=... or use docker compose with .env)"
  exit 1
fi

exec streamlit run streamlit_app.py \
  --server.address=0.0.0.0 \
  --server.port=8501 \
  --server.headless=true
