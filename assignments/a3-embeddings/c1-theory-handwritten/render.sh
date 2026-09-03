#!/usr/bin/env bash
# Regenerate DRAFT.pdf from DRAFT.html using headless Chrome.
# Run from this directory:  bash render.sh
set -e
cd "$(dirname "$0")"
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="DRAFT.pdf" --virtual-time-budget=4000 \
  "file://$PWD/DRAFT.html" >/dev/null 2>&1
echo "wrote DRAFT.pdf"
