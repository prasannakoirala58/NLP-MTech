#!/usr/bin/env bash
# Regenerate DRAFT.pdf from DRAFT.html using a headless Chromium browser.
# Run from anywhere:  bash assignments/a5-transformers/render.sh
set -e
cd "$(dirname "$0")"

# Any Chromium-based browser prints PDFs the same way. Use the first one installed.
browser=""
for candidate in \
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser" \
  "/Applications/Chromium.app/Contents/MacOS/Chromium"; do
  if [ -x "$candidate" ]; then browser="$candidate"; break; fi
done
if [ -z "$browser" ]; then echo "No Chrome / Brave / Chromium found in /Applications" >&2; exit 1; fi

"$browser" --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="DRAFT.pdf" --virtual-time-budget=4000 \
  "file://$PWD/DRAFT.html" >/dev/null 2>&1
echo "wrote DRAFT.pdf  (via $(basename "$browser"))"
