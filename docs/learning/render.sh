#!/usr/bin/env bash
# Render a learning-notes HTML file to a PDF next to it, using a headless Chromium browser.
# Usage (from anywhere):  bash docs/learning/render.sh unit4-transformers.html
# If the PDF is ever deleted, this rebuilds it from the HTML.
set -e
cd "$(dirname "$0")"
src="${1:-unit4-transformers.html}"
out="${src%.html}.pdf"

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
  --print-to-pdf="$out" --virtual-time-budget=4000 \
  "file://$PWD/$src" >/dev/null 2>&1
echo "wrote docs/learning/$out  (via $(basename "$browser"))"
