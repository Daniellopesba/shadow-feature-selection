#!/usr/bin/env bash
# Build the paper from Markdown sources with pandoc (>= 3.0).
# Outputs paper/build/paper.docx and paper/build/paper.html; also a PDF when a
# LaTeX engine (xelatex) or Node with Playwright is available.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
SECTIONS=(sections/0*.md sections/90_references.md sections/91_appendix_a.md sections/92_appendix_b.md sections/93_appendix_c.md)
COMMON=(metadata.yaml "${SECTIONS[@]}" --citeproc --bibliography=references.bib
        --resource-path=.:..)
pandoc "${COMMON[@]}" -o build/paper.docx
pandoc "${COMMON[@]}" --standalone --embed-resources --mathml --toc -o build/paper.html
if command -v xelatex >/dev/null 2>&1; then
  pandoc "${COMMON[@]}" --pdf-engine=xelatex -V geometry:margin=1in -o build/paper.pdf
elif command -v node >/dev/null 2>&1 && node -e "require('playwright')" 2>/dev/null; then
  node html_to_pdf.js   # headless Chromium; renders the MathML in paper.html
fi
echo "built: $(ls build)"
