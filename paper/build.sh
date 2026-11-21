#!/usr/bin/env bash
# Build the paper from Markdown sources with pandoc (>= 3.0).
# Outputs paper/build/paper.docx and paper/build/paper.html; also a PDF when a
# LaTeX engine is available.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
SECTIONS=(sections/0*.md sections/9*.md)
COMMON=(metadata.yaml "${SECTIONS[@]}" --citeproc --bibliography=references.bib
        --resource-path=.:.. --number-sections=false)
pandoc "${COMMON[@]}" -o build/paper.docx
pandoc "${COMMON[@]}" --standalone --mathjax --toc -o build/paper.html
if command -v xelatex >/dev/null 2>&1; then
  pandoc "${COMMON[@]}" --pdf-engine=xelatex -V geometry:margin=1in -o build/paper.pdf
fi
echo "built: $(ls build)"
