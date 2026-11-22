#!/usr/bin/env bash
# Build the paper with pdflatex + bibtex (TeX Live 2023 or later).
#   paper.pdf                    preprint version with author and repository link
#   build/paper_anonymous.pdf    anonymised TMLR submission version
# Tables and numbers in generated/ come from experiments/paper_numbers.py.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex >/dev/null || { grep -n "^!" build/main.log; exit 1; }
cp build/main.pdf paper.pdf
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build -jobname=paper_anonymous \
  -usepretex='\def\anonymous{1}' main.tex >/dev/null
echo "built: paper.pdf, build/paper_anonymous.pdf"
