#!/usr/bin/env bash
set -euo pipefail

# Build paper.pdf from paper.tex using latexmk (preferred) or pdflatex + bibtex fallback
if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf paper.tex
else
  pdflatex paper.tex
  bibtex paper.aux || true
  pdflatex paper.tex
  pdflatex paper.tex
fi

echo "paper.pdf is generated in the repository root (CA27)."