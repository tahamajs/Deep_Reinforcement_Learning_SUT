#!/usr/bin/env bash
set -euo pipefail

# Simple helper to compile the LaTeX report (report.tex -> report.pdf)
# Requires: pdflatex, bibtex (or latexmk)
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

pdflatex -interaction=nonstopmode report.tex
bibtex report || true
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex

echo "Report compiled: $(pwd)/report.pdf"