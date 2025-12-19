#!/usr/bin/env bash
# Simple helper to build REPORT.tex to PDF (run from CA26 root)
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf REPORT.tex
else
  pdflatex REPORT.tex
  pdflatex REPORT.tex
fi
echo "Built REPORT.pdf"
