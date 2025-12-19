#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 path/to/report.tex" >&2
  exit 2
fi
TEXFILE="$1"
if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -silent "$TEXFILE"
else
  pdflatex -interaction=nonstopmode -halt-on-error "$TEXFILE"
  bibtex "${TEXFILE%.tex}"
  pdflatex -interaction=nonstopmode -halt-on-error "$TEXFILE"
  pdflatex -interaction=nonstopmode -halt-on-error "$TEXFILE"
fi
echo "Built ${TEXFILE%.tex}.pdf"
