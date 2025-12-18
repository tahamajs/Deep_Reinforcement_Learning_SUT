#!/usr/bin/env bash
set -euo pipefail

# Simple helper to execute the HW11 notebook and compile the LaTeX report.
# Usage: ./compile_and_run.sh

echo "1) Running the notebook (may take long)..."
jupyter nbconvert --to notebook --execute code/HW11_Meta_Learning_Complete.ipynb \
  --ExecutePreprocessor.timeout=600 --output code/executed.ipynb

echo "2) Converting executed notebook to HTML"
jupyter nbconvert --to html code/executed.ipynb --output code/executed.html

echo "3) Compiling LaTeX report (if pdflatex/latexmk available)"
cd Homework-11-Template
if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -quiet main.tex
else
  if command -v pdflatex >/dev/null 2>&1; then
    pdflatex -interaction=nonstopmode main.tex
    pdflatex -interaction=nonstopmode main.tex
  else
    echo "pdflatex not found. Please install MacTeX (macOS) or TeX Live and rerun."
    exit 0
  fi
fi

echo "Done. HTML/PDF outputs (if produced) are in the homework folder."
