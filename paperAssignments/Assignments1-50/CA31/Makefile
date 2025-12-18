# Makefile for building the LaTeX report and converting SVGs
SVG_DIR = results/figures
SVGS = $(wildcard $(SVG_DIR)/*.svg)
PDFS = $(patsubst %.svg,%.pdf,$(SVGS))
REPORT = report.pdf

.PHONY: report convert_svgs clean

# Default target: build report (converts svgs first)
report: $(PDFS)
	@echo "Building $(REPORT)..."
	pdflatex -interaction=nonstopmode report.tex >/dev/null || pdflatex report.tex
	pdflatex -interaction=nonstopmode report.tex
	@echo "Built $(REPORT)"

# Convert SVGs to PDF using rsvg-convert (or inkscape if available)
$(SVG_DIR)/%.pdf: $(SVG_DIR)/%.svg
	@command -v rsvg-convert >/dev/null 2>&1 && rsvg-convert -f pdf -o $@ $< || \
	 (command -v inkscape >/dev/null 2>&1 && inkscape $< --export-filename=$@) || \
	 (echo "Warning: rsvg-convert or inkscape not found; skipping SVG -> PDF conversion" && exit 0)

convert_svgs: $(PDFS)
	@echo "Converted SVGs to PDF"

clean:
	rm -f *.aux *.log *.out *.toc *.pdf $(SVG_DIR)/*.pdf
	@echo "Cleaned build artifacts"
