# Makefile for CA12: build report and generate figures
.PHONY: all report figs clean

all: figs report

figs:
	python paperAssignments/Assignments1-50/CA12/analysis/plot_results.py \
		--ckpt_dir outputs/ca12_checkpoints \
		--out_dir paperAssignments/Assignments1-50/CA12/pictures

report:
	# Build the PDF with latexmk if available; fall back to pdflatex/bibtex
	if command -v latexmk >/dev/null 2>&1; then \
		( cd paperAssignments/Assignments1-50/CA12 && latexmk -pdf -interaction=nonstopmode report.tex ); \
	else \
		( cd paperAssignments/Assignments1-50/CA12 && pdflatex -interaction=nonstopmode report.tex && bibtex report && pdflatex -interaction=nonstopmode report.tex && pdflatex -interaction=nonstopmode report.tex ); \
	fi

test:
	python -m pytest tests -q

clean:
	( cd paperAssignments/Assignments1-50/CA12 && latexmk -C ) || true

echo:
	@echo "Targets: all, figs, report, test, clean"
