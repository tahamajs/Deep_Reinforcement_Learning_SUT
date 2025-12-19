# Makefile for CA17 report build
REPORT=REPORT.tex
PDF=$(REPORT:.tex=.pdf)

.PHONY: report clean

report:
	@echo "Building $(PDF)"
	latexmk -pdf -silent $(REPORT)

clean:
	latexmk -c
	@echo "cleaned"
